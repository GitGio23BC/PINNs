import logging
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import create_mesh
from src.loader import PINNSampler, generate_ground_truth
from src.loss import bc_loss, mse, r_loss, traction_bc_loss
from src.models import BACKBONE_REGISTRY, ParametricPINN
from src.physics import (
    HUGO,
    HolzapfelEnergy_2D,
    Kinematics,
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import CSVLogger, grad, init_logging, load_config, set_seed
from src.utils.tensor_tools import voigt_to_tensor


def train():
    # Logging and set-up
    init_logging()
    cfg = load_config("config.yaml")
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["model_name"]
    checkpoint_dir = Path(output_dir / "checkpoints" / model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
            "epoch",
            "step_loss",
            "loss_data",
            "loss_haslach",
            "loss_pako",
            "loss_bc_base",
            "loss_bc_tip",
        ],
    )

    # Mesh
    x_range = cfg["domain"]["x_range"]
    y_range = cfg["domain"]["y_range"]
    t_range = cfg["domain"]["t_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_range[-1] - x_range[0]),
        height=float(y_range[-1] - y_range[0]),
        nx=nx,
        ny=ny,
        device=device,
    )

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    sampler = PINNSampler(dataset=dataset, cfg=cfg, mesh=mesh, device=device)

    # Model
    backbone = BACKBONE_REGISTRY[cfg["model"]["architecture"]](
        in_dim=3,
        hidden_layers=int(cfg["model"]["num_layers"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        out_dim=5,
    )
    pinn = ParametricPINN(
        backbone=backbone,
        x_range=x_range,
        y_range=y_range,
        t_range=t_range,
    ).to(device)
    optimizer = Adam(
        pinn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants and Models
    p_cfg = cfg["physics"]
    visco_model = HUGO(
        HolzapfelEnergy_2D(
            c=float(p_cfg["c"]),
            c1=float(p_cfg["c1"]),
            c2=float(p_cfg["c2"]),
            c3=float(p_cfg["c3"]),
            device=device,
        ),
        k_relax=float(p_cfg["k_relax"]),
    )
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float32)

    # Loss weights
    loss_w = cfg["training"]["loss_weights"]
    w_data = float(loss_w["lambda_data"])
    w_haslach = float(loss_w["lambda_haslach"])
    w_momentum = float(loss_w["lambda_momentum"])
    #w_initial = float(loss_w["lambda_initial"])
    #w_bc_base = float(loss_w["lambda_bc_base"])
    w_bc_tip = float(loss_w["lambda_bc_tip"])

    # Training Parameters
    epochs = int(cfg["training"]["epochs"])
    time_steps = len(dataset["time"])

    # Training
    logger.info("Stratring PINN-MSG trainingin...")

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch: "):
        # Initialisation
        optimizer.zero_grad()

        # Sampling
        batch = sampler.sample_window(step_start=0, step_end=time_steps)

        preds_res = pinn(t=batch.t_res, X=batch.X_res)
        u_pred_res = preds_res[:, :2]
        S_pred_res = preds_res[:, 2:]

        # Data Loss
        u_exact_flat = dataset["u"].reshape(-1, 2)
        #S_exact_flat = dataset["S"].reshape(-1, 3)
        loss_data = mse(u_pred_res, u_exact_flat) #+ mse(S_pred_res, S_exact_flat)

        # Viscoelastic Loss
        haslash_residuals = haslach_constitutive_residual_2D(
            u_pred=u_pred_res,
            S_pred=S_pred_res,
            X_ref=batch.X_res,
            t=batch.t_res,
            visco_model=visco_model,
        )
        loss_haslach = r_loss(haslash_residuals)

        # Momentum Loss
        pako_residual = pako_residual_2D(
            u_pred=u_pred_res,
            S_pred=S_pred_res,
            X_ref=batch.X_res,
            b=b,
        )
        loss_pako = r_loss(pako_residual)

        # Initial Loss
        #preds_ic = pinn(t=batch.t_ic, X=batch.X_ic)
        #u_ic_pred = preds_ic[:, :2]
        #S_ic_pred = preds_ic[:, 2:]
        #loss_ic = mse(u_ic_pred, batch.u_ic_target) + mse(S_ic_pred, batch.S_ic_target)

        # Boundary Loss
        #preds_base = pinn(t=batch.t_base, X=batch.X_base)
        #u_base = preds_base[:, :2]
        #loss_bc_base = bc_loss(u_base, torch.zeros_like(preds_base[:, :2]))

        preds_tip = pinn(t=batch.t_neu, X=batch.X_neu)
        u_tip = preds_tip[:, :2]
        S_tip = preds_tip[:, 2:]

        kin = Kinematics(grad(u_tip, batch.X_neu))
        S_tip_voigt = voigt_to_tensor(S_tip, is_shear=False)
        P_tip = kin.compute_P(S_tip_voigt)
        loss_bc_tip = traction_bc_loss(P_tip, batch.normals_neu, batch.trac_target)

        # Total Loss
        step_loss = (
            w_data * loss_data
            + w_haslach * loss_haslach
            + w_momentum * loss_pako
        #    + w_initial * loss_ic
        #    + w_bc_base * loss_bc_base
            + w_bc_tip * loss_bc_tip
        )

        step_loss.backward()
        optimizer.step()

        # Checkpoint
        metrics = {
            "epoch": epoch,
            "step_loss": step_loss.detach().item(),
            "loss_data": loss_data.detach().item(),
            "loss_haslach": loss_haslach.detach().item(),
            "loss_pako": loss_pako.detach().item(),
            #"loss_initial": loss_initial.detach().item(),
            #"loss_bc_base": loss_bc_base.detach().item(),
            "loss_bc_tip": loss_bc_tip.detach().item(),
        }

        metrics_logger.log(metrics)

        if epoch % 50 == 0:
            torch.save(
                {
                    "model_state_dict": pinn.state_dict(),
                    "config": cfg,
                },
                checkpoint_dir / f"{model_name}_{epoch}.pt",
            )

    torch.save(
        {
            "model_state_dict": pinn.state_dict(),
            "config": cfg,
        },
        output_dir / f"{model_name}.pt",
    )
    logger.info("-----------------------------Train Ended-----------------------------")


if __name__ == "__main__":
    train()
