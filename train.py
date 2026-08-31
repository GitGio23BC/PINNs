import logging
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import create_graph, create_mesh
from src.loader import generate_ground_truth
from src.loss import bc_loss, mse, r_loss
from src.models import MeshGraphNet
from src.physics import (
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import CSVLogger, init_logging, load_config, set_seed


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
        output_dir / "metrics.csv",
        fieldnames=[
            "step",
            "epoch",
            "total_loss",
            "loss_data",
            "loss_haslach",
            "loss_pako",
            "loss_bc_base",
            "loss_bc_tip",
        ],
    )

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)
    time_grid = dataset["time"]
    u_exact_traj = dataset["u"]
    F_trajectory = dataset["F_applied"]

    # Mesh
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        nx=nx,
        ny=ny,
        device=device,
    )

    num_nodes = mesh.n_nodes
    boundary_indices = mesh.boundary_nodes

    tol = 1e-7
    base_mask = mesh.nodes[boundary_indices, 0] <= (x_min + tol)
    tip_mask = mesh.nodes[boundary_indices, 0] >= (x_max - tol)

    base_nodes_idx = boundary_indices[base_mask]
    tip_nodes_idx = boundary_indices[tip_mask]

    # Model
    mgn = MeshGraphNet(
        node_in_dim=int(cfg["model"]["node_in_dim"]),
        edge_in_dim=int(cfg["model"]["edge_in_dim"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        output_dim=5,
    ).to(device)
    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants
    p_cfg = cfg["physics"]
    A = float(p_cfg["area"])
    k_relax = float(p_cfg["k_relax"])
    c_iso = float(p_cfg["c"])
    c1 = float(p_cfg["c1"])
    c2 = float(p_cfg["c2"])
    c3 = float(p_cfg["c3"])
    b = torch.tensor(
        p_cfg["body_force"],
        device=device,
        dtype=torch.float32,
    )

    # Loss weights
    loss_w = cfg["training"]["loss_weights"]
    w_data = float(loss_w["lambda_data"])
    w_haslach = float(loss_w["lambda_haslach"])
    w_momentum = float(loss_w["lambda_momentum"])
    w_bc_base = float(loss_w["lambda_bc_base"])
    w_bc_tip = float(loss_w["lambda_bc_tip"])

    # Training Parameters
    epochs = int(cfg["training"]["epochs"])
    time_steps = len(time_grid)
    dt = float((time_grid[-1] - time_grid[0]) / max(time_steps - 1, 1))

    # Initialisation
    E_prev_voigt = torch.zeros((num_nodes, 3), device=device, dtype=torch.float32)
    u_prev = torch.zeros((num_nodes, 2), device=device, dtype=torch.float32)

    # Training
    logger.info("Stratring PINN-MSG trainingin...")

    for t_step in range(1, time_steps):
        t_curr = time_grid[t_step].item()
        u_target = u_exact_traj[t_step]
        S_applied = F_trajectory[t_step] / A
        logger.info(f"\n--- Time Step {t_step}/{time_steps} (t = {t_curr:.3f}s) ---")

        for epoch in tqdm(range(1, epochs + 1), desc="Epoch: "):
            optimizer.zero_grad()

            graph = create_graph(mesh=mesh, u=u_prev)
            predictions = mgn(graph)
            X_ref = graph.mesh_nodes

            u_pred = predictions[:, :2]
            S_pred = predictions[:, 2:]

            # Data Loss
            loss_data = mse(u_pred, u_target)

            # Viscoelastic Loss
            E_current_voigt, haslash_residuals = haslach_constitutive_residual_2D(
                u_pred=u_pred,
                S_pred=S_pred,
                X_ref=X_ref,
                E_prev=E_prev_voigt,
                dt=dt,
                k_relax=k_relax,
                c=c_iso,
                c1=c1,
                c2=c2,
                c3=c3,
            )
            loss_haslach = r_loss(haslash_residuals)

            # Momentum Loss
            pako_residual = pako_residual_2D(
                u_pred=u_pred,
                S_pred=S_pred,
                X_ref=X_ref,
                b=b,
            )
            loss_pako = r_loss(pako_residual)

            # Boundary Loss
            u_base_pred = u_pred[base_nodes_idx]
            loss_bc_base = bc_loss(u_base_pred, torch.zeros_like(u_base_pred))

            S_tip_pred = S_pred[tip_nodes_idx]
            S_tip_applied = S_applied[tip_nodes_idx]
            loss_bc_tip = bc_loss(S_tip_pred, S_tip_applied)

            # Total Loss
            total_loss = (
                w_data * loss_data
                + w_haslach * loss_haslach
                + w_momentum * loss_pako
                + w_bc_base * loss_bc_base
                + w_bc_tip * loss_bc_tip
            )

            total_loss.backward()
            optimizer.step()

            # Checkpoint
            if epoch % 100 == 0 or epoch == epochs:
                metrics = {
                    "step": t_step,
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "loss_data": loss_data.item(),
                    "loss_haslach": loss_haslach.item(),
                    "loss_pako": loss_pako.item(),
                    "loss_bc_base": loss_bc_base.item(),
                    "loss_bc_tip": loss_bc_tip.item(),
                }
                logger.info(
                    f"Checkpoint {' | '.join([str(k) + ': ' + str(v) for k, v in metrics.items()])}"
                )
                metrics_logger.log(metrics)
                torch.save(
                    {
                        "model_state_dict": mgn.state_dict(),
                        "config": cfg,
                    },
                    checkpoint_dir / f"{model_name}_{epoch}{t_step}.pt",
                )

        E_prev_voigt = E_current_voigt.detach()  # type: ignore
        u_prev = u_pred.detach()  # type: ignore
    torch.save(
        {
            "model_state_dict": mgn.state_dict(),
            "config": cfg,
        },
        output_dir / f"{model_name}.pt",
    )
    logger.info("-----------------------------Train Ended-----------------------------")


if __name__ == "__main__":
    train()
