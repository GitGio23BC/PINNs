import logging
from pathlib import Path

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.loader import boundary_points, initial_points, train_points
from src.loss import bc_loss, ic_loss, r_loss, traction_bc_loss
from src.models.architectures import ParametricPINN
from src.models.backbones import BACKBONE_REGISTRY
from src.physics import OPERATOR_REGISTRY
from src.utils import CSVLogger, init_logging, set_seed

CONFIG_PATH = Path("config.yaml")

if __name__ == "__main__":
    # Log & Config
    init_logging()
    logger = logging.getLogger(__name__)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded. Config file path: {CONFIG_PATH.absolute()}")

    metrics_logger = CSVLogger(
        filepath=Path(cfg["output_dir"]) / "metrics.csv",
        fieldnames=[
            "λ_force",
            "epoch",
            "total_loss",
            "loss_momentum",
            "loss_bc",
            "loss_ic",
            "learning_rate",
        ],
    )

    # Folder structure
    out_dir = Path(cfg["output_dir"])
    checkpoint_dir = out_dir / "checkpoints" / cfg["model"]["model_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Model COnfig
    model_name = cfg["model"]["model_name"]
    arch = cfg["model"]["architecture"]
    in_channels = cfg["model"]["in_channels"]
    hidden_layers = cfg["model"]["hidden_layers"]
    hidden_dim = cfg["model"]["hidden_dim"]
    out_class = cfg["model"]["out_class"]

    backbone = BACKBONE_REGISTRY[arch](
        in_channels, hidden_layers, hidden_dim, out_class
    )
    pinn = ParametricPINN(
        backbone=backbone,
        x_ref=cfg.get("domain", {}).get("x_range", 1.0)[-1],
        t_ref=cfg.get("domain", {}).get("t_range", 1.0)[-1],
        F_ref=cfg.get("domain", {}).get("F_range", 10.0)[-1],
    ).to(device)

    optimizer = Adam(pinn.parameters(), lr=float(cfg["optimizer"]["lr"]))
    epochs = int(cfg["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    eq_name = cfg["physics"]["equation"]
    k = float(cfg["physics"]["relaxation_modulus"])
    c = float(cfg["physics"]["stress_scaling_factor"])
    c1, c2, c3 = [float(val) for val in cfg["physics"]["c_constants"]]
    b_val = cfg["physics"].get("body_force", [0.0, 0.0])

    pde_operator = OPERATOR_REGISTRY[eq_name]

    w_momentum = float(cfg["training"]["loss_weights"]["lambda_momentum"])
    w_bc = float(cfg["training"]["loss_weights"]["lambda_bc"])
    w_ic = float(cfg["training"]["loss_weights"]["lambda_ic"])

    logger.info("Starting Parametric Temporal Training...")

    for epoch in range(epochs):
        pinn.train()
        optimizer.zero_grad()

        # Domain Pass
        x, t, F = train_points(cfg, device)
        n_pts = x.shape[0]
        b = torch.tensor(b_val, dtype=torch.float32, device=device).expand(n_pts, 2)

        u, P, res_momentum = pde_operator(pinn, x, t, F, k, c1, c2, c3, c, b)
        loss_momentum = r_loss(res_momentum)

        # Initial Condition
        x_ic, t_ic, F_ic = initial_points(cfg, device)
        u_ic_pred = pinn(x_ic, t_ic, F_ic)
        loss_ic = ic_loss(u_ic_pred, torch.zeros_like(u_ic_pred))

        # Boundary Conditions
        left, right, bottom, top = boundary_points(cfg, device)
        x_l, t_l, F_l = left
        x_r, t_r, F_r = right
        x_b, t_b, F_b = bottom
        x_top, t_top, F_top = top

        u_left_pred = pinn(x_l, t_l, F_l)
        loss_bc_left = bc_loss(u_left_pred, torch.zeros_like(u_left_pred))

        u_bottom_pred = pinn(x_b, t_b, F_b)
        loss_bc_bottom = bc_loss(
            u_bottom_pred[:, 1:2], torch.zeros_like(u_bottom_pred[:, 1:2])
        )

        _, P_top, _ = pde_operator(
            pinn, x_top, t_top, F_top, k, c1, c2, c3, c, torch.zeros_like(x_top)
        )
        n_top = torch.tensor([0.0, 1.0], device=device).expand(x_top.shape[0], 2)
        loss_bc_top = traction_bc_loss(P_top, n_top, torch.zeros_like(x_top))

        _, P_right, _ = pde_operator(
            pinn, x_r, t_r, F_r, k, c1, c2, c3, c, torch.zeros_like(x_r)
        )
        n_right = torch.tensor([1.0, 0.0], device=device).expand(x_r.shape[0], 2)
        traction_target_right = torch.cat([F_r, torch.zeros_like(F_r)], dim=-1)
        loss_bc_right = traction_bc_loss(P_right, n_right, traction_target_right)

        loss_bc = loss_bc_left + loss_bc_bottom + loss_bc_top + loss_bc_right

        # Total Objective
        loss = (w_momentum * loss_momentum) + (w_bc * loss_bc) + (w_ic * loss_ic)

        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:5d}/{epochs} | "
                f"Total: {loss.item():.4e} | "
                f"Momentum: {loss_momentum.item():.4e} | "
                f"BC: {loss_bc.item():.4e} | "
                f"IC: {loss_ic.item():.4e}"
            )


        metrics_logger.log(
            {
                "epoch": epoch + 1,
                "total_loss": loss.item(),
                "loss_momentum": loss_momentum.item(),
                "loss_bc": loss_bc.item(),
                "loss_ic": loss_ic.item(),
            }
        )

    torch.save(pinn.state_dict(), out_dir / f"{model_name}.pt")
    logger.info("Parametric temporal PINN training completed and model saved.")
