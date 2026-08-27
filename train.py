import logging
from pathlib import Path

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.loader import (
    boundary_points,
    train_points,
)
from src.loss import bc_loss, r_loss
from src.models.backbones import BACKBONE_REGISTRY
from src.physics import OPERATOR_REGISTRY
from src.utils import CSVLogger, init_logging, set_seed

CONFIG_PATH = Path("config.yaml")

if __name__ == "__main__":
    init_logging()
    logger = logging.getLogger(__name__)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from: {CONFIG_PATH.absolute()}")

    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    checkpoint_dir = out_dir / "checkpoints" / cfg["model"]["model_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = CSVLogger(
        filepath=out_dir / "metrics.csv",
        fieldnames=[
            "epoch",
            "total_loss",
            "loss_haslach",
            "loss_momentum",
            "loss_bc",
            "learning_rate",
        ],
    )

    device = torch.device(cfg["training"]["device"])

    model_name = cfg["model"]["model_name"]
    arch = cfg["model"]["architecture"]
    in_channels = cfg["model"]["in_channels"]
    hidden_layers = cfg["model"]["hidden_layers"]
    hidden_dim = cfg["model"]["hidden_dim"]
    out_class = cfg["model"]["out_class"]

    pinn = BACKBONE_REGISTRY[arch](
        in_channels, hidden_layers, hidden_dim, out_class
    ).to(device)
    optimizer = Adam(pinn.parameters(), lr=float(cfg["optimizer"]["lr"]))

    epochs = int(cfg["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    eq_name = cfg["physics"]["equation"]
    k = float(cfg["physics"]["relaxation_modulus"])
    c = float(cfg["physics"]["stress_scaling_factor"])
    c1, c2, c3 = [float(val) for val in cfg["physics"]["c_constants"]]
    b_val = cfg["physics"].get("body_force", [0.0, 0.0])
    s_val = cfg["physics"]["stress"]

    pde_operator = OPERATOR_REGISTRY[eq_name]

    w_haslach = float(cfg["training"]["loss_weights"]["lambda_haslach"])
    w_momentum = float(cfg["training"]["loss_weights"]["lambda_momentum"])
    w_bc = float(cfg["training"]["loss_weights"].get("lambda_bc", 10.0))

    logger.info("Starting Training...")

    for epoch in range(1, epochs + 1):
        pinn.train()
        optimizer.zero_grad()

        x = boundary_points(cfg, device)
        if isinstance(x, tuple):
            x = x[0]

        n_pts = x.shape[0]
        b = torch.tensor(b_val, dtype=torch.float32, device=device).expand(n_pts, 2)
        S = torch.tensor(s_val, dtype=torch.float32, device=device).expand(
            n_pts, len(s_val)
        )

        u, res_haslach, res_momentum = pde_operator(pinn, x, k, c1, c2, c3, c, b, S)

        loss_haslach = r_loss(res_haslach)
        loss_momentum = r_loss(res_momentum)

        x_left, x_right, x_bottom, _ = boundary_points(cfg, device)
        n_bc_segment = x_left.shape[0]

        u_left_pred = pinn(x_left)
        u_right_pred = pinn(x_right)
        u_bottom_pred = pinn(x_bottom)

        loss_bc_left = bc_loss(
            u_left_pred[:, 0:1], torch.zeros((n_bc_segment, 1), device=device)
        )

        loss_bc_right = bc_loss(
            u_right_pred[:, 0:1], torch.full((n_bc_segment, 1), 0.10, device=device)
        )

        loss_bc_bottom = bc_loss(
            u_bottom_pred[:, 1:2], torch.zeros((n_bc_segment, 1), device=device)
        )

        loss_bc = loss_bc_left + loss_bc_right + loss_bc_bottom

        loss = (
            (w_haslach * loss_haslach)
            + (w_momentum * loss_momentum)
            + (w_bc * loss_bc)
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        if epoch % (epochs // 10) == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:5d}/{epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Total: {loss.item():.4e} | "
                f"Haslach: {loss_haslach.item():.4e} | "
                f"Momentum: {loss_momentum.item():.4e} | "
                f"BC: {loss_bc.item():.4e}"
            )
            torch.save(pinn.state_dict(), checkpoint_dir / f"{model_name}_{epoch}.pt")

        metrics_logger.log(
            {
                "epoch": epoch,
                "total_loss": loss.item(),
                "loss_haslach": loss_haslach.item(),
                "loss_momentum": loss_momentum.item(),
                "loss_bc": loss_bc.item(),
                "learning_rate": current_lr,
            }
        )

    torch.save(pinn.state_dict(), out_dir / f"{model_name}.pt")
    logger.info("Static training completed and final model saved.")
