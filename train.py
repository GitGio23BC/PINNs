import logging
from pathlib import Path

import torch
import yaml
from torch.optim import Adam

from src.loader import data_norm, init_n_bound_data
from src.loss import bc_loss, ic_loss, r_loss
from src.models import ARCHITECTURE_REGISTRY
from src.operators import OPERATOR_REGISTRY
from src.utils import CSVLogger, init_logging

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
            "loss_PDE",
            "loss_Dirichlet",
            "loss_Neumann",
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
    device = cfg["training"]["device"]

    if arch not in ARCHITECTURE_REGISTRY:
        logger.error(
            f"{arch} isn't available.\n Available architectures are: {', '.join(ARCHITECTURE_REGISTRY.keys())}"
        )
        raise KeyError("Architecture error")

    pinn = ARCHITECTURE_REGISTRY[arch](
        in_channels, hidden_layers, hidden_dim, out_class
    )

    pinn.to(device)

    optimizer = Adam(
        pinn.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    # Physics Parameters
    eq_name = cfg["physics"]["equation"]
    if eq_name not in OPERATOR_REGISTRY:
        logger.error(
            f"{eq_name} isn't available.\n Available operators are: {', '.join(OPERATOR_REGISTRY.keys())}"
        )
        raise KeyError("Operator error")

    A_min, A_max = cfg["training"]["parameters"]["area_range"]
    E = cfg["training"]["parameters"]["young"]
    F_min, F_max = cfg["training"]["parameters"]["ext_force_range"]
    f0 = cfg["training"]["parameters"]["int_force"]

    x, u_real, ε_real = init_n_bound_data(cfg, device)

    if cfg["data"]["normalisation"]:
        x_hat, scale_factor = data_norm(x)
    else:
        x_hat = x

    n = OPERATOR_REGISTRY[eq_name]

    # Traning Parameters
    λ_ic = float(cfg["training"]["loss_weights"]["lambda_ic"])
    λ_bc = float(cfg["training"]["loss_weights"]["lambda_bc"])
    λ_r = float(cfg["training"]["loss_weights"]["lambda_r"])
    epochs = int(cfg["training"]["epochs"])
    ramp_epochs = int(cfg["training"]["ramp_epochs"])
    load_continuation = bool(cfg["training"]["load_continuation"])

    s = (
        "Starting Curriculum Training..."
        if load_continuation
        else "Starting Training..."
    )
    logger.info(s)

    for epoch in range(epochs):
        pinn.train()
        optimizer.zero_grad()

        A = A_min + (A_max - A_min) * torch.rand(1).item()
        F = F_min + (F_max - F_min) * torch.rand(1).item()

        λ = (
            min(1.0, epoch / ramp_epochs)
            if ramp_epochs > 0 and load_continuation
            else 1.0
        )
        F_curr = λ * F

        x_hat = x_hat.clone().detach().requires_grad_(True)

        pde_residual, u_pred, u_x_pred, _ = n(pinn, x_hat, F_curr, A, E, f0)

        u_dirichlet_pred = u_pred[0]
        initial_u = torch.tensor([0.0], device=device)
        loss_Dirichlet = ic_loss(u_dirichlet_pred, initial_u)

        u_x_neumann_pred = u_x_pred[-1]
        loss_Neumann = bc_loss(
            u_x_neumann_pred, torch.tensor([F_curr / (E * A)], device=device) # target ε
        )

        loss_PDE = r_loss(pde_residual)

        loss = λ_r * loss_PDE + λ_ic * loss_Dirichlet + λ_bc * loss_Neumann

        loss.backward()
        optimizer.step()

        if epoch % (epochs / 10) == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:5d}/{epochs} | λ_force: {λ:.2f} | "
                f"Total Loss: {loss.item():.6e} | PDE: {loss_PDE.item():.6e} | "
                f"Dirichlet: {loss_Dirichlet.item():.6e} | Neumann: {loss_Neumann.item():.6e}"
            )
            torch.save(pinn.state_dict(), checkpoint_dir / f"{model_name}_{epoch}.pt")

        metrics_logger.log(
            {
                "λ_force": λ,
                "epoch": epoch + 1,
                "total_loss": loss.item(),
                "loss_Dirichlet": loss_Dirichlet.item(),
                "loss_Neumann": loss_Neumann.item(),
                "loss_PDE": loss_PDE.item(),
            }
        )

    torch.save(pinn.state_dict(), out_dir / f"{model_name}.pt")
    logger.info("Training ended and model saved.")
