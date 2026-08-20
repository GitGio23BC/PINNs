import logging
from pathlib import Path

import torch
import yaml
from torch.optim import Adam

from src.loader import init_n_bound_data
from src.loss import bc_loss, ic_loss, r_loss
from src.loss.loss import mse
from src.models import BACKBONE_REGISTRY, ParametricPINN
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
    _, x_ref, _ = cfg["training"]["domain"]
    _, F_ref = cfg["training"]["parameters"]["ext_force_range"]
    _, A_ref = cfg["training"]["parameters"]["area_range"]

    if arch not in BACKBONE_REGISTRY:
        logger.error(
            f"{arch} isn't available.\n Available architectures are: {', '.join(BACKBONE_REGISTRY.keys())}"
        )
        raise KeyError("Architecture error")

    backbone = BACKBONE_REGISTRY[arch](
        in_channels, hidden_layers, hidden_dim, out_class
    )
    pinn = ParametricPINN(backbone, x_ref, F_ref, A_ref)
    pinn.to(device)

    E_learned = torch.nn.Parameter(
        torch.tensor([1.0], requires_grad=True, device=device)
    )
    optimizer = Adam([
    {"params": pinn.parameters(), "lr": float(cfg["optimizer"]["lr"])},
    {"params": [E_learned], "lr": 1e-1}  # Faster parameter convergence
])

    """optimizer = Adam(
        pinn.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )"""

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

        x = x.clone().detach().requires_grad_(True)
        pde_residual, u_pred, u_x_pred, _ = n(pinn, x, F_curr, A, E_learned, f0)
        E_discovered = E_learned.item()

        u_dirichlet_pred = u_pred[0]
        initial_u = torch.tensor([0.0], device=device)
        loss_Dirichlet = ic_loss(u_dirichlet_pred, initial_u)

        u_x_neumann_pred = u_x_pred[-1]
        loss_Neumann = bc_loss(
            u_x_neumann_pred,
            F_curr / (E_learned * A),  # target ε
        )

        loss_PDE = r_loss(pde_residual)

        idx_sensor = torch.randperm(len(x))[:100]
        x_sensor = x[idx_sensor]
        u_sensor = (
            1 / (A * E) * ((F_curr + f0 * x_ref) * x_sensor - f0 / 2 * x_sensor**2)
        ) + 0.01 * torch.rand_like(x_sensor)
        u_sensor_pred = u_pred[idx_sensor]

        loss_data = mse(u_sensor_pred, u_sensor)

        loss = loss_data + λ_r * loss_PDE + λ_ic * loss_Dirichlet + λ_bc * loss_Neumann

        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:5d}/{epochs} | λ_force: {λ:.2f} | "
                f"Total Loss: {loss.item():.6e} | PDE: {loss_PDE.item():.6e} | "
                f"Dirichlet: {loss_Dirichlet.item():.6e} | Neumann: {loss_Neumann.item():.6e}  | "
                f"Discovered E: {E_discovered:.4f} Pa (True: {E:.4f} Pa)"
            )
            torch.save(
                {
                    "model_state_dict": pinn.state_dict(),
                    "E_learned": E_learned.item(),
                },
                checkpoint_dir / f"{model_name}_{epoch}.pt",
            )

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

    torch.save(
        {
            "model_state_dict": pinn.state_dict(),
            "E_learned": E_learned.item(),
        },
        out_dir / f"{model_name}.pt",
    )
    logger.info("Training ended and model saved.")
