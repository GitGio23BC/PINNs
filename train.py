import logging
from pathlib import Path

import torch
import yaml
from torch.optim import Adam

from src.loader import load_train_data
from src.loss import ic_loss, r_loss
from src.models import BACKBONE_REGISTRY
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

    if arch not in BACKBONE_REGISTRY:
        logger.error(
            f"{arch} isn't available.\n Available architectures are: {', '.join(BACKBONE_REGISTRY.keys())}"
        )
        raise KeyError("Architecture error")

    pinn = BACKBONE_REGISTRY[arch](in_channels, hidden_layers, hidden_dim, out_class)
    pinn.to(device)

    optimizer = Adam(pinn.parameters(), lr=float(cfg["optimizer"]["lr"]))

    # Physics Parameters
    eq_name = cfg["physics"]["equation"]
    k = cfg["physics"]["relaxation_modulus"]
    c = cfg["physics"]["stress_scaling_factor"]
    ct = cfg["physics"]["exponential_non_linearity"]
    y = cfg["physics"]["stress"]
    e0 = cfg["training"]["parameters"]["intial_strain"]

    if eq_name not in OPERATOR_REGISTRY:
        logger.error(
            f"{eq_name} isn't available.\n Available operators are: {', '.join(OPERATOR_REGISTRY.keys())}"
        )
        raise KeyError("Operator error")

    t = load_train_data(cfg, device)[0]

    n = OPERATOR_REGISTRY[eq_name]

    # Traning Parameters
    λ_ic = float(cfg["training"]["loss_weights"]["lambda_ic"])
    λ_r = float(cfg["training"]["loss_weights"]["lambda_r"])
    epochs = int(cfg["training"]["epochs"])

    logger.info("Starting Training...")

    for epoch in range(epochs):
        pinn.train()
        optimizer.zero_grad()

        pde_residual, E = n(pinn, t, k, ct, c, y)

        E_dirichlet_pred = E[0]
        E0 = torch.tensor([float(e0)], device=device)
        loss_Dirichlet = ic_loss(E_dirichlet_pred, E0)

        loss_PDE = r_loss(pde_residual)

        # loss_data = mse(u_sensor_pred, u_sensor) # Maybe I can add also a data loss

        loss = λ_r * loss_PDE + λ_ic * loss_Dirichlet  # + loss_data

        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:5d}/{epochs} |"
                f"Total Loss: {loss.item():.6e} | PDE: {loss_PDE.item():.6e} | "
                f"Dirichlet: {loss_Dirichlet.item():.6e} | Neumann: {None}  | "
                f"Output E: {E[-1].item():.4f}"
            )
            torch.save(
                pinn.state_dict(),
                checkpoint_dir / f"{model_name}_{epoch}.pt",
            )

        metrics_logger.log(
            {
                "epoch": epoch + 1,
                "total_loss": loss.item(),
                "loss_Dirichlet": loss_Dirichlet.item(),
                "loss_Neumann": 0.0,
                "loss_PDE": loss_PDE.item(),
            }
        )

    torch.save(
        pinn.state_dict(),
        out_dir / f"{model_name}.pt",
    )
    logger.info("Training ended and model saved.")
