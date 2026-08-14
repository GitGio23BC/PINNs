import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.loader import test_data
from src.loss import mse
from src.models import ARCHITECTURE_REGISTRY
from src.operators import OPERATOR_REGISTRY
from src.utils import init_logging

CONFIG_PATH = Path("config.yaml")


def test(model_path: Path | None = None, config_path: Path = CONFIG_PATH):
    init_logging()
    logger = logging.getLogger(__name__)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    in_channels = cfg["model"]["in_channels"]
    hidden_layers = cfg["model"]["hidden_layers"]
    hidden_dim = cfg["model"]["hidden_dim"]
    out_class = cfg["model"]["out_class"]
    arch = cfg["model"]["architecture"]
    eq_name = cfg["physics"]["equation"]

    if model_path is None:
        model_path = Path(cfg["output_dir"]) / f"{cfg['model']['model_name']}.pt"

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Train the model first.")
        raise FileNotFoundError("Model error")

    try:
        pinn = ARCHITECTURE_REGISTRY[arch](
            in_channels, hidden_layers, hidden_dim, out_class
        ).to(device)
        pinn.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        pinn = ARCHITECTURE_REGISTRY["MLP"](
            in_channels, hidden_layers, hidden_dim, out_class
        ).to(device)
        pinn.load_state_dict(torch.load(model_path, map_location=device))
    pinn.eval()
    logger.info(f"Loaded trained {arch} model from {model_path}")

    # Physics Parameters
    x_test, u_exact, ε_exact, F, A, E, f0 = test_data(cfg, device)

    x_test = x_test.clone().detach().requires_grad_(True)

    pde_operator = OPERATOR_REGISTRY[eq_name]
    _, u_pred, u_x_pred, _ = pde_operator(pinn, x_test, F, A, E, f0)

    mse_u = mse(u_pred, u_exact).item()
    mse_ε = mse(u_x_pred, ε_exact).item()

    logger.info(f"Test Set Displacement MSE: {mse_u:.6e}")
    logger.info(f"Test Set Strain MSE: {mse_ε:.6e}")

    # Plotting
    x_np = x_test.detach().cpu().numpy().flatten()
    u_exact_np = u_exact.detach().cpu().numpy().flatten()
    u_pred_np = u_pred.detach().cpu().numpy().flatten()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_np, u_exact_np, "k--", label="Exact Solution", linewidth=2)
    plt.plot(x_np, u_pred_np, "r-", label="PINN Prediction", linewidth=1.5)
    plt.title(f"Displacement u(x) | MSE: {mse_u:.2e}")
    plt.xlabel("Position x")
    plt.ylabel("Displacement u")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        x_np,
        ε_exact.detach().cpu().numpy().flatten(),
        "k--",
        label="Exact Strain",
        linewidth=2,
    )
    plt.plot(
        x_np,
        u_x_pred.detach().cpu().numpy().flatten(),
        "b-",
        label="PINN Strain",
        linewidth=1.5,
    )
    plt.title(f"Strain du/dx | MSE: {mse_ε:.2e}")
    plt.xlabel("Position x")
    plt.ylabel("Strain ε")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(cfg["output_dir"]) / f"{model_path.stem}_results.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved test evaluation plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])

    for model_path in out_dir.glob("*.pt"):
        test(model_path)
