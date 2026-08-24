import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from src.loader import test_data
from src.loss import rmse
from src.models import BACKBONE_REGISTRY
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
        raise FileNotFoundError("Model file missing.")

    if arch not in BACKBONE_REGISTRY:
        logger.error(f"Architecture {arch} not found in BACKBONE_REGISTRY.")
        raise KeyError("Architecture error")

    pinn = BACKBONE_REGISTRY[arch](
        in_channels, hidden_layers, hidden_dim, out_class
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        pinn.load_state_dict(state_dict["model_state_dict"])
    else:
        pinn.load_state_dict(state_dict)

    pinn.eval()
    logger.info(f"Loaded trained {arch} model from {model_path}")

    # Physics Constants
    k = float(cfg["physics"]["relaxation_modulus"])
    c = float(cfg["physics"]["stress_scaling_factor"])
    c1, c2, c3 = [float(v) for v in cfg["physics"]["c_constants"]]
    y_tensor = torch.tensor(
        cfg["physics"]["stress"], dtype=torch.float32, device=device
    )

    t_test, E_exact = test_data(cfg, device)
    t_test = t_test.clone().detach().requires_grad_(True)

    pde_operator = OPERATOR_REGISTRY[eq_name]
    residual, E_pred = pde_operator(pinn, t_test, k, c1, c2, c3, c, y_tensor)

    E1_t = torch.autograd.grad(
        E_pred[:, 0],
        t_test,
        grad_outputs=torch.ones_like(E_pred[:, 0]),
        retain_graph=True,
    )[0]

    E2_t = torch.autograd.grad(
        E_pred[:, 1],
        t_test,
        grad_outputs=torch.ones_like(E_pred[:, 1]),
        create_graph=False,
    )[0]

    # Shape: (N, 2)
    E_t_pred = torch.cat([E1_t, E2_t], dim=-1)

    error_E1 = rmse(E_pred[:, 0:1], E_exact[:, 0:1]).item()
    error_E2 = rmse(E_pred[:, 1:2], E_exact[:, 1:2]).item()
    total_error = rmse(E_pred, E_exact).item()
    mean_res = torch.mean(torch.abs(residual)).item()

    logger.info("=" * 55)
    logger.info(f"Evaluation Results for: {model_path.name}")
    logger.info(f"Total Vector L2 Error (E):    {total_error:.6e}")
    logger.info(f"Relative L2 Error Strain E1:  {error_E1:.6e}")
    logger.info(f"Relative L2 Error Strain E2:  {error_E2:.6e}")
    logger.info(f"Mean Absolute ODE Residual:   {mean_res:.6e}")
    logger.info(
        f"Initial Strain E(0) Pred:     [{E_pred[0, 0].item():.4e}, {E_pred[0, 1].item():.4e}]"
    )
    logger.info(
        f"Final Asymptotic Strain E(T): [{E_pred[-1, 0].item():.4f}, {E_pred[-1, 1].item():.4f}]"
    )
    logger.info("=" * 55)

    # Plotting
    t_np = t_test.detach().cpu().numpy().flatten()
    E_exact_np = E_exact.detach().cpu().numpy()
    E_pred_np = E_pred.detach().cpu().numpy()
    E_t_pred_np = E_t_pred.detach().cpu().numpy()

    _fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # Panel 1: Training Loss Convergence
    metrics_path = Path(cfg["output_dir"]) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        axes[0].plot(
            df["epoch"], df["total_loss"], "k-", label="Total Loss", linewidth=1.5
        )
        if "loss_PDE" in df.columns:
            axes[0].plot(
                df["epoch"], df["loss_PDE"], "r--", label="PDE Residual", alpha=0.7
            )
        if "loss_Dirichlet" in df.columns:
            axes[0].plot(
                df["epoch"], df["loss_Dirichlet"], "b:", label="IC Loss", alpha=0.7
            )
        axes[0].set_yscale("log")
        axes[0].set_title("Training Loss Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (Log Scale)")
        axes[0].legend()
        axes[0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")
        axes[0].set_title("Training Loss")

    # Panel 2: Strain E1(t)
    axes[1].plot(t_np, E_exact_np[:, 0], "k--", label="Exact E1", linewidth=2)
    axes[1].plot(t_np, E_pred_np[:, 0], "r-", label="PINN E1", linewidth=1.5)
    axes[1].set_title(f"Strain E1(t) | rMSE: {error_E1:.2e}")
    axes[1].set_xlabel("Time t [s]")
    axes[1].set_ylabel("Strain E1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Strain E2(t)
    axes[2].plot(t_np, E_exact_np[:, 1], "k--", label="Exact E2", linewidth=2)
    axes[2].plot(t_np, E_pred_np[:, 1], "b-", label="PINN E2", linewidth=1.5)
    axes[2].set_title(f"Strain E2(t) | rMSE: {error_E2:.2e}")
    axes[2].set_xlabel("Time t [s]")
    axes[2].set_ylabel("Strain E2")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Strain Rates dE/dt(t)
    axes[3].plot(t_np, E_t_pred_np[:, 0], "r-", label="PINN Ė1", linewidth=1.5)
    axes[3].plot(t_np, E_t_pred_np[:, 1], "b-", label="PINN Ė2", linewidth=1.5)
    axes[3].set_title(f"Strain Rates Ė(t) | Res: {mean_res:.2e}")
    axes[3].set_xlabel("Time t [s]")
    axes[3].set_ylabel("Rate Ė")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(cfg["output_dir"]) / f"{model_path.stem}_2D_evaluation.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved evaluation plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    if cfg.get("bulk", False):
        pt_files = [p for p in out_dir.glob("*.pt") if "checkpoints" not in str(p)]
        for model_file in pt_files:
            test(model_file)
    else:
        Tk().withdraw()
        model_path = Path(askopenfilename(initialdir=out_dir))
        test(model_path)
