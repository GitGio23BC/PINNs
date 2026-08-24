import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.integrate import solve_ivp

from src.loss import rmse
from src.models import BACKBONE_REGISTRY
from src.operators import OPERATOR_REGISTRY
from src.utils import init_logging

CONFIG_PATH = Path("config.yaml")


def solve_ground_truth_ode(t_span, t_eval, k, ct, c, y, e0=0.0):
    def ode_func(t, E_val):
        E = E_val[0]
        exp_term = np.exp(ct * (E**2))
        psi_E = 2.0 * c * ct * E * exp_term
        H = 2.0 * c * ct * (1.0 + 2.0 * ct * (E**2)) * exp_term
        H_inv2 = 1.0 / (H**2)
        dE_dt = -k * H_inv2 * (psi_E - y)
        return [dE_dt]

    sol = solve_ivp(
        ode_func,
        t_span,
        [e0],
        t_eval=t_eval,
        method="Radau",  # Stiff solver for exponential models
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.y[0]


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

    # Instantiate Model
    if arch not in BACKBONE_REGISTRY:
        logger.error(f"Architecture {arch} not found in BACKBONE_REGISTRY.")
        raise KeyError("Architecture error")

    pinn = BACKBONE_REGISTRY[arch](
        in_channels, hidden_layers, hidden_dim, out_class
    ).to(device)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    if "model_state_dict" in state_dict:
        pinn.load_state_dict(state_dict["model_state_dict"])
    else:
        pinn.load_state_dict(state_dict)

    pinn.eval()
    logger.info(f"Loaded trained {arch} model from {model_path}")

    # Physics Constants
    k = float(cfg["physics"]["relaxation_modulus"])
    c = float(cfg["physics"]["stress_scaling_factor"])
    ct = float(cfg["physics"]["exponential_non_linearity"])
    y = float(cfg["physics"]["stress"])
    e0 = float(cfg["training"]["parameters"]["intial_strain"])

    # Time evaluation domain
    t0, T, t_step = cfg["synt"]["domain"]
    t_test_np = np.linspace(t0, T, int(t_step))
    t_test = torch.tensor(t_test_np, dtype=torch.float32, device=device).view(-1, 1)
    t_test.requires_grad_(True)

    pde_operator = OPERATOR_REGISTRY[eq_name]
    residual, E_pred = pde_operator(pinn, t_test, k, ct, c, y)

    E_t_pred = torch.autograd.grad(
        E_pred,
        t_test,
        grad_outputs=torch.ones_like(E_pred),
        create_graph=False,
    )[0]

    E_exact_np = solve_ground_truth_ode((t0, T), t_test_np, k, ct, c, y, e0=e0)
    E_exact = torch.tensor(E_exact_np, dtype=torch.float32, device=device).view(-1, 1)

    error_E = rmse(E_pred, E_exact).item()
    mean_res = torch.mean(torch.abs(residual)).item()

    # Log
    logger.info("=" * 45)
    logger.info(f"Evaluation Results for: {model_path.name}")
    logger.info(f"Relative L2 Error (Strain E): {error_E:.6e}")
    logger.info(f"Mean Absolute ODE Residual:   {mean_res:.6e}")
    logger.info(f"Initial Strain E(0) Pred:     {E_pred[0].item():.6e} (Target: {e0})")
    logger.info(f"Final Asymptotic Strain E(T): {E_pred[-1].item():.4f}")
    logger.info("=" * 45)

    # Plot
    t_np = t_test.detach().cpu().numpy().flatten()
    E_pred_np = E_pred.detach().cpu().numpy().flatten()
    E_t_pred_np = E_t_pred.detach().cpu().numpy().flatten()

    _fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Training Loss Convergence
    metrics_path = Path(cfg["output_dir"]) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        axes[0].plot(df["epoch"], df["total_loss"], "k-", label="Total Loss", linewidth=1.5)
        if "loss_PDE" in df.columns:
            axes[0].plot(df["epoch"], df["loss_PDE"], "r--", label="PDE Residual Loss", alpha=0.7)
        if "loss_Dirichlet" in df.columns:
            axes[0].plot(df["epoch"], df["loss_Dirichlet"], "b:", label="IC Loss", alpha=0.7)
        axes[0].set_yscale("log")
        axes[0].set_title("Training Loss Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (Log Scale)")
        axes[0].legend()
        axes[0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")
        axes[0].set_title("Training Loss")

    # Panel 2: Strain Profile E(t)
    axes[1].plot(t_np, E_exact_np, "k--", label="Exact (ODE Solver)", linewidth=2)
    axes[1].plot(t_np, E_pred_np, "r-", label="PINN Prediction", linewidth=1.5)
    axes[1].set_title(f"Strain E(t) | rMSE: {error_E:.2e}")
    axes[1].set_xlabel("Time t [s]")
    axes[1].set_ylabel("Green Strain E")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Strain Rate dE/dt(t)
    axes[2].plot(t_np, E_t_pred_np, "g-", label="PINN dE/dt", linewidth=1.5)
    axes[2].set_title(f"Strain Rate Ė(t) | Mean Res: {mean_res:.2e}")
    axes[2].set_xlabel("Time t [s]")
    axes[2].set_ylabel("Ė = dE/dt")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(cfg["output_dir"]) / f"{model_path.stem}_evaluation.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved complete evaluation plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    pt_files = list(out_dir.glob("*.pt"))
    if not pt_files:
        test()
    else:
        for model_file in pt_files:
            test(model_file)