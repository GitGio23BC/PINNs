import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from src.loader import load_test_data
from src.models import BACKBONE_REGISTRY
from src.physics import OPERATOR_REGISTRY
from src.utils import init_logging

CONFIG_PATH = Path("config.yaml")


def test(model_path: Path | None = None, config_path: Path = CONFIG_PATH):
    init_logging()
    logger = logging.getLogger(__name__)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])
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
    logger.info(f"Loaded trained static {arch} model from {model_path}")

    # Physics Constants
    k = float(cfg["physics"]["relaxation_modulus"])
    c = float(cfg["physics"]["stress_scaling_factor"])
    c1, c2, c3 = [float(v) for v in cfg["physics"]["c_constants"]]
    b_val = cfg["physics"].get("body_force", [0.0, 0.0])
    s_val = cfg["physics"]["stress"]

    x_val = load_test_data(cfg, device)
    if isinstance(x_val, tuple):
        x_val = x_val[0]

    n_pts = x_val.shape[0]
    b = torch.tensor(b_val, dtype=torch.float32, device=device).expand(n_pts, 2)
    S = torch.tensor(s_val, dtype=torch.float32, device=device).expand(n_pts, len(s_val))

    pde_operator = OPERATOR_REGISTRY[eq_name]
    u_pred, res_haslach, res_momentum = pde_operator(
        pinn, x_val, k, c1, c2, c3, c, b, S
    )

    mean_haslach_res = torch.mean(torch.abs(res_haslach)).item()
    mean_momentum_res = torch.mean(torch.abs(res_momentum)).item()
    max_ux = torch.max(torch.abs(u_pred[:, 0])).item()
    max_uy = torch.max(torch.abs(u_pred[:, 1])).item()

    logger.info("=" * 60)
    logger.info(f"Static Evaluation Results for: {model_path.name}")
    logger.info(f"Mean Absolute Constitutive Residual: {mean_haslach_res:.6e}")
    logger.info(f"Mean Absolute Momentum Residual:     {mean_momentum_res:.6e}")
    logger.info(f"Peak Magnitude u_x:                  {max_ux:.6e}")
    logger.info(f"Peak Magnitude u_y:                  {max_uy:.6e}")
    logger.info("=" * 60)

    # Visualization
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_np = x_val.detach().cpu().numpy()
    u_np = u_pred.detach().cpu().numpy()
    u_mag = (u_np[:, 0] ** 2 + u_np[:, 1] ** 2) ** 0.5

    metrics_path = Path(cfg["output_dir"]) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        axes[0].plot(df["epoch"], df["total_loss"], "k-", label="Total Loss", linewidth=1.5)
        if "loss_haslach" in df.columns:
            axes[0].plot(df["epoch"], df["loss_haslach"], "r--", label="Constitutive Loss", alpha=0.7)
        if "loss_momentum" in df.columns:
            axes[0].plot(df["epoch"], df["loss_momentum"], "b--", label="Momentum Loss", alpha=0.7)
        if "loss_bc" in df.columns:
            axes[0].plot(df["epoch"], df["loss_bc"], "m:", label="BC Loss", alpha=0.7)
        axes[0].set_yscale("log")
        axes[0].set_title("Training Loss Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (Log Scale)")
        axes[0].legend()
        axes[0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")

    sc = axes[1].scatter(
        x_np[:, 0] + u_np[:, 0],
        x_np[:, 1] + u_np[:, 1],
        c=u_mag,
        cmap="viridis",
        s=35,
    )
    plt.colorbar(sc, ax=axes[1], label="||u|| Magnitude [m]")
    axes[1].scatter(x_np[:, 0], x_np[:, 1], c="gray", alpha=0.2, s=10, label="Undeformed")
    axes[1].set_title("Static Deformed State")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    res_mom_norm = torch.norm(res_momentum, dim=-1).detach().cpu().numpy()
    axes[2].hist(res_mom_norm, bins=40, color="crimson", alpha=0.7, edgecolor="black")
    axes[2].set_yscale("log")
    axes[2].set_title("Static Equilibrium Residual Distribution")
    axes[2].set_xlabel("Residual Norm ||R_mom||")
    axes[2].set_ylabel("Count (Log Scale)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(cfg["output_dir"]) / f"{model_path.stem}_static_evaluation.png"
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
        selected_file = askopenfilename(initialdir=out_dir)
        if selected_file:
            test(Path(selected_file))