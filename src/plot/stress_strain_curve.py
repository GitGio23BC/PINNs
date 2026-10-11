from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..loader import generate_inertial_dataset
from ..physics import Kinematics, Ogden
from ..utils import load_config


def stress_strain_curve(
    output_filename: str = "stress_strain_curve.png",
):
    cfg = load_config()
    device = torch.device(cfg["training"].get("device", "cpu"))
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if not dataset_path.exists():
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    # Isolate loading phase (t <= T_final / 2) to avoid plotting unloading/oscillations
    t = dataset["time"]
    T_final = t[-1].item()
    time_mask = t <= (T_final / 2.0)

    # Get the gradient
    grad_u = dataset["grad_u"].to(device).to(torch.float64).detach().clone()
    grad_u.requires_grad_(True)

    # Green-Lagrange computation
    kin = Kinematics(grad_u)
    E = kin.E

    # Ogden paramentes
    p_cfg = cfg["physics"]
    tissue = p_cfg["tissue"]
    alphas = torch.tensor(
        p_cfg["alphas"][tissue], device=device, dtype=torch.float64
    )
    mus_list = [float(mu) for mu in p_cfg["mus"][tissue]]
    mus = torch.tensor(mus_list, device=device, dtype=torch.float64)
    lam = float(p_cfg["lam"])
    beta = float(p_cfg["beta"])

    # Stress evaluation
    ogden = Ogden(mus, alphas, beta, lam, kin)
    ogden.get_2Dpsi()
    S = ogden.get_S()  

    E_avg = E.mean(dim=1).detach().cpu()
    S_avg = S.mean(dim=1).detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    components = [
        (0, 0, "$xx$ (Tension)", "$11$"),
        (1, 1, "$yy$ (Transverse)", "$22$"),
        (0, 1, "$xy$ (Shear)", "$12$"),
    ]

    for idx, (i, j, label_suffix, sub_idx) in enumerate(components):
        ax = axes[idx]

        ax.plot(
            E_avg[time_mask, i, j],
            S_avg[time_mask, i, j],
            color="navy",
            linewidth=2,
            label="Loading ($t < T/2$)",
        )

        ax.set_title(
            f"PK2 Stress vs Green Strain ({label_suffix})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(f"Green-Lagrange Strain $E_{{{sub_idx[1:-1]}}}$", fontsize=10)
        ax.set_ylabel(
            f"2nd Piola-Kirchhoff Stress $S_{{{sub_idx[1:-1]}}}$ [Pa]", fontsize=10
        )

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.legend()
        ax.margins(0.05)

    #plt.tight_layout()
    save_target = output_dir / output_filename
    plt.savefig(save_target, dpi=300)
    plt.close()
    print(f"Stress strain curve saved to: {save_target}")


if __name__ == "__main__":
    stress_strain_curve()