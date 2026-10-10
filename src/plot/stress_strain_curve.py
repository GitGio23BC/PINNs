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
    device = cfg["training"]["device"]
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if not dataset_path.exists():
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    # Loading phase time
    t = dataset["time"]
    T_final = t[-1].item()
    time_mask = t >= 0

    # Taking grad
    grad_u = dataset["grad_u"].to(device).to(torch.float64).detach().clone()
    grad_u.requires_grad_(True)

    # Getting the Green strain
    kin = Kinematics(grad_u)
    E = kin.E

    # Odgen model definition
    p_cfg = cfg["physics"]
    alphas = torch.tensor(p_cfg["alphas"], device=device)
    mus_list = [float(mu) for mu in p_cfg["mus"]]
    mus = torch.tensor(mus_list, dtype=torch.float64, device=device)
    lam = float(p_cfg["lambda"])
    beta = float(p_cfg["beta"])

    # Compute energy density and differentiate
    ogden = Ogden(torch.zeros(1), mus, alphas, beta, lam, kin)
    psi = ogden._get_2Dpsi()

    # Compute stress
    S = ogden.get_S(psi.sum(), kin.E)
    sigma = kin.compute_sigma(S)

    # Body average values
    # To change if elements area isn't the same for all the elments in the mesh
    E_avg = E.mean(dim=1).detach().cpu()
    Sigma_avg = sigma.mean(dim=1).detach().cpu()

    _, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    components = [
        (0, 0, "$xx$ (Tension)"),
        (1, 1, "$yy$ (Transverse)"),
        (0, 1, "$xy$ (Shear)"),
    ]

    for idx, (i, j, title_suffix) in enumerate(components):
        ax = axes[idx]

        ax.plot(
            E_avg[time_mask, i, j],
            Sigma_avg[time_mask, i, j],
            color="navy",
            linewidth=2,
            label="Loading ($t \\le T/2$)",
        )

        ax.set_title(
            f"Cauchy Stress vs Green Strain ({title_suffix})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(f"Green-Lagrange Strain $E_{{{title_suffix[1:3]}}}$", fontsize=10)
        ax.set_ylabel(
            f"Cauchy Stress $\\sigma_{{{title_suffix[1:3]}}}$ [Pa]", fontsize=10
        )

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.margins(0.05)

    axes[0].set_xlim(left=-0.001)
    axes[0].set_ylim(bottom=-100)

    plt.tight_layout()
    save_target = output_dir / output_filename
    plt.savefig(save_target)
    plt.close()
    print(f"Stress strain curve saved to: {save_target}")


if __name__ == "__main__":
    stress_strain_curve()
