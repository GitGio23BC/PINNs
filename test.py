import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.geometry import create_graph, create_mesh
from src.loader import MGNData, generate_ground_truth
from src.loss import rl2e
from src.models import FEATURE_DIMS, MeshGraphNetAn, build_mgn_model
from src.physics import (
    HUGO,
    HolzapfelEnergy_2D,
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import init_logging, load_config


def test(model_path: Path | None = None, method: str | None = None):
    # Logging, Config & Paths
    init_logging()
    logger = logging.getLogger("TestMGN")

    cfg = load_config("config.yaml")
    device = torch.device(cfg["training"].get("device", "cpu"))

    if model_path is None:
        output_dir = Path(cfg.get("output_dir", "./output"))
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"
    else:
        output_dir = model_path.parent

    model_name = model_path.stem

    if method is None:
        method = model_name.split("_")[1]
        method = (
            method if method in FEATURE_DIMS else cfg["training"].get("method", "base")
        )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Type check
    ckpt_cfg = checkpoint.get("config", {})
    if "type" in ckpt_cfg["model"]:
        model_type = ckpt_cfg["model"]["type"]
    else:
        model_type = cfg["model"].get("type", "standard")

    cfg["model"]["type"] = model_type

    # Dataset & Loader
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"]
    time_steps = len(time_grid)
    data_loader = MGNData(cfg, dataset)

    # Geometry
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    mesh = create_mesh(
        width=width,
        height=height,
        nx=int(cfg["domain"]["nx"]),
        ny=int(cfg["domain"]["ny"]),
        device=device,
    )

    center_coords = torch.tensor([width / 2.0, height / 2.0], device=device)
    center_node_idx = int(
        torch.argmin(torch.linalg.norm(mesh.nodes - center_coords, dim=-1)).item()
    )

    # Model
    mgn = build_mgn_model(cfg, method=method, device=device)
    mgn.load_state_dict(state_dict)
    mgn.eval()
    is_ansatz = isinstance(mgn, MeshGraphNetAn)
    logger.info(
        f"Loaded [{method.upper()}] | Type: {model_type} | Path: {model_path.name}"
    )

    # Physics
    p_cfg = cfg["physics"]
    visco_model = HUGO(
        HolzapfelEnergy_2D(
            c=float(p_cfg["c"]),
            c1=float(p_cfg["c1"]),
            c2=float(p_cfg["c2"]),
            c3=float(p_cfg["c3"]),
            device=device,
        ),
        k_relax=float(p_cfg["k_relax"]),
    )
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float32)

    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1
    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    # Initial states
    u_prev = torch.zeros((mesh.n_nodes, 2), device=device, dtype=torch.float32)
    u_dot_prev = torch.zeros((mesh.n_nodes, 2), device=device, dtype=torch.float32)
    E_prev_voigt = torch.zeros((mesh.n_nodes, 3), device=device, dtype=torch.float32)
    S_prev_voigt = torch.zeros((mesh.n_nodes, 3), device=device, dtype=torch.float32)

    all_u_pred, all_P_pred, all_E_pred = [], [], []
    all_haslach_res, all_pako_res = [], []

    # Rollout
    logger.info("Executing evaluation rollout...")
    with torch.enable_grad():
        for t_step in range(time_steps):
            batch = data_loader.get_batch(t_step)
            t_norm = (batch.t - time_grid[0]) / (time_grid[-1] - time_grid[0])

            if method == "base":
                graph = create_graph(
                    mesh=mesh, u=u_prev, node_type=node_type, device=device
                )
            elif method == "traction":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    trac=batch.trac,
                    device=device,
                )
            elif method == "dynamic":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    u_dot=u_dot_prev,
                    trac=batch.trac,
                    device=device,
                )
            elif method == "visco":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    E_prev=E_prev_voigt,
                    trac=batch.trac,
                    device=device,
                )
            elif method == "full":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    u_dot=u_dot_prev,
                    E_prev=E_prev_voigt,
                    S_prev=S_prev_voigt,
                    trac=batch.trac,
                    device=device,
                )
            else:
                raise (
                    KeyError(
                        f"Method '{method}' do not found.\n Available methods {FEATURE_DIMS.keys()}"
                    )
                )

            preds = mgn(graph)
            u_pred = preds[:, :2]
            S_pred = preds[:, 2:]
            X_ref = graph.mesh_nodes

            haslach_res, E_curr = haslach_constitutive_residual_2D(
                u_pred=u_pred,
                S_pred=S_pred,
                X_ref=X_ref,
                E_voigt_prev=E_prev_voigt,
                dt=batch.dt,
                visco_model=visco_model,
            )
            pako_res, P_curr = pako_residual_2D(
                u_pred=u_pred,
                S_pred=S_pred,
                X_ref=X_ref,
                b=b,
            )

            u_dot_prev = (u_pred.detach() - u_prev) / batch.dt
            u_prev = u_pred.detach()
            E_prev_voigt = E_curr.detach()
            S_prev_voigt = S_pred.detach()

            all_u_pred.append(u_prev.cpu())
            all_E_pred.append(E_curr.detach().cpu())
            all_P_pred.append(P_curr.detach().cpu())
            all_haslach_res.append(haslach_res.detach().cpu())
            all_pako_res.append(pako_res.detach().cpu())

    # Metrics Computation
    u_final_pred = all_u_pred[-1]
    u_final_exact = dataset["u"][-1].cpu()
    final_l2_error = rl2e(u_final_pred, u_final_exact).item()
    mean_haslach = torch.cat(all_haslach_res).abs().mean().item()
    mean_pako = torch.cat(all_pako_res).abs().mean().item()

    logger.info("=" * 60)
    logger.info(f"Evaluation Results for: {model_name} ({model_type})")
    logger.info(f"Final Relative L2 Displacement Error: {final_l2_error:.2%}")
    logger.info(f"Trajectory Mean Haslach Residual:     {mean_haslach:.6e}")
    logger.info(f"Trajectory Mean Momentum Residual:    {mean_pako:.6e}")
    logger.info("=" * 60)

    # Diagnostic Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Model: {model_name} [{model_type}] | Relative L2 Error: {final_l2_error:.2%}",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: Convergence
    metrics_path = output_dir / f"metrics_{model_name}.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)

        metrics_pack = (
            ("step_loss", "k-", "Total Loss"),
            ("loss_data", "c-", "Data Loss"),
            ("loss_haslach", "r--", "Constitutive Loss"),
            ("loss_pako", "b--", "Momentum Loss"),
            ("loss_bc_base", "g:", "BC Base"),
            ("loss_bc_tip", "m:", "BC Traction"),
        )

        for col_name, style, label in metrics_pack:
            if col_name in df.columns:
                axes[0, 0].plot(df[col_name], style, label=label, linewidth=1.2)

        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Training Loss Convergence")
        axes[0, 0].set_xlabel("Logged Step")
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")

    # Subplot 2: Peak Deformation
    mid_step = time_steps // 2
    x_nodes = mesh.nodes.cpu().numpy()
    u_mid = all_u_pred[mid_step].numpy()
    sc = axes[0, 1].scatter(
        x_nodes[:, 0] + u_mid[:, 0],
        x_nodes[:, 1] + u_mid[:, 1],
        c=np.linalg.norm(u_mid, axis=-1),
        cmap="viridis",
        s=35,
    )
    plt.colorbar(sc, ax=axes[0, 1], label=r"$\|\mathbf{u}\|$ [m]")
    axes[0, 1].scatter(
        x_nodes[:, 0], x_nodes[:, 1], c="gray", alpha=0.25, s=15, label="Reference"
    )
    axes[0, 1].set_title(
        f"Peak Deformed State ($t = {time_grid[mid_step].item():.2f}$ s)"
    )
    axes[0, 1].set_xlabel("X [m]")
    axes[0, 1].set_ylabel("Y [m]")
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Momentum Residual Norm
    res_norm = torch.linalg.norm(torch.cat(all_pako_res), dim=-1).numpy()
    axes[0, 2].hist(res_norm, bins=30, color="crimson", alpha=0.7, edgecolor="black")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Momentum Residual Distribution")
    axes[0, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{mom}}\|$")
    axes[0, 2].grid(True, alpha=0.3)

    # Subplot 4: Hysteresis Loop
    pred_E11 = [E[center_node_idx, 0].item() for E in all_E_pred]
    pred_P11 = [P[center_node_idx, 0, 0].item() for P in all_P_pred]
    exact_E11 = dataset["E"][:, center_node_idx, 0].cpu().numpy()
    exact_P11 = dataset["trac_ext"][:, mesh.right_nodes[0], 0].cpu().numpy()

    axes[1, 0].plot(exact_E11, exact_P11, "k-", linewidth=2.0, label="Exact Target")
    axes[1, 0].plot(pred_E11, pred_P11, "r--", linewidth=1.8, label="MGN Prediction")
    axes[1, 0].set_title("Hysteresis Loop (Centre Node)")
    axes[1, 0].set_xlabel(r"Strain $E_{11}$ [-]")
    axes[1, 0].set_ylabel(r"Stress $P_{11}$ [Pa]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, ls="--", alpha=0.3)

    # Subplot 5: Neumann Traction Boundary Condition
    t_plot = time_grid.cpu().numpy()
    pred_tip_P11 = [P[mesh.right_nodes, 0, 0].mean().item() for P in all_P_pred]
    exact_tip_P11 = dataset["trac_ext"][:, mesh.right_nodes[0], 0].cpu().numpy()

    axes[1, 1].plot(
        t_plot, exact_tip_P11, "k--", linewidth=1.5, label=r"Target $t_{\mathrm{ext}}$"
    )
    axes[1, 1].plot(
        t_plot, pred_tip_P11, "m-", linewidth=1.8, label=r"Predicted $P_{11} \cdot n$"
    )
    axes[1, 1].set_title("Neumann Boundary Traction ($X = W$)")
    axes[1, 1].set_xlabel("Time $t$ [s]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, ls="--", alpha=0.3)

    # Subplot 6: Constitutive Residual Norm
    haslach_norm = torch.linalg.norm(torch.cat(all_haslach_res), dim=-1).numpy()
    axes[1, 2].hist(
        haslach_norm, bins=30, color="darkorange", alpha=0.7, edgecolor="black"
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Constitutive Residual Distribution")
    axes[1, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{visco}}\|$")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{model_name}_evaluation.png"
    plt.show()
    logger.info(f"Saved evaluation plot to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    out_dir = Path(cfg.get("output_dir", "./output"))

    if cfg.get("bulk", False):
        model_paths = [
            p
            for p in out_dir.glob("*.pt")
            if not any(part in str(p) for part in ["checkpoint", "2D", "PINN"])
        ]
        if model_paths == []:
            model_paths = [
                p
                for p in out_dir.parent.rglob("*.pt")
                if not any(part in str(p) for part in ["checkpoint", "2D", "PINN"])
            ]

        for model_file in model_paths:
            try:
                print("=" * 10)
                print(model_file.stem)
                test(model_file)
            except Exception as e:
                print("-" * 10)
                print(e)
    else:
        Tk().withdraw()
        selected = askopenfilename(initialdir=out_dir)
        if selected:
            test(Path(selected))
