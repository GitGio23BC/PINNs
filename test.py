import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # Necessario per i plot 3D

from src.geometry import create_graph, create_mesh_3D
from src.loader import BodyState, MGNData, generate_inertial_dataset
from src.models import MeshGraphNet
from src.physics import Kinematics, Ogden, pako_residual
from src.plot import animate_deformation
from src.utils import init_logging, load_config, stencil_grad, tetrahedral_shape_gradients


def test(model_path: Path | None = None):
    # Logging and set-up
    init_logging()
    logger = logging.getLogger(name=__name__)

    cfg = load_config(config_path="config.yaml")
    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(requested_device)
    dtype = torch.float32
    output_dir = Path(cfg.get("output_dir", "./output")) / str(cfg["model"]["model_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = (
        model_path
        if model_path is not None
        else output_dir / f"{cfg['model']['model_name']}.pt"
    )
    model_name = model_file.stem

    # Mesh 3D Alignment
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]
    nx, ny, nz = (
        int(cfg["domain"]["nx"]),
        int(cfg["domain"]["ny"]),
        int(cfg["domain"]["nz"]),
    )

    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=nx,
        ny=ny,
        nz=nz,
        device=device,
        dtype=dtype,
    )

    dN_dX, _ = tetrahedral_shape_gradients(nodes=mesh.nodes, elements=mesh.elements)

    # Node Type Encoding allineato a train.py
    directions = ('left', 'right', 'bottom', 'top', 'back', 'front')
    node_type = torch.zeros((mesh.n_nodes, len(directions)), dtype=dtype, device=device)
    for i, direction in enumerate(directions):
        node_type[mesh.get_boundary_nodes(direction), i] = 1.0

    # Data Loader
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / str(cfg["data"]["dataset_name"])

    if not dataset_path.exists():
        dataset = generate_inertial_dataset(cfg=cfg, device=device)
    else:
        dataset = torch.load(
            f=dataset_path,
            map_location=device,
            weights_only=False,
        )

    dataset = {
        key: value.to(device=device, dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.to(device=device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in dataset.items()
    }

    data_loader = MGNData(
        dataset=dataset,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )
    time_steps = data_loader.num_steps
    dt = data_loader.dt
    time_grid = data_loader.time
    
    # Scaling
    trac_scale = float(cfg["physics"]["sigma_max"])

    # Model definition & loading
    checkpoint = torch.load(f=model_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_cfg = cfg["model"]
    mgn = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)
    mgn.load_state_dict(state_dict=state_dict)
    mgn.eval()
    logger.info(f"Loaded trained MeshGraphNet model from {model_file}")

    # Physics Constants allineati a train.py
    p_cfg = cfg["physics"]
    tissue = p_cfg["tissue"]
    alphas = torch.tensor(data=p_cfg["alphas"][tissue], device=device, dtype=dtype)
    mus_list = [float(mu) for mu in p_cfg["mus"][tissue]]
    mus = torch.tensor(data=mus_list, device=device, dtype=dtype)
    lam = float(p_cfg["lam"])
    beta = float(p_cfg["beta"])
    b = torch.tensor(data=p_cfg["body_force"], device=device, dtype=dtype)
    rho = torch.tensor(data=[p_cfg["rho_0"]], device=device, dtype=dtype)
    
    zeta = float(p_cfg["zeta"])
    w_min, w_max = [2*torch.pi*float(f) for f in p_cfg["f"]]
    alpha_rayleigh = 2.0 * zeta * (w_max + w_min)
    beta_rayleigh = 2.0 * zeta * (w_max * w_min) / (w_max + w_min)

    # Rollout Initialisation (3D = 3 components)
    u_prev = torch.zeros(size=(mesh.n_nodes, 3), dtype=dtype, device=device)
    u_prev2 = torch.zeros_like(input=u_prev)
    velocity_prev = torch.zeros_like(input=u_prev)
    acceleration_prev = torch.zeros_like(input=u_prev)
    world_pos = mesh.nodes.clone()

    all_u_pred = []
    all_S_pred = []
    all_pako_res = []

    # Diagnostics arrays for stress-strain analysis
    E11_pred_traj = []
    S11_pred_traj = []
    E11_gt_traj = []
    S11_gt_traj = []

    eval_elems = mesh.elements

    # Rollout Testing
    logger.info("Starting PINN test rollout...")
    for t_step in range(time_steps):
        batch = data_loader.get_batch(t_step)
        trac_feat = torch.zeros((mesh.n_nodes, 3), device=device, dtype=dtype)
        trac_feat[mesh.get_boundary_nodes("right")] = batch.trac_ext / trac_scale

        graph = create_graph(
            mesh=mesh,
            world_pos=world_pos,
            node_type=node_type,
            velocity=velocity_prev,
            boundary_traction=trac_feat,
            device=device,
            dtype=dtype,
        )

        with torch.enable_grad():
            u = mgn(graph).to(dtype=dtype)
            u.requires_grad_(True)

            # Acceleration & Velocity
            u_t = 2/dt * (u - u_prev) - velocity_prev
            u_tt = (
                4/dt**2 * (u - u_prev) - 4/dt * velocity_prev - acceleration_prev
            )

            # Kinematics & Stress (Prediction)
            grad_u = stencil_grad(u, mesh.elements, dN_dX)
            kin = Kinematics(grad_u)
            ogden = Ogden(mus=mus, alphas=alphas, beta=beta, lam=lam, kin=kin)
            psi = ogden.get_psi()
            P = ogden.get_P()
            S = torch.linalg.solve(kin.F, P)

            state = BodyState(
                b=b,
                rho=rho,
                u=u,
                u_t=u_t,
                u_tt=u_tt,
                P=P,
                psi=psi,
            )

            # Momentum Residual
            R_pako = pako_residual(
                P=state.P,
                mesh=mesh,
                dN_dX=dN_dX,
                b=state.b,
                rho=state.rho,
                u=state.u,
                u_t=state.u_t,
                u_tt=state.u_tt,
                alpha_rayleigh=alpha_rayleigh,
                beta_rayleigh=beta_rayleigh,
                F_ext=batch.F_ext,
            )

            # Kinematics & Stress (Reference)
            u_exact = batch.u_exact.to(device=device, dtype=dtype).detach().clone()
            u_exact.requires_grad_(True)
            grad_u_exact = stencil_grad(u_exact, mesh.elements, dN_dX)
            kin_exact = Kinematics(grad_u=grad_u_exact)
            ogden_exact = Ogden(mus=mus, alphas=alphas, beta=beta, lam=lam, kin=kin_exact)
            ogden_exact.get_psi()
            S_exact = torch.linalg.solve(kin_exact.F, ogden_exact.get_P())

        # Track E11 and S11 for stress-strain curves
        e11_pred = kin.E[eval_elems, 0, 0].mean().item()
        s11_pred = S[eval_elems, 0, 0].mean().item()
        e11_gt = kin_exact.E[eval_elems, 0, 0].mean().item()
        s11_gt = S_exact[eval_elems, 0, 0].mean().item()

        E11_pred_traj.append(e11_pred)
        S11_pred_traj.append(s11_pred)
        E11_gt_traj.append(e11_gt)
        S11_gt_traj.append(s11_gt)

        all_u_pred.append(u.detach().cpu())
        all_S_pred.append(S.detach().cpu())
        all_pako_res.append(R_pako.reshape(-1)[mesh.free_dofs].detach().cpu())

        # Update Variables
        u_prev2 = u_prev.detach().clone()
        u_prev = u.detach().clone()
        velocity_prev = u_t.detach()
        acceleration_prev = u_tt.detach().clone()
        world_pos = (mesh.nodes + u_prev).detach()

    # Metrics computation
    reference_trajectory = (
        (dataset["u_complete"] if "u_complete" in dataset else dataset["u"])
        .detach()
        .cpu()
    )
    reference_norm = torch.linalg.norm(input=reference_trajectory.flatten()).clamp_min(1e-8)
    
    l2_err_traj = [
        torch.linalg.norm(input=u_pred - reference_trajectory[t]).div(reference_norm).item()
        for t, u_pred in enumerate(all_u_pred)
    ]
    final_l2_error = l2_err_traj[-1]
    
    residual_free = torch.cat(tensors=all_pako_res)
    mean_pako = residual_free.abs().mean().item()
    
    logger.info("=" * 60)
    logger.info(f"Evaluation Results for: {model_file.name}")
    logger.info(f"Final Relative L2 Displacement Error: {final_l2_error:.2%}")
    logger.info(f"Mean Trajectory Relative L2 Error:   {np.mean(a=l2_err_traj):.2%}")
    logger.info(f"Trajectory Mean Momentum Residual:    {mean_pako:.6e}")
    logger.info("=" * 60)

    # Plotting Setup
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))
    fig.suptitle(
        f"Model Evaluation: {model_file.name} | Relative L2 Error: {final_l2_error:.2%}",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: Training Convergence
    metrics_path = output_dir / f"metrics_{model_name}.csv"
    losses = {
        "l_data": "red",
        "l_pako": "blue",
        "l_dirichlet": "green",
        "l_neumann": "magenta",
    }
    if metrics_path.exists():
        df = pd.read_csv(filepath_or_buffer=metrics_path)
        axes[0, 0].plot(
            df["l_step"], color="black", linestyle="-", label="Total Loss", linewidth=1.5
        )
        for metric_name, color in losses.items():
            if metric_name in df.columns:
                axes[0, 0].plot(
                    df[metric_name], color=color, linestyle="--", label=metric_name.capitalize(), alpha=0.7
                )
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Training Loss Convergence")
        axes[0, 0].set_xlabel("Logged Step")
        axes[0, 0].set_ylabel("Loss (Log Scale)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")

    # Subplot 2: Deformed Mesh State t_final (MODIFICATO PER IL 3D)
    axes[0, 1].remove()  # Rimuovi l'asse 2D
    ax3d = fig.add_subplot(2, 3, 2, projection='3d') # Aggiungi l'asse 3D
    
    x_nodes_np = mesh.nodes.cpu().numpy()
    u_final_np = all_u_pred[-1].numpy()
    u_mag = np.linalg.norm(u_final_np, axis=-1)

    def_x = x_nodes_np[:, 0] + u_final_np[:, 0]
    def_y = x_nodes_np[:, 1] + u_final_np[:, 1]
    def_z = x_nodes_np[:, 2] + u_final_np[:, 2]

    sc = ax3d.scatter(def_x, def_y, def_z, c=u_mag, cmap="viridis", s=15, alpha=0.8)
    plt.colorbar(mappable=sc, ax=ax3d, label=r"$\|\mathbf{u}\|$ Magnitude [m]", pad=0.1)
    
    # Reference in grigio
    ax3d.scatter(x_nodes_np[:, 0], x_nodes_np[:, 1], x_nodes_np[:, 2], c="gray", alpha=0.1, s=5, label="Reference")
    
    ax3d.set_title("Final Deformed State 3D ($t = T$)")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend()

    # Subplot 3: Momentum Residual Norm
    res_free = residual_free.numpy()
    axes[0, 2].hist(res_free, bins=30, color="crimson", alpha=0.7, edgecolor="black")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Momentum Balance Residual Norm")
    axes[0, 2].set_xlabel(r"$|R_{\mathrm{mom,free}}|$")
    axes[0, 2].set_ylabel("Count (Log Scale)")
    axes[0, 2].grid(True, alpha=0.3)

    # Subplot 4: Stress-Strain Response (S11 vs E11)
    axes[1, 0].plot(E11_gt_traj, S11_gt_traj, color="black", linestyle="--", linewidth=2.0, label=r"Ground Truth")
    axes[1, 0].plot(E11_pred_traj, S11_pred_traj, color="red", linestyle="-", marker="o", linewidth=1.5, markersize=4, alpha=0.85, label=r"Prediction")
    axes[1, 0].set_title("Constitutive Response ($S_{11}$ vs $E_{11}$)")
    axes[1, 0].set_xlabel(r"Green-Lagrange Strain $E_{11}$ [-]")
    axes[1, 0].set_ylabel(r"2nd Piola-Kirchhoff Stress $S_{11}$ [Pa]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # Subplot 5: Tip Stress Trajectory (S11 vs Time)
    t_plot = time_grid.cpu().numpy()
    axes[1, 1].plot(t_plot, S11_gt_traj, color="black", linestyle="--", linewidth=1.8, label=r"Ground Truth $S_{11}$")
    axes[1, 1].plot(t_plot, S11_pred_traj, color="red", linestyle="-", linewidth=1.8, label=r"Predicted $S_{11}$")
    axes[1, 1].set_title("Mean Stress Trajectory")
    axes[1, 1].set_xlabel("Time $t$ [s]")
    axes[1, 1].set_ylabel("Stress $S_{11}$ [Pa]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    # Subplot 6: Relative L2 Error over Time
    axes[1, 2].plot(t_plot, l2_err_traj, color="green", linestyle="-", linewidth=1.8, label=r"Relative $L_2$ Error")
    axes[1, 2].set_title("Displacement Error Trajectory")
    axes[1, 2].set_xlabel("Time $t$ [s]")
    axes[1, 2].set_ylabel("Relative $L_2$ Error")
    axes[1, 2].legend()
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{model_file.stem}_evaluation.png"
    plt.savefig(fname=plot_path, dpi=300)
    logger.info(f"Saved evaluation plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if cfg.get("bulk"):
        pt_files = [p for p in out_dir.glob("*.pt") if "checkpoints" not in str(p)]
        for model_file in pt_files:
            test(model_path=model_file)
    else:
        Tk().withdraw()
        selected_model = askopenfilename(initialdir=out_dir, filetypes=[("PyTorch Models", "*.pt")])
        if selected_model:
            selected_model_path = Path(selected_model)
            test(model_path=selected_model_path)
            # animate_deformation(model_path=selected_model_path) # Decommenta se animate_deformation supporta il 3D