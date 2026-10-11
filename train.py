import logging
import math
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import Mesh, create_graph, create_mesh
from src.loader import MGNBatch, MGNData, generate_inertial_dataset
from src.loss import *
from src.models import MeshGraphNet
from src.physics import Kinematics, Ogden, pako_residual_2D
from src.utils import (
    CSVLogger,
    deep_merge,
    init_logging,
    load_config,
    set_seed,
    stencil_grad,
    triangle_shape_gradients,
)


def compute_step_loss(
    b: torch.Tensor,
    rho: float,
    alpha_rayleigh: float,
    weights: dict,
    u_pred: torch.Tensor,
    u_t: torch.Tensor,
    u_tt: torch.Tensor,
    P: torch.Tensor,
    ogden: Ogden,
    mesh: Mesh,
    dN_dX: torch.Tensor,
    batch: MGNBatch,
    F_ext: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    lambda_data = weights["lambda_data"]
    lambda_dirichlet = weights["lambda_dirichlet"]
    lambda_neumann = weights["lambda_neumann"]
    lambda_pako = weights["lambda_pako"]

    # Data loss
    psi_pred = torch.sum(ogden.psi * mesh.elements_area)  # type: ignore
    l_data = data_loss(u_pred, batch.u_exact, psi_pred, batch.psi_exact)

    # Dirichlet loss
    l_dirichlet = dirichlet_loss(
        u_pred, mesh.fixed_dofs, u_fixed_gt=batch.u_fixed_exact
    )

    # Neumann loss
    P_right = P[mesh.right_elements]
    n_normal = torch.tensor([1.0, 0.0], dtype=P.dtype, device=P.device)
    l_neumann = neumann_loss(P_right, n_normal, batch.trac_ext)

    # Pako loss
    R = pako_residual_2D(P, mesh, dN_dX, b, rho, u_t, u_tt, alpha_rayleigh, F_ext)
    l_pako = pako_loss(R, F_ext)

    l_physics = l_dirichlet + l_neumann + l_pako

    # Unweighted step loss sum
    l_step = (
        lambda_data * l_data
        + lambda_dirichlet * l_dirichlet
        + lambda_neumann * l_neumann
        + lambda_pako * l_pako
    )

    metrics = {
        "l_step": l_step.item(),
        "l_data": l_data.item(),
        "l_dirichlet": l_dirichlet.item(),
        "l_neumann": l_neumann.item(),
        "l_pako": l_pako.item(),
    }

    return l_step, l_physics, metrics


def train(overrides: dict | None = None):
    # Logging and set-up
    cfg = load_config("config.yaml")
    cfg = deep_merge(cfg, overrides)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg["training"]["device"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["model_name"]
    checkpoint_dir = Path(output_dir / "checkpoints" / model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    init_logging()
    logger = logging.getLogger(__name__)
    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
            "epoch",
            "step",
            "l_step",
            "l_data",
            "l_dirichlet",
            "l_neumann",
            "l_pako",
            "causal_w",
            "eps",
        ],
    )

    method = cfg["training"]["method"]

    # Mesh
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        nx=nx,
        ny=ny,
        device=device,
    )

    ## stenchil derivate
    dN_dX, _ = triangle_shape_gradients(mesh.nodes, mesh.elements)

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)
    data_loader = MGNData(dataset, mesh)

    # Time mesh
    time_steps = len(dataset["time"])
    dt = data_loader.dt

    # Model
    mgn = MeshGraphNet().to(device=device, dtype=torch.float64)

    ## Add Ansatz check (optional)

    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants
    p_cfg = cfg["physics"]
    tissue = p_cfg["tissue"]
    alphas = torch.tensor(p_cfg["alphas"][tissue], device=device, dtype=torch.float64)
    mus_list = [float(mu) for mu in p_cfg["mus"][tissue]]
    mus = torch.tensor(mus_list, device=device, dtype=torch.float64)
    lam = float(p_cfg["lam"])  # TO SEARCH
    beta = float(p_cfg["beta"])  # TO SEARCH
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float64)
    rho = float(p_cfg["rho_0"])

    # TO PUT IN THE CONFIG
    zeta = 0.03
    w_max, w_min = 1.0, 0.01
    alpha_rayleigh = 2.0 * zeta * (w_max + w_min)

    ## Define tissue model
    # To define in accordently with the loss function
    # data_loader.load_physics(mus, alphas, beta, lam)

    # Training Parameters
    ## Annealing
    causal_cfg = cfg["training"]["causal_annealing"]
    eps_init = float(causal_cfg["eps_init"])
    eps_final = float(causal_cfg["eps_final"])

    ## Loss weights
    cfg_w = cfg["training"]["loss_weights"]
    loss_weights = {
        "lambda_data": float(cfg_w["lambda_data"]),
        "lambda_dirichlet": float(cfg_w["lambda_dirichlet"]),
        "lambda_neumann": float(cfg_w["lambda_neumann"]),
        "lambda_pako": float(cfg_w["lambda_pako"]),
    }

    epochs = int(cfg["training"]["epochs"])

    ## Mesh border index assuming a square/cube
    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1

    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    ## Ablation study section (optional)
    use_trac = method in {"traction", "dynamic", "visco", "full"}  # noqa: F841
    use_u_dot = method in {"dynamic", "full"}  # noqa: F841
    use_E = method in {"visco", "full"}  # noqa: F841
    use_S = method == "full"  # noqa: F841

    # Training Loop
    logger.info("Start Hybrid MSG training...")

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        # Causal annealing
        if epochs > 1:
            eps = eps_init * ((eps_final / eps_init) ** ((epoch - 1) / (epochs - 1)))
        else:
            eps = eps_init
        # Initialisation
        u_prev = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64, device=device)
        u_prev2 = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64, device=device)

        u_t = torch.zeros_like(u_prev)
        u_tt = torch.zeros_like(u_prev)

        world_pos = mesh.nodes.clone()

        l_prev_physics = 0.0
        total_epoch_loss = 0.0

        optimizer.zero_grad()

        for t_step in range(time_steps):
            batch = data_loader.get_batch(t_step)

            graph = create_graph(
                mesh=mesh,
                world_pos=world_pos,
                node_type=node_type,
                device=device,
            )

            # Get the prediciton
            u = mgn(graph)
            u = u.to(dtype=torch.float64)

            # Get dynamics
            u_t = (u - u_prev) / dt
            u_tt = (u - 2.0 * u_prev + u_prev2) / (dt**2)

            F_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64, device=device)
            F_ext[mesh.right_nodes] = batch.trac_ext * mesh.right_lengths.unsqueeze(-1)

            # Get kinematics
            grad_u = stencil_grad(u, mesh.elements, dN_dX)
            kin = Kinematics(grad_u)
            ogden = Ogden(mus=mus, alphas=alphas, beta=beta, lam=lam, kin=kin)
            ogden.get_2Dpsi()
            S = ogden.get_S()
            P = kin.compute_P(S)

            # Loss compute
            l_step, l_physics, metrics = compute_step_loss(
                b=b,
                rho=rho,
                alpha_rayleigh=alpha_rayleigh,
                weights=loss_weights,
                u_pred=u,
                u_t=u_t,
                u_tt=u_tt,
                P=P,
                ogden=ogden,
                mesh=mesh,
                dN_dX=dN_dX,
                batch=batch,
                F_ext=F_ext,
            )

            # Annealing
            causal_arg = -eps * min(l_prev_physics, 20.0)
            causal_w = math.exp(causal_arg)

            weighted_loss = causal_w * l_step
            total_epoch_loss += weighted_loss.detach().item()

            # Backpropagation
            weighted_loss.backward()

            # Update rollout prediction terms
            l_prev_physics += l_physics.detach().item() / (t_step + 1)
            u_prev2 = u_prev.detach().clone()
            u_prev = u.detach().clone()
            world_pos = (mesh.nodes + u_prev).detach()

            # Logging
            metrics["step"] = t_step
            metrics["epoch"] = epoch
            metrics["causal_w"] = causal_w
            metrics["eps"] = eps
            metrics_logger.log(metrics)

        # Weight optimisation
        torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
        optimizer.step()

        # Backup saving
        if epoch % 50 == 0:
            torch.save(
                {"model_state_dict": mgn.state_dict(), "config": cfg},
                checkpoint_dir / f"{model_name}_{epoch}.pt",
            )
    # Final Save
    torch.save(
        {"model_state_dict": mgn.state_dict(), "config": cfg},
        output_dir / f"{model_name}.pt",
    )
    logger.info("Training Completed Successfully!")

    logger.info("-----------------------------Train Ended-----------------------------")


if __name__ == "__main__":
    cfg = load_config()
    if cfg["bulk"]:
        for train_type in ["standard", "ansatz_space"]:
            for method in ["base", "traction", "dynamic", "visco", "full"]:
                overrides = {
                    "model": {
                        "type": train_type,
                        "model_name": f"{train_type}_model_{method}",
                    },
                    "output_dir": f"{train_type}/{method}",
                    "training": {
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                }

                train(overrides)
    else:
        train()
