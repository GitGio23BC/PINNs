import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import create_mesh
from src.geometry.graph import Graph, Mesh, create_graph
from src.loader import MGNBatch, MGNData, generate_ground_truth
from src.loss import bc_loss, mse, r_loss, traction_bc_loss
from src.models import MeshGraphNetAn, build_mgn_model
from src.physics import (
    HUGO,
    HolzapfelEnergy_2D,
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import (
    CSVLogger,
    init_logging,
    load_config,
    set_seed,
    update_config,
)


def compute_loss(
    mgn: nn.Module,
    graph: Graph,
    mesh: Mesh,
    batch: MGNBatch,
    visco_model: HUGO,
    E_prev: torch.Tensor,
    b: torch.Tensor,
    loss_weights: dict[str, float],
    is_ansatz: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:

    w_data = loss_weights["lambda_data"]
    w_haslach = loss_weights["lambda_haslach"]
    w_momentum = loss_weights["lambda_momentum"]
    # w_initial = loss_weights["lambda_initial"]
    w_bc_base = loss_weights.get("lambda_bc_base", 1.0)
    w_bc_tip = loss_weights["lambda_bc_tip"]

    # Residual
    preds_res = mgn(graph)
    u_pred = preds_res[:, :2]
    S_pred = preds_res[:, 2:]

    loss_data = mse(u_pred, batch.u)

    X_ref = graph.mesh_nodes

    haslach_residuals, E_pred = haslach_constitutive_residual_2D(
        u_pred=u_pred,
        S_pred=S_pred,
        X_ref=X_ref,
        E_voigt_prev=E_prev,
        dt=batch.dt,
        visco_model=visco_model,
    )
    loss_haslach = r_loss(haslach_residuals)

    pako_residual, P_pred = pako_residual_2D(
        u_pred=u_pred,
        S_pred=S_pred,
        X_ref=X_ref,
        b=b,
    )
    loss_pako = r_loss(pako_residual)

    # Boundary conditions
    loss_bc_base = torch.tensor(0.0, device=X_ref.device)
    if not is_ansatz:
        u_base = u_pred[mesh.left_nodes]
        loss_bc_base = bc_loss(u_base, torch.zeros_like(u_base))

    right_idx = mesh.right_nodes
    P_tip = P_pred[right_idx]

    loss_bc_tip = traction_bc_loss(P_tip, mesh.right_normals, batch.trac[right_idx])

    # Total Loss
    total_loss = (
        w_data * loss_data
        + w_haslach * loss_haslach
        + w_momentum * loss_pako
        # + w_initial * loss_ic
        + (w_bc_base * loss_bc_base if not is_ansatz else 0.0)
        + w_bc_tip * loss_bc_tip
    )

    metrics = {
        "step_loss": total_loss.detach().item(),
        "loss_data": loss_data.detach().item(),
        "loss_haslach": loss_haslach.detach().item(),
        "loss_pako": loss_pako.detach().item(),
        "loss_ic": 0,
        "loss_bc_base": loss_bc_base.detach().item(),
        "loss_bc_tip": loss_bc_tip.detach().item(),
    }

    return total_loss, E_pred.detach(), u_pred.detach(), S_pred.detach(), metrics


def train(method: str):
    # Logging and set-up
    cfg = load_config("config.yaml")
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(f"./output_{method}")  # Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = cfg["model"]["type"]

    model_name = f"mgn_{method}_{model_type}"
    checkpoint_dir = Path(output_dir / "checkpoints" / model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    init_logging()
    logger = logging.getLogger(__name__)
    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
            "step",
            "epoch",
            "step_loss",
            "loss_data",
            "loss_haslach",
            "loss_pako",
            "loss_ic",
            "loss_bc_base",
            "loss_bc_tip",
        ],
    )

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)
    data_loader = MGNData(cfg, dataset)
    time_grid = dataset["time"]
    time_steps = len(dataset["time"])

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

    # Model
    mgn = build_mgn_model(cfg, method, device=device)
    is_ansatz = isinstance(mgn, MeshGraphNetAn)
    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants
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

    # Loss weights
    cfg_t = cfg["training"]
    loss_weights = {
        "lambda_data": float(cfg_t["loss_weights"]["lambda_data"]),
        "lambda_haslach": float(cfg_t["loss_weights"]["lambda_haslach"]),
        "lambda_momentum": float(cfg_t["loss_weights"]["lambda_momentum"]),
        "lambda_initial": float(cfg_t["loss_weights"].get("lambda_initial", 0.0)),
        "lambda_bc_base": float(cfg_t["loss_weights"].get("lambda_bc_base", 1.0)),
        "lambda_bc_tip": float(cfg_t["loss_weights"]["lambda_bc_tip"]),
    }

    # Training Parameters
    epochs = int(cfg["training"]["adam_epochs"])
    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)

    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1

    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    # Training
    logger.info("Start Hybrid MSG training...")

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch: "):
        # Initialisation
        u_prev = torch.zeros((mesh.n_nodes, 2), device=device, dtype=torch.float32)
        u_dot_prev = torch.zeros((mesh.n_nodes, 2), device=device, dtype=torch.float32)
        E_prev_voigt = torch.zeros(
            (mesh.n_nodes, 3), device=device, dtype=torch.float32
        )
        S_prev_voigt = torch.zeros(
            (mesh.n_nodes, 3), device=device, dtype=torch.float32
        )
        for t_step in range(time_steps):
            optimizer.zero_grad()
            batch = data_loader.get_batch(t_step)

            t_norm = (batch.t - time_grid[0]) / (time_grid[-1] - time_grid[0])

            if method == "base":
                graph = create_graph(
                    mesh=mesh, u=u_prev, node_type=node_type, t=t_norm, device=device
                )
            elif method == "traction":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    trac=batch.trac,
                    t=t_norm,
                    device=device,
                )
            elif method == "dynamic":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    u_dot=u_dot_prev,
                    trac=batch.trac,
                    t=t_norm,
                    device=device,
                )
            elif method == "visco":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    E_prev=E_prev_voigt,
                    trac=batch.trac,
                    t=t_norm,
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
                    t=t_norm,
                    device=device,
                )
            else:
                raise KeyError(f"Training method '{method}' not found")

            total_loss, E_curr, u_pred, S_pred, metrics = compute_loss(
                mgn=mgn,
                graph=graph,
                mesh=mesh,
                batch=batch,
                visco_model=visco_model,
                E_prev=E_prev_voigt,
                b=b,
                loss_weights=loss_weights,
                is_ansatz=is_ansatz,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
            optimizer.step()

            u_dot_prev = (u_pred - u_prev) / batch.dt
            u_prev = u_pred
            E_prev_voigt = E_curr
            S_prev_voigt = S_pred

            # Metrics
            metrics["step"] = t_step
            metrics["epoch"] = epoch
            metrics_logger.log(metrics)
        if epoch % 50 == 0:
            torch.save(
                {
                    "model_state_dict": mgn.state_dict(),
                    "config": cfg,
                },
                checkpoint_dir / f"{model_name}_{epoch}.pt",
            )

    torch.save(
        {
            "model_state_dict": mgn.state_dict(),
            "config": cfg,
        },
        output_dir / f"{model_name}.pt",
    )
    logger.info("-----------------------------Train Ended-----------------------------")


if __name__ == "__main__":
    cfg = load_config()
    if cfg["bulk"]:
        for train_type in ["standard", "ansatz_space"]:
            update_config("config.yaml", ["model", "type"], train_type)
            for method in ["base", "traction", "dynamic", "visco", "full"]:
                train(method)
    else:
        train(cfg["training"]["method"])
