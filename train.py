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


def compute_causal_weight(
    epsilon: float, previous_physics_loss: torch.Tensor
) -> torch.Tensor:
    return torch.exp(-epsilon * previous_physics_loss).detach()


def anneal_causal_epsilon(
    epoch: int, epochs: int, epsilon_start: float, epsilon_end: float
) -> float:
    progress = (epoch - 1) / max(epochs - 1, 1)
    return epsilon_start * (epsilon_end / epsilon_start) ** progress


def compute_loss(
    mgn: nn.Module,
    graph: Graph,
    mesh: Mesh,
    batch: MGNBatch,
    visco_model: HUGO,
    E_prev: torch.Tensor,
    b: torch.Tensor,
    loss_weights: dict[str, float],
    causal_weight: torch.Tensor | None = None,
    u_prev: torch.Tensor | None = None,
    u_dot_target: torch.Tensor | None = None,
    num_time_steps: int = 1,
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
    loss_velocity = torch.zeros((), device=u_pred.device)
    if u_dot_target is not None:
        if u_prev is None:
            raise ValueError("u_prev is required when velocity supervision is enabled")
        loss_velocity = mse((u_pred - u_prev) / batch.dt, u_dot_target)
        loss_data = loss_data + loss_velocity

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

    loss_phys = w_haslach * loss_haslach + w_momentum * loss_pako
    if causal_weight is None:
        total_loss = (
            w_data * loss_data
            + loss_phys
            + (w_bc_base * loss_bc_base if not is_ansatz else 0.0)
            + w_bc_tip * loss_bc_tip
        )
        causal_weight_value = 1.0
    else:
        causal_weight = causal_weight.detach()
        total_loss = (
            w_data * loss_data + causal_weight * loss_phys
            + (w_bc_base * loss_bc_base if not is_ansatz else 0.0)
            + w_bc_tip * loss_bc_tip
        ) / num_time_steps
        causal_weight_value = causal_weight.item()

    metrics = {
        "step_loss": total_loss.detach().item(),
        "loss_data": loss_data.detach().item(),
        "loss_haslach": loss_haslach.detach().item(),
        "loss_pako": loss_pako.detach().item(),
        "loss_ic": 0,
        "loss_bc_base": loss_bc_base.detach().item(),
        "loss_bc_tip": loss_bc_tip.detach().item(),
    }
    if causal_weight is not None:
        metrics.update(
            {
                "loss_velocity": loss_velocity.detach().item(),
                "loss_phys": loss_phys.detach().item(),
                "causal_weight": causal_weight_value,
            }
        )

    return total_loss, E_pred.detach(), u_pred.detach(), S_pred.detach(), metrics


def train(method: str):
    cfg = load_config("config.yaml")
    set_seed(int(cfg.get("seed", 42)))

    cfg_t = cfg["training"]
    tr_loop = bool(cfg_t.get("tr_loop", False))
    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(f"./output_{method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = cfg["model"]["type"]
    loop_suffix = "_tr_loop" if tr_loop else ""
    model_name = f"mgn_{method}_{model_type}{loop_suffix}"
    checkpoint_dir = output_dir / "checkpoints" / model_name
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
        ]
        + (
            ["loss_velocity", "loss_phys", "causal_weight", "epsilon"]
            if tr_loop
            else []
        ),
    )

    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]
    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)
    data_loader = MGNData(cfg, dataset)
    time_grid = dataset["time"]
    time_steps = len(time_grid)
    if tr_loop and time_steps < 2:
        raise ValueError("tr_loop training requires at least two time frames")

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

    mgn = build_mgn_model(cfg, method, device=device)
    is_ansatz = isinstance(mgn, MeshGraphNetAn)
    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

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

    loss_weights = {
        "lambda_data": float(cfg_t["loss_weights"]["lambda_data"]),
        "lambda_haslach": float(cfg_t["loss_weights"]["lambda_haslach"]),
        "lambda_momentum": float(cfg_t["loss_weights"]["lambda_momentum"]),
        "lambda_initial": float(cfg_t["loss_weights"].get("lambda_initial", 0.0)),
        "lambda_bc_base": float(cfg_t["loss_weights"].get("lambda_bc_base", 1.0)),
        "lambda_bc_tip": float(cfg_t["loss_weights"]["lambda_bc_tip"]),
    }
    epochs = int(cfg_t["adam_epochs"])
    epsilon_start = float(cfg_t.get("causal_epsilon_start", 1e-2))
    epsilon_end = float(cfg_t.get("causal_epsilon_end", 100.0))
    if tr_loop and (epsilon_start <= 0 or epsilon_end <= 0):
        raise ValueError("Causal epsilon endpoints must be positive")
    tr_loop_steps = max(time_steps - 1, 1)
    ground_truth_velocity = torch.zeros_like(dataset["u"])
    if time_steps > 1:
        ground_truth_velocity[1:] = (
            dataset["u"][1:] - dataset["u"][:-1]
        ) / data_loader.dt
        ground_truth_velocity[0] = ground_truth_velocity[1]

    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1
    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    logger.info("Start Hybrid MSG training...")
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch: "):
        is_tr_loop = tr_loop
        epsilon = (
            anneal_causal_epsilon(epoch, epochs, epsilon_start, epsilon_end)
            if is_tr_loop
            else None
        )
        if is_tr_loop:
            u_prev = dataset["u"][0].detach()
            u_dot_prev = ground_truth_velocity[0].detach()
            E_prev_voigt = dataset["E"][0].detach()
            S_prev_voigt = dataset["S"][0].detach()
            previous_physics_loss = torch.zeros((), device=device)
            optimizer.zero_grad()
            step_range = range(1, time_steps)
        else:
            u_prev = torch.zeros((mesh.n_nodes, 2), device=device, dtype=torch.float32)
            u_dot_prev = torch.zeros((mesh.n_nodes, 2), device=device, dtype=torch.float32)
            E_prev_voigt = torch.zeros(
                (mesh.n_nodes, 3), device=device, dtype=torch.float32
            )
            S_prev_voigt = torch.zeros(
                (mesh.n_nodes, 3), device=device, dtype=torch.float32
            )
            step_range = range(time_steps)

        for t_step in step_range:
            if not is_tr_loop:
                optimizer.zero_grad()
            batch = data_loader.get_batch(t_step)
            input_step = t_step - 1 if is_tr_loop else t_step
            t_input = time_grid[input_step]
            t_norm = (t_input - time_grid[0]) / (time_grid[-1] - time_grid[0])

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
            elif method == "full" or is_tr_loop:
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

            causal_weight = (
                compute_causal_weight(epsilon, previous_physics_loss)
                if is_tr_loop
                else None
            )
            u_dot_target = (
                ground_truth_velocity[t_step] if is_tr_loop else None
            )
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
                causal_weight=causal_weight,
                u_prev=u_prev if is_tr_loop else None,
                u_dot_target=u_dot_target,
                num_time_steps=tr_loop_steps,
            )

            total_loss.backward()
            if not is_tr_loop:
                torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
                optimizer.step()

            u_dot_prev = (u_pred - u_prev) / batch.dt
            u_prev = u_pred
            E_prev_voigt = E_curr
            S_prev_voigt = S_pred
            if is_tr_loop:
                previous_physics_loss = previous_physics_loss + torch.tensor(
                    metrics["loss_phys"], device=device
                )

            metrics["step"] = t_step - 1 if is_tr_loop else t_step
            metrics["epoch"] = epoch
            if is_tr_loop:
                metrics["epsilon"] = epsilon
            metrics_logger.log(metrics)

        if is_tr_loop:
            torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 50 == 0:
            torch.save(
                {"model_state_dict": mgn.state_dict(), "config": cfg},
                checkpoint_dir / f"{model_name}_{epoch}.pt",
            )

    torch.save(
        {"model_state_dict": mgn.state_dict(), "config": cfg},
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
