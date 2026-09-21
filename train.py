import logging
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import create_graph, create_mesh
from src.loader import MGNData, generate_ground_truth
from src.models import build_mgn_model
from src.utils import CSVLogger, deep_merge, init_logging, load_config, set_seed


def compute_loss(*args) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

    # Weight Load

    # Residual
    u_pred = torch.zeros(1)

    # Boundary conditions

    # Total Loss
    total_loss = torch.zeros(1)

    # Metrics Log
    metrics = {"metric": 0.0}

    return total_loss, u_pred, metrics  # Eventually add other returns


def train(overrides: dict | None = None):
    # Logging and set-up
    cfg = load_config("config.yaml")
    cfg = deep_merge(cfg, overrides)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg["training"]["device"])
    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = cfg["model"]["type"]

    model_name = cfg["model"]["model_name"]
    checkpoint_dir = Path(output_dir / "checkpoints" / model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    init_logging()
    logger = logging.getLogger(__name__)
    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
            "epoch",
            # Loss terms
        ],
    )

    method = cfg["training"]["method"]

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

    ## Add Ansatz check

    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants
    p_cfg = cfg["physics"]

    ## Define tissue model

    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float32)

    # Loss weights
    cfg_w = cfg["training"]["loss_weights"]
    loss_weights = {
        "lambda_data": float(cfg_w["lambda_data"]),
        "lambda_haslach": float(cfg_w["lambda_haslach"]),
        "lambda_momentum": float(cfg_w["lambda_momentum"]),
        "lambda_initial": float(cfg_w["lambda_initial"]),
        "lambda_bc_base": float(cfg_w["lambda_bc_base"]),
        "lambda_bc_tip": float(cfg_w["lambda_bc_tip"]),
    }

    # Training Parameters
    epochs = int(cfg["training"]["adam_epochs"])
    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)

    ## ADD MESH BORDER INDEX (assuming a square/cube)
    raw_node_type[mesh.top_nodes] = 3  # type: ignore
    raw_node_type[mesh.bottom_nodes] = 3  # type: ignore
    raw_node_type[mesh.right_nodes] = 2  # type: ignore
    raw_node_type[mesh.left_nodes] = 1  # type: ignore

    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    ## Ablation study section (optional)
    use_trac = method in {"traction", "dynamic", "visco", "full"}
    use_u_dot = method in {"dynamic", "full"}
    use_E = method in {"visco", "full"}
    use_S = method == "full"

    # Training
    logger.info("Start Hybrid MSG training...")

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch: "):
        # Initialisation

        ## Rollout prediction terms

        for t_step in range(time_steps):
            optimizer.zero_grad()
            batch = data_loader.get_batch(t_step)

            ## CREATE GRAPH MUST BE REWORKED
            graph = create_graph(
                mesh=mesh,  # type: ignore
                u=u_prev,  # type: ignore # noqa: F821
                node_type=node_type,  # type: ignore
                t=t_norm,  # type: ignore # noqa: F821
                device=device,  # type: ignore
                trac=batch.trac if use_trac else None,  # type: ignore
                u_dot=u_dot_prev if use_u_dot else None,  # type: ignore  # noqa: F821
                E_prev=E_prev_voigt if use_E else None,  # type: ignore # noqa: F821
                S_prev=S_prev_voigt if use_S else None,  # type: ignore # noqa: F821
            )

            # Add/return needed term accordantly to loss compute function
            total_loss, u_pred, metrics = compute_loss()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
            optimizer.step()

            # Update tollaoout prediction terms

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

match input():
    case "base":
        print("base")
