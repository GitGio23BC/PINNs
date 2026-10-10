import logging
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import create_graph, create_mesh
from src.loader import MGNData, generate_ground_truth
from src.models import MeshGraphNet
from src.utils import CSVLogger, deep_merge, init_logging, load_config, set_seed

loss_pred = 0.0
w = 1.0
def compute_loss(l_weighted, w) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

    l_data = data_loss(u_pred, u_exact, psi_pred, psi_exact)
    l_dirichlet = dirichlet_loss(u_bc_pred)
    l_neumann = neumann_loss(P, n_normal, F_ext)
    l_pako = pako_loss(div_P, b, rho, u_tt)

    # Total Loss
    
    l_step = lambda_data * l_data + lambda_dirichlet * l_dirichlet + lambda_neumann * l_neumann + 
    lambda_pako * l_pako

    l_weighted += w * l_step
     
    total_loss = torch.mean(l_weighted)

    # Metrics Log
    metrics = {"metric": 0.0}
    
    loss_prev += l_dirichlet + l_neumann + l_pako
    w = torch.detach(torch.exp(-eps * loss_prev))

    return total_loss, metrics, w, l_weighted   # Eventually add other returns


def train(overrides: dict | None = None):
    # Logging and set-up
    cfg = load_config("config.yaml")
    cfg = deep_merge(cfg, overrides)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg["training"]["device"])
    output_dir = Path(cfg["output_dir"])
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
    data_loader = MGNData(dataset)
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
    mgn = MeshGraphNet().to(device)

    ## Add Ansatz check (optional)

    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants
    p_cfg = cfg["physics"]
    alphas = torch.tensor(p_cfg["alphas"], device=device, dtype=torch.float32)
    mus = torch.tensor(p_cfg["mus"], device=device, dtype=torch.float32)
    lam = float(p_cfg["lam"])
    beta = float(p_cfg["beta"])
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float32)

    ## Define tissue model
    # To define in accordently with the loss function
    data_loader.load_physics(mus, alphas, beta, lam)

    # Training Parameters
    ## Loss weights
    cfg_w = cfg["training"]["loss_weights"]
    loss_weights = {
        "lambda_data": float(cfg_w["lambda_data"]),
        "lambda_haslach": float(cfg_w["lambda_haslach"]),
        "lambda_momentum": float(cfg_w["lambda_momentum"]),
        "lambda_initial": float(cfg_w["lambda_initial"]),
        "lambda_bc_base": float(cfg_w["lambda_bc_base"]),
        "lambda_bc_tip": float(cfg_w["lambda_bc_tip"]),
    }

    epochs = int(cfg["training"]["adam_epochs"])
    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)

    ## Mesh border index assuming a square/cube
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1

    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    ## Ablation study section (optional)
    use_trac = method in {"traction", "dynamic", "visco", "full"}
    use_u_dot = method in {"dynamic", "full"}
    use_E = method in {"visco", "full"}
    use_S = method == "full"

    # Training Loop
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
<<<<<<< HEAD

match input():
    case "base":
        print("base")
=======
>>>>>>> OdgenBranch/Dataset_Generation
