import logging
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.geometry import Mesh, create_graph, create_mesh
from src.loader import (
    BodyState,
    LossScales,
    MGNBatch,
    MGNData,
    generate_inertial_dataset,
)
from src.loss import data_loss, dirichlet_loss, neumann_loss, pako_loss
from src.models import MeshGraphNet
from src.physics import Kinematics, Ogden, pako_residual_2D
from src.plot import animate_deformation
from src.utils import (
    CSVLogger,
    deep_merge,
    init_logging,
    load_config,
    set_seed,
    stencil_grad,
    triangle_shape_gradients,
)
from test import test


def compute_step_loss(
    mesh: Mesh,
    batch: MGNBatch,
    ogden: Ogden,
    state: BodyState,
    alpha_rayleigh: float,
    beta_rayleigh: float,
    weights: dict,
    dN_dX: torch.Tensor,
    scales: LossScales,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    lambda_data = weights["lambda_data"]
    lambda_dirichlet = weights["lambda_dirichlet"]
    lambda_neumann = weights["lambda_neumann"]
    lambda_pako = weights["lambda_pako"]

    # Data loss
    psi_pred = torch.sum(ogden.psi * mesh.elements_area)  # type: ignore
    l_data = data_loss(
        u_pred= state.u,
        u_exact=batch.u_exact,
        u_scale=scales.u_scale,
        psi_pred = psi_pred,
        psi_exact=batch.psi_exact,
        psi_scale=scales.psi_scale,
    )

    # Dirichlet loss
    l_dirichlet = dirichlet_loss(
        u_fixed_pred=state.u.reshape(-1)[mesh.fixed_dofs],
        u_fixed_gt=batch.u_fixed_exact,
    )

    # Neumann loss
    P_right = state.P[mesh.right_elements]
    n_normal = torch.tensor([1.0, 0.0], dtype=P_right.dtype, device=P_right.device)
    l_neumann = neumann_loss(
        P_boundary = P_right,
        n_normal = n_normal,
        trac_ext = batch.trac_ext,
        trac_scale = scales.trac_scale,
    )

    # Pako loss
    R = pako_residual_2D(
        P=state.P,
        mesh=mesh,
        dN_dX=dN_dX,
        b=state.b,
        rho=state.rho,
        u_t=state.u_t,
        u_tt=state.u_tt,
        alpha_rayleigh=alpha_rayleigh,
        beta_rayleigh=beta_rayleigh,
        F_ext=batch.F_ext
    )
    l_pako = pako_loss(
        residual=R, 
        force_scale=scales.force_scale, 
        free_dofs=mesh.free_dofs)

    l_physics = l_dirichlet + l_neumann + l_pako

    # Unweighted step loss sum
    l_step = (
        lambda_data * l_data
        + lambda_dirichlet * l_dirichlet
        + lambda_neumann * l_neumann
        + lambda_pako * l_pako
    )

    metrics = {
        "l_step": l_step.detach(),
        "l_data": l_data.detach(),
        "l_dirichlet": l_dirichlet.detach(),
        "l_neumann": l_neumann.detach(),
        "l_pako": l_pako.detach(),
        "l_physics": l_physics.detach(),
    }

    return l_step, l_physics, metrics



def train(overrides: dict | None = None, dtype = torch.float32) -> Path:
    # Logging and set-up
    cfg = load_config("config.yaml")
    cfg = deep_merge(cfg, overrides)
    set_seed(int(cfg.get("seed", 42)))

    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(requested_device)

    logger = logging.getLogger(__name__)
    logger.info("Training device: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    model_name = cfg["model"]["model_name"]
    print(model_name)

    output_dir = Path(cfg["output_dir"]) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir / "checkpoints")
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
            "l_physics",
            "l_neumann",
            "l_pako",
            "causal_w",
            "eps",
        ],
    )
    metrics = {}

    # Mesh VEDO COME AGGIUNGERE IL TRD
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        nx=nx,
        ny=ny,
        device=device,
        dtype=dtype,
    )

    ## stenchil derivate
    dN_dX, _ = triangle_shape_gradients(mesh.nodes, mesh.elements)

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        dataset = torch.load(
            dataset_path,
            map_location=device,
            weights_only=False,
        )

    data_loader = MGNData(
        dataset=dataset, 
        mesh=mesh,
        device=device,
        dtype=dtype,
    )

    # Time mesh
    time_steps = len(dataset["time"])
    dt = data_loader.dt

    # Model
    model_cfg = cfg["model"]
    mgn = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)

    ## Add Ansatz check (optional)
>>>>>>> a8ac6895ca9d332c901b9fa6ece7a7ffe2b36876

    optimizer = Adam(
        mgn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    # Physics Constants
    p_cfg = cfg["physics"]
    tissue = p_cfg["tissue"]
    alphas = torch.tensor(p_cfg["alphas"][tissue], device=device, dtype=dtype)
    mus_list = [float(mu) for mu in p_cfg["mus"][tissue]]
    mus = torch.tensor(mus_list, device=device, dtype=dtype)
    lam = float(p_cfg["lam"])  # TO SEARCH
    beta = float(p_cfg["beta"])  # TO SEARCH
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=dtype)
    rho = torch.tensor(p_cfg["rho_0"], device=device, dtype=dtype)
    sigma_max = float(p_cfg["sigma_max"])

    u_scale = torch.mean(dataset["u"]**2).clamp_min(1e-8).detach()
    psi_scale = torch.mean(dataset["psi"]**2).clamp_min(1e-8).detach()
    trac_scale = float(p_cfg["sigma_max"])

    force_reference = torch.zeros(
        (mesh.n_nodes, 2),
        dtype=dtype,
        device=device,
    )
    force_reference[mesh.right_nodes, 0] = sigma_max * mesh.right_lengths
    force_scale = (
        torch.mean(force_reference.reshape(-1)[mesh.free_dofs] ** 2)
        .clamp_min(1e-8)
        .detach()
    )
    scales = LossScales(
        u_scale=u_scale,
        psi_scale=psi_scale,
        trac_scale=trac_scale,
        force_scale=force_scale,
    )

    # TO PUT IN THE CONFIG
    zeta = float(p_cfg["zeta"])
    w_min, w_max = [float(w) for w in p_cfg["w"]]
    alpha_rayleigh = 2.0 * zeta * (w_max + w_min)
    beta_rayleigh = 0

    ## Define tissue model NON NECESSARIO, LO DEFINIAMO NELLA LOSS FUNCTION, SE SERVE
    # To define in accordently with the loss function
    # data_loader.load_physics(mus, alphas, beta, lam)

    t_cfg = cfg["training"]

    # Training Parameters
    ## Annealing
    causal_cfg = t_cfg["causal_annealing"]
    eps_init = float(causal_cfg["eps_init"])
    eps_final = float(causal_cfg["eps_final"])

    loss_weights = {
        "lambda_data": 1.0,
        "lambda_dirichlet": 0.0,
        "lambda_neumann": 0.0,
        "lambda_pako": 0.0,
    }

    epochs = int(t_cfg["epochs"])
    warming_epochs = int(t_cfg["warming_epochs"])

    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1

    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    # Training Loop
    logger.info("Start Hybrid MSG training...")

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        # Causal annealing
        eps = (
            eps_init * ((eps_final / eps_init) ** ((epoch - 1) / (epochs - 1)))
            if epoch > warming_epochs
            else eps_init
        )

        if epoch == warming_epochs:
            temp_metric = metrics.copy()
            loss_weights = {
                "lambda_data": 1 / temp_metric["l_data"],
                "lambda_dirichlet": 1 / temp_metric["l_dirichlet"],
                "lambda_neumann": 1 / temp_metric["l_neumann"],
                "lambda_pako": 1 / temp_metric["l_pako"],
            }
        # Initialisation
        u_prev = torch.zeros((mesh.n_nodes, 2), dtype=dtype, device=device)
        u_prev2 = torch.zeros((mesh.n_nodes, 2), dtype=dtype, device=device)
        velocity_prev = torch.zeros_like(u_prev)
        acceleration_prev = torch.zeros_like(u_prev)

        u_t = torch.zeros_like(u_prev)
        u_tt = torch.zeros_like(u_prev)

        world_pos = mesh.nodes.clone()

        l_prev_physics = torch.zeros((), dtype=dtype, device=device)
        total_epoch_loss = torch.zeros((), dtype=dtype, device=device)
        epoch_metrics = []

        optimizer.zero_grad()
        for t_step in range(time_steps):
            batch = data_loader.get_batch(t_step)
            trac_feat = torch.zeros((mesh.n_nodes, 2), device=device, dtype=dtype)
            trac_feat[mesh.right_nodes] = batch.trac_ext / trac_scale
            velocity_feat = velocity_prev

            graph = create_graph(
                mesh=mesh,
                world_pos=world_pos,
                node_type=node_type,
                velocity=velocity_feat,
                acceleration = acceleration_prev,
                boundary_traction=trac_feat,
                device=device,
                dtype=dtype,
            )

            # Get the prediciton
            u = mgn(graph)
            u = u.to(dtype=dtype)

            # Get dynamics
            u_t = (u - u_prev) / dt
            u_tt = (u - 2.0 * u_prev + u_prev2) / (dt**2)

            # Get external forces
            F_ext = torch.zeros((mesh.n_nodes, 2), dtype=dtype, device=device)
            F_ext[mesh.right_nodes] = batch.trac_ext * mesh.right_lengths.unsqueeze(-1)

            # Get kinematics
            grad_u = stencil_grad(u, mesh.elements, dN_dX)
            kin = Kinematics(grad_u)
            ogden = Ogden(mus=mus, alphas=alphas, beta=beta, lam=lam, kin=kin)
            psi = ogden.get_2Dpsi()
            S = ogden.get_S()
            P = kin.compute_P(S)

            state = BodyState(
                u=u,
                u_t=u_t,
                u_tt=u_tt,
                P=P,
                b=b,
                rho=rho,
                psi=psi,
            )

            # Loss compute
            l_step, l_physics, metrics = compute_step_loss(
                alpha_rayleigh=alpha_rayleigh,
                beta_rayleigh=beta_rayleigh,
                weights=loss_weights,
                mesh=mesh,
                dN_dX=dN_dX,
                batch=batch,
                ogden=ogden,
                state=state,
                scales=scales,
            )

            # Annealing
            causal_w = torch.exp(-eps * torch.clamp(l_prev_physics.to(dtype=dtype), min=20.0))

            weighted_loss = causal_w * l_step
            total_epoch_loss += weighted_loss.detach().to(dtype=dtype)

            # The constitutive stress is obtained with autograd.grad(create_graph=True).
            # Retain only this step's graph until the backward pass completes.
            weighted_loss.backward()

            # Update rollout prediction terms
            l_prev_physics += (l_physics.detach().to(dtype=dtype).div(t_step + 1))
            u_prev2 = u_prev.detach().clone()
            u_prev = u.detach().clone()
            velocity_prev = ((u_prev - u_prev2) / dt).detach()
            acceleration_prev = u_tt.detach().clone()
            world_pos = (mesh.nodes + u_prev).detach()

            epoch_metrics.append({"causal_w": causal_w, **metrics})


        # Weight optimisation
        torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
        optimizer.step()

        metric_names = (
            "causal_w",
            "l_step",
            "l_data",
            "l_physics",
            "l_dirichlet",
            "l_neumann",
            "l_pako",
        )
        
        logged_metrics = torch.stack(
            [
                torch.stack([step_data[name] for step_data in epoch_metrics])
                for name in metric_names
            ]
        ).cpu()
        log_entries = []
        for t_step in range(time_steps):
            log_entry = {
                "epoch": epoch,
                "step": t_step,
                "causal_w": logged_metrics[0, t_step].item(),
                "eps": eps,
                **{
                    name: logged_metrics[index, t_step].item()
                    for index, name in enumerate(metric_names[1:], start=1)
                },
            }
            log_entries.append(log_entry)
        metrics_logger.log_many(log_entries)

        avg_epoch_loss = total_epoch_loss.item() / time_steps
        logger.info(
            f"Epoch {epoch}/{epochs} Complete | Avg Step Loss: {avg_epoch_loss:.6f}"
        )

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

    return output_dir / f"{model_name}.pt"


if __name__ == "__main__":
    dataset_names = [
        "F_const_stiff.pt",
        "F_step_stiff.pt",
        "F_ramp_stiff.pt",
        "F_half_sin_stiff.pt",
        "F_sin_stiff.pt",
    ]
    for dataset_name in dataset_names:
        model_name = f"DEBUG_{Path(dataset_name).stem}"
        overrides = {
            "model": {"model_name": model_name},
            "data": {"dataset_name": dataset_name},
        }
        model_pth = train(overrides=overrides)
        test(model_path=model_pth)
        animate_deformation(model_path=model_pth)
