import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import LBFGS, Adam
from tqdm import tqdm

from src.geometry import Mesh, create_mesh
from src.loader import PINNBatch, PINNSampler, generate_ground_truth
from src.loss import bc_loss, mse, r_loss, traction_bc_loss
from src.models import ParametricPINN, build_model
from src.physics import (
    HUGO,
    HolzapfelEnergy_2D,
    Kinematics,
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import (
    CSVLogger,
    #alert,
    grad,
    init_logging,
    load_config,
    set_seed,
    voigt_to_tensor,
)


def compute_loss(
    pinn: nn.Module,
    batch: PINNBatch,
    visco_model: HUGO,
    b: torch.Tensor,
    loss_weights: dict[str, float],
    u_ic_target: torch.Tensor | None = None,
    is_ansatz: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:

    w_data = loss_weights["lambda_data"]
    w_haslach = loss_weights["lambda_haslach"]
    w_momentum = loss_weights["lambda_momentum"]
    w_initial = loss_weights["lambda_initial"]
    w_bc_base = loss_weights.get("lambda_bc_base", 1.0)
    w_bc_tip = loss_weights["lambda_bc_tip"]

    # Residual
    preds_res = pinn(t=batch.t_res, X=batch.X_res)
    u_pred_res = preds_res[:, :2]
    S_pred_res = preds_res[:, 2:]

    loss_data = mse(u_pred_res, batch.u_res)

    haslach_residuals = haslach_constitutive_residual_2D(
        u_pred=u_pred_res,
        S_pred=S_pred_res,
        X_ref=batch.X_res,
        t=batch.t_res,
        visco_model=visco_model,
    )
    loss_haslach = r_loss(haslach_residuals)

    pako_residual = pako_residual_2D(
        u_pred=u_pred_res,
        S_pred=S_pred_res,
        X_ref=batch.X_res,
        b=b,
    )
    loss_pako = r_loss(pako_residual)

    # Initial conditions
    preds_ic = pinn(t=batch.t_ic, X=batch.X_ic)
    u_ic_pred = preds_ic[:, :2]
    S_ic_pred = preds_ic[:, 2:]

    target_u = batch.u_ic_target if u_ic_target is None else u_ic_target
    loss_ic = mse(u_ic_pred, target_u) + mse(S_ic_pred, batch.S_ic_target)

    # Boundary conditions
    loss_bc_base = torch.tensor(0.0, device=batch.X_res.device)
    if not is_ansatz:
        preds_base = pinn(t=batch.t_base, X=batch.X_base)
        u_base = preds_base[:, :2]
        loss_bc_base = bc_loss(u_base, torch.zeros_like(u_base))

    preds_tip = pinn(t=batch.t_neu, X=batch.X_neu)
    u_tip = preds_tip[:, :2]
    S_tip = preds_tip[:, 2:]

    kin = Kinematics(grad(u_tip, batch.X_neu))
    S_tip_voigt = voigt_to_tensor(S_tip, is_shear=False)
    P_tip = kin.compute_P(S_tip_voigt)
    loss_bc_tip = traction_bc_loss(P_tip, batch.normals_neu, batch.trac_target)

    # Total Loss
    total_loss = (
        w_data * loss_data
        + w_haslach * loss_haslach
        + w_momentum * loss_pako
        + w_initial * loss_ic
        + (w_bc_base * loss_bc_base if not is_ansatz else 0.0)
        + w_bc_tip * loss_bc_tip
    )

    metrics = {
        "step_loss": total_loss.detach().item(),
        "loss_data": loss_data.detach().item(),
        "loss_haslach": loss_haslach.detach().item(),
        "loss_pako": loss_pako.detach().item(),
        "loss_ic": loss_ic.detach().item(),
        "loss_bc_base": loss_bc_base.detach().item(),
        "loss_bc_tip": loss_bc_tip.detach().item(),
    }

    return total_loss, metrics


def run_train_standard(
    cfg: dict[str, dict],
    pinn: nn.Module,
    sampler: PINNSampler,
    visco_model: HUGO,
    b: torch.Tensor,
    loss_weights: dict[str, float],
    output_dir: Path,
    time_steps: int,
) -> None:
    logger = logging.getLogger("TrainStandard")
    logger.info("Executing Standard PINN Training...")

    model_name = "PINN_std"
    checkpoint_dir = output_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
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

    is_ansatz = not isinstance(pinn, ParametricPINN)
    adam_epochs = int(cfg["training"].get("adam_epochs", 100))
    optimizer_adam = Adam(
        pinn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    for epoch in tqdm(range(1, adam_epochs + 1), desc="Standard Adam"):
        optimizer_adam.zero_grad()
        batch = sampler.sample_window(step_start=0, step_end=time_steps)
        loss, metrics = compute_loss(
            pinn=pinn,
            batch=batch,
            visco_model=visco_model,
            b=b,
            loss_weights=loss_weights,
            is_ansatz=is_ansatz,
        )
        loss.backward()
        optimizer_adam.step()

        metrics["epoch"] = epoch
        metrics_logger.log(metrics)

        if epoch % 50 == 0:
            torch.save(
                {"model_state_dict": pinn.state_dict(), "config": cfg},
                checkpoint_dir / f"{model_name}_adam_{epoch}.pt",
            )

    lbfgs_iters = int(cfg["training"].get("lbfgs_epochs", 50))
    optimizer_lbfgs = LBFGS(
        pinn.parameters(),
        lr=float(cfg["optimizer"].get("lbfgs_lr", 0.5)),
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    global_epoch = adam_epochs

    for _ in tqdm(range(1, lbfgs_iters + 1), desc="Standard L-BFGS"):
        batch = sampler.sample_window(step_start=0, step_end=time_steps)
        current_metrics = {}

        def closure():
            optimizer_lbfgs.zero_grad()
            step_loss, step_metrics = compute_loss(
                pinn=pinn,
                batch=batch,  # noqa: B023
                visco_model=visco_model,
                b=b,
                loss_weights=loss_weights,
                is_ansatz=is_ansatz,
            )
            step_loss.backward()
            current_metrics.update(step_metrics)  # noqa: B023
            return step_loss

        optimizer_lbfgs.step(closure)
        global_epoch += 1

        current_metrics["epoch"] = global_epoch
        metrics_logger.log(current_metrics)

    torch.save(
        {"model_state_dict": pinn.state_dict(), "config": cfg},
        output_dir / f"{model_name}.pt",
    )
    logger.info("Standard Training Completed.")


def run_train_curriculum(
    cfg: dict[str, dict],
    pinn: nn.Module,
    sampler: PINNSampler,
    visco_model: HUGO,
    b: torch.Tensor,
    loss_weights: dict[str, float],
    output_dir: Path,
    time_steps: int,
) -> None:
    logger = logging.getLogger("TrainCurriculum")
    logger.info("Executing Curriculum Time-Expansion Training...")

    model_name = "PINN_curr"
    checkpoint_dir = output_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
            "epoch",
            "step_end",
            "step_loss",
            "loss_data",
            "loss_haslach",
            "loss_pako",
            "loss_ic",
            "loss_bc_base",
            "loss_bc_tip",
        ],
    )

    is_ansatz = not isinstance(pinn, ParametricPINN)
    adam_epochs = int(cfg["training"].get("adam_epochs", 100))
    ramp_step = (time_steps / adam_epochs) / 0.7

    optimizer_adam = Adam(
        pinn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )

    for epoch in tqdm(range(1, adam_epochs + 1), desc="Curriculum Adam"):
        optimizer_adam.zero_grad()
        step_end = max(1, min(time_steps, int(ramp_step * epoch)))
        batch = sampler.sample_window(step_start=0, step_end=step_end)

        loss, metrics = compute_loss(
            pinn=pinn,
            batch=batch,
            visco_model=visco_model,
            b=b,
            loss_weights=loss_weights,
            is_ansatz=is_ansatz,
        )
        loss.backward()
        optimizer_adam.step()

        metrics["epoch"] = epoch
        metrics["step_end"] = step_end
        metrics_logger.log(metrics)

        if epoch % 50 == 0:
            torch.save(
                {"model_state_dict": pinn.state_dict(), "config": cfg},
                checkpoint_dir / f"{model_name}_adam_{epoch}.pt",
            )

    lbfgs_iters = int(cfg["training"].get("lbfgs_epochs", 50))
    optimizer_lbfgs = LBFGS(
        pinn.parameters(),
        lr=float(cfg["optimizer"].get("lbfgs_lr", 0.5)),
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    global_epoch = adam_epochs
    step_end = time_steps

    for _ in tqdm(range(1, lbfgs_iters + 1), desc="Curriculum L-BFGS"):
        batch = sampler.sample_window(step_start=0, step_end=step_end)
        current_metrics: dict[str, float] = {}

        def closure():
            optimizer_lbfgs.zero_grad()
            step_loss, step_metrics = compute_loss(
                pinn=pinn,
                batch=batch,  # noqa: B023
                visco_model=visco_model,
                b=b,
                loss_weights=loss_weights,
                is_ansatz=is_ansatz,
            )
            step_loss.backward()
            current_metrics.update(step_metrics)  # noqa: B023
            return step_loss

        optimizer_lbfgs.step(closure)
        global_epoch += 1

        current_metrics["epoch"] = global_epoch
        current_metrics["step_end"] = step_end
        metrics_logger.log(current_metrics)

    torch.save(
        {"model_state_dict": pinn.state_dict(), "config": cfg},
        output_dir / f"{model_name}.pt",
    )
    logger.info("Curriculum Training Completed.")


def run_train_seq2seq(
    cfg: dict[str, dict],
    pinn: nn.Module,
    sampler: PINNSampler,
    dataset: dict[str, torch.Tensor],
    mesh: Mesh,
    visco_model: HUGO,
    b: torch.Tensor,
    loss_weights: dict[str, float],
    output_dir: Path,
    time_steps: int,
) -> None:
    logger = logging.getLogger("TrainSeq2Seq")
    logger.info("Executing Sequence-to-Sequence (Time-Marching) Training...")

    model_name = "PINN_s2s"
    checkpoint_dir = output_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = CSVLogger(
        output_dir / f"metrics_{model_name}.csv",
        fieldnames=[
            "epoch",
            "start_step",
            "step_loss",
            "loss_data",
            "loss_haslach",
            "loss_pako",
            "loss_ic",
            "loss_bc_base",
            "loss_bc_tip",
        ],
    )

    is_ansatz = not isinstance(pinn, ParametricPINN)
    window_size = int(cfg["training"].get("window_size", 10))
    time_range = range(0, time_steps - window_size + 1, window_size)

    adam_epochs = int(cfg["training"].get("adam_epochs", 100))
    lbfgs_iters = int(cfg["training"].get("lbfgs_epochs", 50))
    global_epoch = 0

    optimizer_adam = Adam(
        pinn.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
    )
    optimizer_lbfgs = LBFGS(
        pinn.parameters(),
        lr=float(cfg["optimizer"].get("lbfgs_lr", 0.5)),
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for step_start in time_range:
        step_end = step_start + window_size
        u_prev_anchor = dataset["u"][step_start].clone().detach()

        logger.info(f"Training Window: [{step_start} : {step_end}]")

        # Adam phase
        for epoch in tqdm(range(1, adam_epochs + 1), desc=f"Win [{step_start}] Adam"):
            optimizer_adam.zero_grad()
            global_epoch += 1
            step_annealing = min(1.0, epoch / (adam_epochs * 0.7))

            batch_adam = sampler.sample_window(step_start=step_start, step_end=step_end)
            target_u_ic = u_prev_anchor * step_annealing + batch_adam.u_ic_target * (
                1.0 - step_annealing
            )

            loss, metrics = compute_loss(
                pinn=pinn,
                batch=batch_adam,
                visco_model=visco_model,
                b=b,
                loss_weights=loss_weights,
                u_ic_target=target_u_ic,
                is_ansatz=is_ansatz,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
            optimizer_adam.step()

            metrics["epoch"] = global_epoch
            metrics["start_step"] = step_start
            metrics_logger.log(metrics)

        # L-BFGS phase
        batch_lbfgs = sampler.sample_window(step_start=step_start, step_end=step_end)
        current_metrics: dict[str, float] = {}

        for _ in tqdm(range(1, lbfgs_iters + 1), desc=f"Win [{step_start}] L-BFGS"):
            global_epoch += 1

            def closure():
                optimizer_lbfgs.zero_grad()
                step_loss, step_metrics = compute_loss(
                    pinn=pinn,
                    batch=batch_lbfgs,  # noqa: B023
                    visco_model=visco_model,
                    b=b,
                    loss_weights=loss_weights,
                    u_ic_target=u_prev_anchor,  # noqa: B023
                    is_ansatz=is_ansatz,
                )
                step_loss.backward()
                current_metrics.update(step_metrics)  # noqa: B023
                return step_loss

            optimizer_lbfgs.step(closure)
            current_metrics["epoch"] = global_epoch
            current_metrics["start_step"] = step_start
            metrics_logger.log(current_metrics)

        with torch.no_grad():
            t_end_tensor = dataset["time"][min(time_steps - 1, step_end)].expand(
                mesh.n_nodes, 1
            )
            u_eval = pinn(t=t_end_tensor, X=mesh.nodes)[:, :2]
            if torch.isnan(u_eval).any():
                u_prev_anchor = (
                    dataset["u"][min(time_steps - 1, step_end)].clone().detach()
                )
            else:
                u_prev_anchor = u_eval.clone().detach()

        torch.save(
            {"model_state_dict": pinn.state_dict(), "config": cfg},
            checkpoint_dir / f"{model_name}_window_{step_start}.pt",
        )

    torch.save(
        {"model_state_dict": pinn.state_dict(), "config": cfg},
        output_dir / f"{model_name}.pt",
    )
    logger.info("Sequence-to-Sequence Training Completed.")


def main() -> None:
    init_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config("config.yaml")
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mesh
    x_range = cfg["domain"]["x_range"]
    y_range = cfg["domain"]["y_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_range[-1] - x_range[0]),
        height=float(y_range[-1] - y_range[0]),
        nx=nx,
        ny=ny,
        device=device,
    )

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    sampler = PINNSampler(dataset=dataset, cfg=cfg, mesh=mesh, device=device)
    time_steps = len(dataset["time"])

    # Physical
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

    cfg_t = cfg["training"]
    loss_weights = {
        "lambda_data": float(cfg_t["loss_weights"]["lambda_data"]),
        "lambda_haslach": float(cfg_t["loss_weights"]["lambda_haslach"]),
        "lambda_momentum": float(cfg_t["loss_weights"]["lambda_momentum"]),
        "lambda_initial": float(cfg_t["loss_weights"]["lambda_initial"]),
        "lambda_bc_base": float(cfg_t["loss_weights"].get("lambda_bc_base", 1.0)),
        "lambda_bc_tip": float(cfg_t["loss_weights"]["lambda_bc_tip"]),
    }

    # Mode Selection
    bulk_run = bool(cfg.get("bulk", False))
    selected_mode = cfg_t.get("method", "standard").lower()

    active_modes = (
        ["standard", "curriculum", "seq2seq"] if bulk_run else [selected_mode]
    )

    for mode in active_modes:
        logger.info(f"Initialising Training Pipeline for Mode: {mode.upper()}")
        pinn = build_model(cfg, device=device)

        if mode == "standard":
            run_train_standard(
                cfg=cfg,
                pinn=pinn,
                sampler=sampler,
                visco_model=visco_model,
                b=b,
                loss_weights=loss_weights,
                output_dir=output_dir,
                time_steps=time_steps,
            )
        elif mode == "curriculum":
            run_train_curriculum(
                cfg=cfg,
                pinn=pinn,
                sampler=sampler,
                visco_model=visco_model,
                b=b,
                loss_weights=loss_weights,
                output_dir=output_dir,
                time_steps=time_steps,
            )
        elif mode == "seq2seq":
            run_train_seq2seq(
                cfg=cfg,
                pinn=pinn,
                sampler=sampler,
                dataset=dataset,
                mesh=mesh,
                visco_model=visco_model,
                b=b,
                loss_weights=loss_weights,
                output_dir=output_dir,
                time_steps=time_steps,
            )
        else:
            raise ValueError(f"Unrecognised training mode: {mode}")

    # alert()
    logger.info("All Specified Training Pipelines Concluded.")


if __name__ == "__main__":
    main()
