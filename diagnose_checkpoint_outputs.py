import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.geometry import Graph, Mesh, create_graph, create_mesh_3D
from src.loader import MGNData
from src.models import MeshGraphNet
from src.physics import Kinematics, Ogden
from src.physics.operators import compute_internal_nodal_forces
from src.utils import stencil_grad, tetrahedral_shape_gradients


CHECKPOINT_EPOCH_PATTERN = re.compile(r"_(\d+)\.pt$")


def next_available_path(path: Path) -> Path:
    candidate = path
    version = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{version}{path.suffix}")
        version += 1
    return candidate


def build_node_type(
    mesh: Mesh, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    directions = ("left", "right", "bottom", "top", "back", "front")
    node_type = torch.zeros(
        (mesh.n_nodes, len(directions)), device=device, dtype=dtype
    )
    for index, direction in enumerate(directions):
        node_type[mesh.get_boundary_nodes(direction), index] = 1.0
    return node_type


def rms(values: torch.Tensor) -> float:
    return values.square().mean().sqrt().item()


def measure_output(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    free_dofs: torch.Tensor,
) -> dict[str, float]:
    error = prediction - reference
    prediction_flat = prediction.reshape(-1)
    error_flat = error.reshape(-1)

    metrics = {
        "predicted_rms_m": rms(prediction),
        "predicted_free_dof_rms_m": rms(prediction_flat[free_dofs]),
        "predicted_max_abs_m": prediction.abs().max().item(),
        "reference_rms_m": rms(reference),
        "error_rms_m": rms(error),
        "error_free_dof_rms_m": rms(error_flat[free_dofs]),
        "error_max_abs_m": error.abs().max().item(),
    }

    for axis, name in enumerate(("x", "y", "z")):
        metrics[f"predicted_{name}_mean_m"] = prediction[:, axis].mean().item()
        metrics[f"predicted_{name}_rms_m"] = rms(prediction[:, axis])
        metrics[f"reference_{name}_rms_m"] = rms(reference[:, axis])
        metrics[f"error_{name}_rms_m"] = rms(error[:, axis])

    return metrics


def save_rest_field(
    prediction: torch.Tensor,
    internal_force: torch.Tensor,
    projected_internal_force: torch.Tensor,
    mesh: Mesh,
    *,
    epoch: int,
    output_path: Path,
) -> None:
    prediction = prediction.detach().cpu()
    internal_force = internal_force.detach().cpu()
    projected_internal_force = projected_internal_force.detach().cpu()
    nodes = mesh.nodes.detach().cpu()
    fixed_dofs = mesh.fixed_dofs.detach().cpu()
    fixed_dof_mask = torch.zeros(
        (mesh.n_nodes, 3), dtype=torch.bool
    )
    fixed_dof_mask.reshape(-1)[fixed_dofs] = True
    fixed_nodes = fixed_dof_mask.all(dim=1)
    free_nodes = ~fixed_nodes

    free_mean = prediction[free_nodes].mean(dim=0)
    all_mean = prediction.mean(dim=0)
    deformation = prediction - free_mean
    free_field_rms = prediction[free_nodes].square().sum(dim=-1).mean().sqrt()
    free_deformation_rms = (
        deformation[free_nodes].square().sum(dim=-1).mean().sqrt()
    )
    fixed_field_rms = prediction[fixed_nodes].square().sum(dim=-1).mean().sqrt()

    field_data = {
        "node_id": torch.arange(mesh.n_nodes).numpy(),
        "x_m": nodes[:, 0].numpy(),
        "y_m": nodes[:, 1].numpy(),
        "z_m": nodes[:, 2].numpy(),
        "node_class": [
            "fixed" if is_fixed else "free"
            for is_fixed in fixed_nodes.tolist()
        ],
        "u_x_mm": prediction[:, 0].numpy() * 1000.0,
        "u_y_mm": prediction[:, 1].numpy() * 1000.0,
        "u_z_mm": prediction[:, 2].numpy() * 1000.0,
        "u_magnitude_mm": torch.linalg.vector_norm(
            prediction, dim=-1
        ).numpy()
        * 1000.0,
        "deformation_x_mm": deformation[:, 0].numpy() * 1000.0,
        "deformation_y_mm": deformation[:, 1].numpy() * 1000.0,
        "deformation_z_mm": deformation[:, 2].numpy() * 1000.0,
        "deformation_magnitude_mm": torch.linalg.vector_norm(
            deformation, dim=-1
        ).numpy()
        * 1000.0,
        "internal_force_x_N": internal_force[:, 0].numpy(),
        "internal_force_y_N": internal_force[:, 1].numpy(),
        "internal_force_z_N": internal_force[:, 2].numpy(),
        "internal_force_magnitude_N": torch.linalg.vector_norm(
            internal_force, dim=-1
        ).numpy(),
        "projected_internal_force_x_N": projected_internal_force[:, 0].numpy(),
        "projected_internal_force_y_N": projected_internal_force[:, 1].numpy(),
        "projected_internal_force_z_N": projected_internal_force[:, 2].numpy(),
        "projected_internal_force_magnitude_N": torch.linalg.vector_norm(
            projected_internal_force, dim=-1
        ).numpy(),
        "projection_delta_force_x_N": (
            projected_internal_force[:, 0] - internal_force[:, 0]
        ).numpy(),
        "projection_delta_force_y_N": (
            projected_internal_force[:, 1] - internal_force[:, 1]
        ).numpy(),
        "projection_delta_force_z_N": (
            projected_internal_force[:, 2] - internal_force[:, 2]
        ).numpy(),
    }
    node_csv_path = output_path.with_name(
        f"{output_path.stem}_rest_field_epoch_{epoch}.csv"
    )
    pd.DataFrame(field_data).to_csv(node_csv_path, index=False)

    figure_path = node_csv_path.with_suffix(".png")
    figure = plt.figure(figsize=(14, 11))
    components = (
        (0, "x displacement", prediction[:, 0].numpy() * 1000.0, "coolwarm"),
        (1, "y displacement", prediction[:, 1].numpy() * 1000.0, "coolwarm"),
        (2, "z displacement", prediction[:, 2].numpy() * 1000.0, "coolwarm"),
        (
            3,
            "displacement magnitude",
            torch.linalg.vector_norm(prediction, dim=-1).numpy() * 1000.0,
            "viridis",
        ),
    )
    for plot_index, (component, title, values, cmap) in enumerate(components, 1):
        axis = figure.add_subplot(2, 2, plot_index, projection="3d")
        if component < 3:
            limit = max(float(abs(values).max()), 1e-6)
            scatter = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                nodes[:, 2],
                c=values,
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                s=30,
            )
        else:
            scatter = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                nodes[:, 2],
                c=values,
                cmap=cmap,
                s=30,
            )
        if fixed_nodes.any():
            axis.scatter(
                nodes[fixed_nodes, 0],
                nodes[fixed_nodes, 1],
                nodes[fixed_nodes, 2],
                s=58,
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
                label="constrained nodes",
            )
            axis.legend(loc="upper left")
        figure.colorbar(scatter, ax=axis, shrink=0.7, pad=0.08, label="mm")
        axis.set_title(title)
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.set_box_aspect((mesh.width, mesh.height, mesh.depth))

    figure.suptitle(f"Rest-input prediction at checkpoint epoch {epoch}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    translation_magnitude = torch.linalg.vector_norm(free_mean).item() * 1000.0
    free_field_rms_mm = free_field_rms.item() * 1000.0
    free_deformation_rms_mm = free_deformation_rms.item() * 1000.0
    fixed_field_rms_mm = fixed_field_rms.item() * 1000.0
    print(f"Saved node-wise rest field: {node_csv_path}")
    print(f"Saved rest-field plot: {figure_path}")
    print(
        "Rest-field decomposition [mm]: "
        f"mean over free nodes=({free_mean[0].item() * 1000.0:.4f}, "
        f"{free_mean[1].item() * 1000.0:.4f}, "
        f"{free_mean[2].item() * 1000.0:.4f}), "
        f"|mean|={translation_magnitude:.4f}, "
        f"free-field RMS={free_field_rms_mm:.4f}, "
        f"free deformation RMS after mean removal={free_deformation_rms_mm:.4f}, "
        f"fixed-node RMS={fixed_field_rms_mm:.4f}"
    )
    print(
        "Mean over all nodes [mm]: "
        f"({all_mean[0].item() * 1000.0:.4f}, "
        f"{all_mean[1].item() * 1000.0:.4f}, "
        f"{all_mean[2].item() * 1000.0:.4f})"
    )


def compute_static_internal_force(
    prediction: torch.Tensor,
    mesh: Mesh,
    dN_dX: torch.Tensor,
    physics: dict,
) -> torch.Tensor:
    u = torch.tensor(
        prediction.detach().cpu().numpy(),
        device=mesh.nodes.device,
        dtype=mesh.nodes.dtype,
        requires_grad=True,
    )
    grad_u = stencil_grad(u, mesh.elements, dN_dX)
    kin = Kinematics(grad_u)
    tissue = physics["tissue"]
    ogden = Ogden(
        mus=torch.tensor(
            [float(mu) for mu in physics["mus"][tissue]],
            device=mesh.nodes.device,
            dtype=mesh.nodes.dtype,
        ),
        alphas=torch.tensor(
            physics["alphas"][tissue],
            device=mesh.nodes.device,
            dtype=mesh.nodes.dtype,
        ),
        beta=float(physics["beta"]),
        lam=float(physics["lam"]),
        kin=kin,
    )
    ogden.get_psi()
    P = ogden.get_P()
    return compute_internal_nodal_forces(P, mesh, dN_dX).detach()


def save_static_force_diagnostic(
    internal_force: torch.Tensor,
    projected_force: torch.Tensor,
    reference_force: torch.Tensor,
    external_force: torch.Tensor,
    mesh: Mesh,
    *,
    epoch: int,
    output_path: Path,
) -> None:
    internal_force = internal_force.detach().cpu()
    projected_force = projected_force.detach().cpu()
    reference_force = reference_force.detach().cpu()
    external_force = external_force.detach().cpu()
    fixed_dofs = mesh.fixed_dofs.detach().cpu()
    free_dofs = mesh.free_dofs.detach().cpu()
    fixed_dof_mask = torch.zeros(3 * mesh.n_nodes, dtype=torch.bool)
    fixed_dof_mask[fixed_dofs] = True
    fixed_nodes = fixed_dof_mask.reshape(mesh.n_nodes, 3).all(dim=1)
    free_nodes = ~fixed_nodes
    force_scale = external_force.reshape(-1)[free_dofs].square().mean().sqrt()
    if force_scale <= 0.0:
        raise ValueError("First-loaded-step external force scale must be positive.")
    raw_free_force = internal_force.reshape(-1)[free_dofs]
    projected_free_force = projected_force.reshape(-1)[free_dofs]
    projection_delta = projected_force - internal_force
    projection_delta_free = projection_delta.reshape(-1)[free_dofs]
    raw_free_rms = raw_free_force.square().mean().sqrt()
    projected_free_rms = projected_free_force.square().mean().sqrt()

    summaries = []
    for state_name, force in (
        ("network_rest_prediction", internal_force),
        ("fixed_nodes_projected_to_zero", projected_force),
        ("zero_displacement_reference", reference_force),
    ):
        force_flat = force.reshape(-1)
        free_force = force_flat[free_dofs]
        fixed_force = force_flat[fixed_dofs]
        nodal_magnitude = torch.linalg.vector_norm(force, dim=-1)
        summaries.append(
            {
                "epoch": epoch,
                "state": state_name,
                "internal_force_rms_all_dofs_N": (
                    force_flat.square().mean().sqrt().item()
                ),
                "internal_force_rms_free_dofs_N": (
                    free_force.square().mean().sqrt().item()
                ),
                "internal_force_rms_fixed_dofs_N": (
                    fixed_force.square().mean().sqrt().item()
                ),
                "internal_force_rms_free_nodes_N": (
                    nodal_magnitude[free_nodes].square().mean().sqrt().item()
                ),
                "internal_force_rms_fixed_nodes_N": (
                    nodal_magnitude[fixed_nodes].square().mean().sqrt().item()
                ),
                "internal_force_max_free_node_N": (
                    nodal_magnitude[free_nodes].max().item()
                ),
                "internal_force_max_fixed_node_N": (
                    nodal_magnitude[fixed_nodes].max().item()
                ),
                "net_internal_force_x_N": force[:, 0].sum().item(),
                "net_internal_force_y_N": force[:, 1].sum().item(),
                "net_internal_force_z_N": force[:, 2].sum().item(),
                "external_force_rms_free_dofs_N": force_scale.item(),
                "internal_to_external_free_rms_ratio": (
                    free_force.square().mean().sqrt() / force_scale
                ).item(),
                "raw_to_projected_delta_free_dof_rms_N": (
                    projection_delta_free.square().mean().sqrt().item()
                ),
                "projected_to_raw_free_dof_rms_ratio": (
                    projected_free_rms / raw_free_rms
                ).item(),
                "projected_free_dof_rms_change_percent": (
                    100.0 * (projected_free_rms / raw_free_rms - 1.0)
                ).item(),
            }
        )

    summary_path = output_path.with_name(
        f"{output_path.stem}_static_internal_force_epoch_{epoch}_summary.csv"
    )
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    nodes = mesh.nodes.detach().cpu()
    force_fields = (
        (
            internal_force,
            summary_path.with_suffix(".png"),
            "Raw network rest-field internal force",
        ),
        (
            projected_force,
            summary_path.with_name(
                f"{summary_path.stem}_boundary_projected.png"
            ),
            "Internal force after zeroing constrained displacements",
        ),
        (
            projection_delta,
            summary_path.with_name(
                f"{summary_path.stem}_projection_delta.png"
            ),
            "Force change caused by boundary projection",
        ),
    )
    for force, figure_path, figure_title in force_fields:
        force_values = force.numpy()
        figure = plt.figure(figsize=(14, 11))
        components = (
            (0, "internal force x", force_values[:, 0], "coolwarm"),
            (1, "internal force y", force_values[:, 1], "coolwarm"),
            (2, "internal force z", force_values[:, 2], "coolwarm"),
            (
                3,
                "internal force magnitude",
                torch.linalg.vector_norm(force, dim=-1).numpy(),
                "viridis",
            ),
        )
        for plot_index, (component, title, values, cmap) in enumerate(
            components, 1
        ):
            axis = figure.add_subplot(2, 2, plot_index, projection="3d")
            if component < 3:
                limit = max(float(abs(values).max()), 1e-8)
                scatter = axis.scatter(
                    nodes[:, 0],
                    nodes[:, 1],
                    nodes[:, 2],
                    c=values,
                    cmap=cmap,
                    vmin=-limit,
                    vmax=limit,
                    s=30,
                )
            else:
                scatter = axis.scatter(
                    nodes[:, 0],
                    nodes[:, 1],
                    nodes[:, 2],
                    c=values,
                    cmap=cmap,
                    s=30,
                )
            if fixed_nodes.any():
                axis.scatter(
                    nodes[fixed_nodes, 0],
                    nodes[fixed_nodes, 1],
                    nodes[fixed_nodes, 2],
                    s=58,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                    label="constrained nodes",
                )
                axis.legend(loc="upper left")
            figure.colorbar(scatter, ax=axis, shrink=0.7, pad=0.08, label="N")
            axis.set_title(title)
            axis.set_xlabel("X [m]")
            axis.set_ylabel("Y [m]")
            axis.set_zlabel("Z [m]")
            axis.set_box_aspect((mesh.width, mesh.height, mesh.depth))

        figure.suptitle(f"{figure_title}, checkpoint epoch {epoch}")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=170)
        plt.close(figure)

    predicted_summary = summaries[0]
    projected_summary = summaries[1]
    print(f"Saved static internal-force summary: {summary_path}")
    print(
        "Internal force RMS on free DOFs [N]: "
        f"free DOFs={predicted_summary['internal_force_rms_free_dofs_N']:.6g}, "
        f"after projection="
        f"{projected_summary['internal_force_rms_free_dofs_N']:.6g}, "
        "zero-reference free DOF RMS="
        f"{summaries[2]['internal_force_rms_free_dofs_N']:.6g}"
    )
    print(
        "Boundary-projection change on free DOFs: "
        f"{projected_summary['projected_free_dof_rms_change_percent']:.3f}% "
        f"(delta RMS={projected_summary['raw_to_projected_delta_free_dof_rms_N']:.6g} N)"
    )


def save_rest_feature_ablation(
    model: MeshGraphNet,
    graph: Graph,
    mesh: Mesh,
    *,
    epoch: int,
    output_path: Path,
) -> None:
    node_type_feature_count = 6
    fixed_dof_mask = torch.zeros(
        3 * mesh.n_nodes, dtype=torch.bool, device=mesh.nodes.device
    )
    fixed_dof_mask[mesh.fixed_dofs] = True
    fixed_nodes = fixed_dof_mask.reshape(mesh.n_nodes, 3).all(dim=1)
    free_nodes = ~fixed_nodes
    free_dofs = mesh.free_dofs
    node_features = graph.node_features
    edge_features = graph.edge_features

    node_types_removed = node_features.clone()
    node_types_removed[:, :node_type_feature_count] = 0.0
    no_geometry = torch.zeros_like(edge_features)
    no_node_types_or_geometry = node_types_removed.clone()

    variants: list[tuple[str, Graph]] = [
        ("baseline_rest_graph", graph),
        (
            "node_type_features_zeroed",
            Graph(
                mesh_nodes=graph.mesh_nodes,
                senders=graph.senders,
                receivers=graph.receivers,
                node_features=node_types_removed,
                edge_features=edge_features,
            ),
        ),
        (
            "edge_geometry_zeroed",
            Graph(
                mesh_nodes=graph.mesh_nodes,
                senders=graph.senders,
                receivers=graph.receivers,
                node_features=node_features,
                edge_features=no_geometry,
            ),
        ),
        (
            "node_types_and_geometry_zeroed",
            Graph(
                mesh_nodes=graph.mesh_nodes,
                senders=graph.senders,
                receivers=graph.receivers,
                node_features=no_node_types_or_geometry,
                edge_features=no_geometry,
            ),
        ),
    ]

    summaries: list[dict[str, float | int | str]] = []
    node_outputs: dict[str, torch.Tensor] = {}
    baseline_free = None
    with torch.inference_mode():
        for variant_name, variant_graph in variants:
            prediction = model(variant_graph)
            node_outputs[variant_name] = prediction.detach().cpu()
            free_prediction = prediction[free_nodes]
            free_mean = free_prediction.mean(dim=0)
            centered_free = free_prediction - free_mean
            free_rms = free_prediction.square().sum(dim=-1).mean().sqrt()
            centered_rms = centered_free.square().sum(dim=-1).mean().sqrt()
            fixed_rms = (
                prediction[fixed_nodes].square().sum(dim=-1).mean().sqrt()
            )
            flat_free = prediction.reshape(-1)[free_dofs]
            if baseline_free is None:
                baseline_free = flat_free.detach().clone()
                difference_rms = torch.zeros((), device=prediction.device)
            else:
                difference_rms = (
                    flat_free - baseline_free
                ).square().mean().sqrt()

            summaries.append(
                {
                    "epoch": epoch,
                    "variant": variant_name,
                    "free_mean_x_mm": free_mean[0].item() * 1000.0,
                    "free_mean_y_mm": free_mean[1].item() * 1000.0,
                    "free_mean_z_mm": free_mean[2].item() * 1000.0,
                    "free_mean_magnitude_mm": (
                        torch.linalg.vector_norm(free_mean).item() * 1000.0
                    ),
                    "free_vector_rms_mm": free_rms.item() * 1000.0,
                    "free_spatial_residual_rms_mm": (
                        centered_rms.item() * 1000.0
                    ),
                    "fixed_vector_rms_mm": fixed_rms.item() * 1000.0,
                    "difference_from_baseline_free_dof_rms_mm": (
                        difference_rms.item() * 1000.0
                    ),
                }
            )

        final_layer = model.decoder.mlp.output_layer[0]
        final_bias = final_layer.bias
        if final_bias is not None:
            saved_bias = final_bias.detach().clone()
            try:
                final_bias.zero_()
                prediction = model(graph)
            finally:
                final_bias.copy_(saved_bias)
            node_outputs["final_decoder_bias_zeroed"] = prediction.detach().cpu()
            free_prediction = prediction[free_nodes]
            free_mean = free_prediction.mean(dim=0)
            summaries.append(
                {
                    "epoch": epoch,
                    "variant": "final_decoder_bias_zeroed",
                    "free_mean_x_mm": free_mean[0].item() * 1000.0,
                    "free_mean_y_mm": free_mean[1].item() * 1000.0,
                    "free_mean_z_mm": free_mean[2].item() * 1000.0,
                    "free_mean_magnitude_mm": (
                        torch.linalg.vector_norm(free_mean).item() * 1000.0
                    ),
                    "free_vector_rms_mm": (
                        free_prediction.square().sum(dim=-1).mean().sqrt().item()
                        * 1000.0
                    ),
                    "free_spatial_residual_rms_mm": (
                        (free_prediction - free_mean)
                        .square()
                        .sum(dim=-1)
                        .mean()
                        .sqrt()
                        .item()
                        * 1000.0
                    ),
                    "fixed_vector_rms_mm": (
                        prediction[fixed_nodes]
                        .square()
                        .sum(dim=-1)
                        .mean()
                        .sqrt()
                        .item()
                        * 1000.0
                    ),
                    "difference_from_baseline_free_dof_rms_mm": (
                        prediction.reshape(-1)[free_dofs] - baseline_free
                    )
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                    * 1000.0,
                }
            )

    summary_path = output_path.with_name(
        f"{output_path.stem}_rest_feature_ablation_epoch_{epoch}_summary.csv"
    )
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    nodes = mesh.nodes.detach().cpu()
    node_data: dict[str, object] = {
        "node_id": torch.arange(mesh.n_nodes).numpy(),
        "x_m": nodes[:, 0].numpy(),
        "y_m": nodes[:, 1].numpy(),
        "z_m": nodes[:, 2].numpy(),
        "node_class": [
            "fixed" if is_fixed else "free"
            for is_fixed in fixed_nodes.detach().cpu().tolist()
        ],
    }
    for variant_name, prediction in node_outputs.items():
        for axis, component in enumerate(("x", "y", "z")):
            node_data[f"{variant_name}_u_{component}_mm"] = (
                prediction[:, axis].numpy() * 1000.0
            )
    node_path = output_path.with_name(
        f"{output_path.stem}_rest_feature_ablation_epoch_{epoch}_nodes.csv"
    )
    pd.DataFrame(node_data).to_csv(node_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary_frame = pd.DataFrame(summaries)
    axes[0].bar(
        summary_frame["variant"],
        summary_frame["free_mean_x_mm"],
        color="tab:blue",
    )
    axes[0].set_title("Mean x displacement on free nodes")
    axes[0].set_ylabel("mm")
    axes[1].bar(
        summary_frame["variant"],
        summary_frame["free_spatial_residual_rms_mm"],
        color="tab:orange",
    )
    axes[1].set_title("Free-field RMS after removing its mean")
    axes[1].set_ylabel("mm")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure_path = summary_path.with_suffix(".png")
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(f"Saved rest-feature ablation summary: {summary_path}")
    print(f"Saved rest-feature ablation node outputs: {node_path}")
    print(f"Saved rest-feature ablation plot: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fixed-input model outputs across saved training checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        required=True,
        help="Directory containing epoch checkpoints named *_<epoch>.pt.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (defaults to the device saved in the checkpoint config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path; defaults to a versioned file in the run's diagnostics folder.",
    )
    parser.add_argument(
        "--field-epoch",
        type=int,
        default=None,
        help="Checkpoint epoch for the node-wise rest-field plot (defaults to the latest).",
    )
    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    checkpoint_files = []
    for path in checkpoints_dir.glob("*.pt"):
        match = CHECKPOINT_EPOCH_PATTERN.search(path.name)
        if match:
            checkpoint_files.append((int(match.group(1)), path))
    checkpoint_files.sort(key=lambda item: item[0])
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No epoch checkpoints named *_<epoch>.pt found in {checkpoints_dir}"
        )
    available_epochs = {epoch for epoch, _ in checkpoint_files}
    field_epoch = (
        checkpoint_files[-1][0] if args.field_epoch is None else args.field_epoch
    )
    if field_epoch not in available_epochs:
        raise ValueError(
            f"No checkpoint for field epoch {field_epoch}; "
            f"available epochs: {sorted(available_epochs)}"
        )

    first_checkpoint = torch.load(
        checkpoint_files[0][1], map_location="cpu", weights_only=False
    )
    cfg = first_checkpoint.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Checkpoint does not contain a training config: {checkpoint_files[0][1]}"
        )

    configured_device = cfg.get("training", {}).get("device", "cpu")
    device = torch.device(args.device or configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({device}) but is not available. "
            "Retry with --device cpu."
        )
    dtype = torch.float32

    domain = cfg["domain"]
    x_min, x_max = domain["x_range"]
    y_min, y_max = domain["y_range"]
    z_min, z_max = domain["z_range"]
    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=int(domain["nx"]),
        ny=int(domain["ny"]),
        nz=int(domain["nz"]),
        device=device,
        dtype=dtype,
    )
    dN_dX, _ = tetrahedral_shape_gradients(mesh.nodes, mesh.elements)

    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    dataset = torch.load(data_path, map_location="cpu", weights_only=False)
    data_loader = MGNData(
        dataset=dataset,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )
    data = data_loader.dataset
    for key in ("u", "trac_ext"):
        if key not in data:
            raise KeyError(f"Dataset is missing required field: {key}")

    traction_norm = torch.linalg.vector_norm(data["trac_ext"], dim=-1)
    loaded_steps = torch.nonzero(traction_norm > 0.0, as_tuple=False).flatten()
    if loaded_steps.numel() == 0:
        raise ValueError("Dataset contains no non-zero external traction.")
    first_loaded_step = int(loaded_steps[0].item())

    model_cfg = cfg["model"]
    model = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)

    node_type = build_node_type(mesh, device, dtype)
    zero_state = torch.zeros_like(mesh.nodes)
    zero_traction = torch.zeros_like(mesh.nodes)
    rest_graph = create_graph(
        mesh=mesh,
        world_pos=mesh.nodes,
        node_type=node_type,
        velocity=zero_state,
        boundary_traction=zero_traction,
        device=device,
        dtype=dtype,
    )

    sigma_max = float(cfg["physics"]["sigma_max"])
    if sigma_max == 0.0:
        raise ValueError("physics.sigma_max must be non-zero to scale traction.")
    right_nodes = mesh.get_boundary_nodes("right")
    loaded_traction = torch.zeros_like(mesh.nodes)
    loaded_traction[right_nodes] = (
        data["trac_ext"][first_loaded_step] / sigma_max
    )
    first_load_graph = create_graph(
        mesh=mesh,
        world_pos=mesh.nodes,
        node_type=node_type,
        velocity=zero_state,
        boundary_traction=loaded_traction,
        device=device,
        dtype=dtype,
    )
    probes = (
        ("rest", rest_graph, data["u"][0]),
        (
            "first_load_zero_history",
            first_load_graph,
            data["u"][first_loaded_step],
        ),
    )

    rows: list[dict[str, float | int | str]] = []
    rest_field: torch.Tensor | None = None
    config_keys = ("model", "domain", "data", "physics")
    free_dofs = mesh.free_dofs
    model.eval()

    with torch.inference_mode():
        for epoch, checkpoint_path in checkpoint_files:
            checkpoint = (
                first_checkpoint
                if checkpoint_path == checkpoint_files[0][1]
                else torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            )
            checkpoint_cfg = checkpoint.get("config")
            if not isinstance(checkpoint_cfg, dict):
                raise ValueError(
                    f"Checkpoint does not contain a training config: {checkpoint_path}"
                )
            for key in config_keys:
                if checkpoint_cfg.get(key) != cfg.get(key):
                    raise ValueError(
                        f"Checkpoint config mismatch for '{key}' in {checkpoint_path}"
                    )

            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                raise KeyError(
                    f"Checkpoint is missing model_state_dict: {checkpoint_path}"
                )
            model.load_state_dict(state_dict)

            for probe_name, graph, reference in probes:
                prediction = model(graph)
                if epoch == field_epoch and probe_name == "rest":
                    rest_field = prediction.detach().cpu()
                rows.append(
                    {
                        "epoch": epoch,
                        "checkpoint": checkpoint_path.name,
                        "probe": probe_name,
                        "first_loaded_step": first_loaded_step,
                        "first_loaded_time_s": data_loader.time[
                            first_loaded_step
                        ].item(),
                        "input_traction_norm_pa": (
                            0.0
                            if probe_name == "rest"
                            else traction_norm[first_loaded_step].item()
                        ),
                        **measure_output(prediction, reference, free_dofs),
                    }
                )

    results = pd.DataFrame(rows)
    if args.output is None:
        diagnostics_dir = checkpoints_dir.parent / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        output_path = next_available_path(
            diagnostics_dir
            / f"{checkpoints_dir.parent.name}_checkpoint_outputs.csv"
        )
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    figure_path = output_path.with_suffix(".png")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for probe_name, probe_data in results.groupby("probe", sort=False):
        axes[0].plot(
            probe_data["epoch"],
            probe_data["predicted_rms_m"] * 1000.0,
            marker="o",
            label=probe_name,
        )
        axes[1].plot(
            probe_data["epoch"],
            probe_data["error_rms_m"] * 1000.0,
            marker="o",
            label=probe_name,
        )

    axes[0].set_title("Predicted displacement")
    axes[0].set_ylabel("RMS (mm)")
    axes[1].set_title("Error against reference")
    axes[1].set_ylabel("RMS (mm)")
    for axis in axes:
        axis.set_xlabel("Training epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    if rest_field is None:
        raise RuntimeError(
            f"Rest-field output was not captured for checkpoint epoch {field_epoch}."
        )
    internal_force = compute_static_internal_force(
        rest_field,
        mesh,
        dN_dX,
        cfg["physics"],
    )
    projected_field = torch.tensor(
        rest_field.detach().cpu().numpy(),
        device=device,
        dtype=dtype,
    )
    projected_field.reshape(-1)[mesh.fixed_dofs] = 0.0
    projected_internal_force = compute_static_internal_force(
        projected_field,
        mesh,
        dN_dX,
        cfg["physics"],
    )
    save_rest_field(
        rest_field,
        internal_force,
        projected_internal_force,
        mesh,
        epoch=field_epoch,
        output_path=output_path,
    )
    reference_force = compute_static_internal_force(
        torch.zeros_like(rest_field),
        mesh,
        dN_dX,
        cfg["physics"],
    )
    save_static_force_diagnostic(
        internal_force,
        projected_internal_force,
        reference_force,
        data["F_ext"][first_loaded_step],
        mesh,
        epoch=field_epoch,
        output_path=output_path,
    )
    field_checkpoint_path = next(
        path for epoch, path in checkpoint_files if epoch == field_epoch
    )
    field_checkpoint = torch.load(
        field_checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(field_checkpoint["model_state_dict"])
    save_rest_feature_ablation(
        model,
        rest_graph,
        mesh,
        epoch=field_epoch,
        output_path=output_path,
    )

    print(f"Evaluated {len(checkpoint_files)} checkpoints.")
    print(
        f"First loaded dataset step: {first_loaded_step} "
        f"(t={data_loader.time[first_loaded_step].item():.6g} s)"
    )
    print(f"Saved metrics: {output_path}")
    print(f"Saved plot: {figure_path}")


if __name__ == "__main__":
    main()
