import torch
from torch import nn

from ..geometry import Graph
from .backbones import MLP


class Encoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        latent_dim: int,
        hidden_dim: int,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        self.node_encoder = MLP(
            in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            out_dim=latent_dim,
            use_layer_norm=True,
            activation=activation,
        )
        self.edge_encoder = MLP(
            in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            out_dim=latent_dim,
            use_layer_norm=True,
            activation=activation,
        )

    def forward(
        self, node_features: torch.Tensor, edge_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        latent_nodes = self.node_encoder(node_features)
        latent_edges = self.edge_encoder(edge_features)

        return latent_nodes, latent_edges


class MessagePassingBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()

        self.edge_mlp = MLP(
            in_dim=3 * latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            out_dim=latent_dim,
            use_layer_norm=True,
            activation=activation,
        )
        self.node_mlp = MLP(
            in_dim=2 * latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            out_dim=latent_dim,
            use_layer_norm=True,
            activation=activation,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        src = node_features[senders]
        dst = node_features[receivers]
        edge_input = torch.cat([src, dst, edge_features], dim=-1)
        updated_edges = edge_features + self.edge_mlp(edge_input)

        num_nodes = node_features.shape[0]
        aggregated_messages = torch.zeros(
            (num_nodes, updated_edges.shape[-1]),
            dtype=updated_edges.dtype,
            device=updated_edges.device,
        )
        aggregated_messages.index_add_(0, receivers, updated_edges)

        node_input = torch.cat([node_features, aggregated_messages], dim=-1)
        updated_nodes = node_features + self.node_mlp(node_input)

        return updated_nodes, updated_edges


class Processor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                MessagePassingBlock(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        h_nodes, h_edges = node_features, edge_features

        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, senders, receivers)

        return h_nodes, h_edges


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        self.mlp = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            out_dim=output_dim,
            use_layer_norm=False,
            activation=activation,
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_features)


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 6,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 15,
        output_dim: int = 2,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        self.processor = Processor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    def forward(
        self,
        graph: Graph,
    ) -> torch.Tensor:

        node_features = graph.node_features
        edge_features = graph.edge_features
        senders = graph.senders
        receivers = graph.receivers

        latent_nodes, latent_edges = self.encoder(node_features, edge_features)
        latent_nodes, _ = self.processor(latent_nodes, latent_edges, senders, receivers)

        return self.decoder(latent_nodes)
