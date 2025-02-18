import torch
import torch.nn as nn
from torch_geometric.data import Data

from liflow.model.layers import (
    BesselBasis,
    CosineCutoff,
    DualMessageBlock,
    GatedEquivariantBlock,
    GaussianFourierBasis,
    UpdateBlock,
)
from liflow.utils.geometry import get_unit_vectors_and_lengths


class DualPaiNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_radial_basis: int,
        num_layers: int,
        num_elements: int,
        r_max: float,
        r_offset: float = 0.0,
        ref_temp: float = 1000.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = r_max
        self.r_offset = r_offset
        self.ref_temp = ref_temp  # reference temperature for scaling

        assert num_features % 2 == 0, "Number of features must be even"
        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.temp_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.radial_embedding = BesselBasis(num_basis=num_radial_basis, r_max=r_max)
        self.cutoff_fn = CosineCutoff(r_max=r_max)
        self.linear_v = nn.Linear(1, num_features, bias=False)

        messages, updates = [], []
        for _ in range(num_layers):
            messages.append(DualMessageBlock(num_features, num_radial_basis))
            updates.append(UpdateBlock(num_features))
        self.messages = nn.ModuleList(messages)
        self.updates = nn.ModuleList(updates)
        self.output_block = GatedEquivariantBlock(
            num_scalar_inputs=num_features,
            num_vector_inputs=num_features,
        )

    def forward(self, data: Data):
        unit_vectors_1, lengths_1 = get_unit_vectors_and_lengths(
            data["positions_1"], data["edge_index"], data["shifts"]
        )
        unit_vectors_2, lengths_2 = get_unit_vectors_and_lengths(
            data["positions_2"], data["edge_index"], data["shifts"]
        )

        # Compute radial basis functions
        lengths_1 = (lengths_1 + self.r_offset).clamp(max=self.r_max)
        lengths_2 = (lengths_2 + self.r_offset).clamp(max=self.r_max)
        radial_embeddings_1 = self.radial_embedding(lengths_1)
        radial_embeddings_2 = self.radial_embedding(lengths_2)
        f_cut_1 = self.cutoff_fn(lengths_1)
        f_cut_2 = self.cutoff_fn(lengths_2)

        # Compute initial scalar and vector features
        s_atom = self.atom_embedding(data["elements"])
        s_time = self.time_embedding(data["time"])
        s_temp = self.temp_embedding(data["temp"] / self.ref_temp)
        s = s_atom + torch.cat([s_time, s_temp], dim=-1)
        s = s[:, None, :]  # [n_nodes, 1, n_feats]
        # v = torch.zeros_like(s).repeat(1, 3, 1)  # [n_nodes, 3, n_feats]
        positions_diff = data["positions_2"] - data["positions_1"]
        v = self.linear_v(positions_diff[..., None])  # [n_nodes, 3, n_feats]

        for message, update in zip(self.messages, self.updates):
            s, v = message(
                s,
                v,
                radial_embeddings_1,
                radial_embeddings_2,
                f_cut_1,
                f_cut_2,
                unit_vectors_1,
                unit_vectors_2,
                data["edge_index"],
            )
            s, v = update(s, v)
        v_out = self.output_block(s, v).squeeze(-1)
        return v_out
