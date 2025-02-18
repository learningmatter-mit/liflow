import math

import torch
import torch.nn as nn
from torch_scatter import scatter_sum


# Modified from yang-song/score_sde_pytorch
class GaussianFourierBasis(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, num_basis: int):
        super().__init__()
        assert num_basis % 2 == 0
        self.num_basis = num_basis
        freqs = torch.randn(num_basis // 2) * 2 * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        return emb


class BesselBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float):
        super().__init__()
        self.num_basis = num_basis
        freqs = torch.arange(1, num_basis + 1, dtype=torch.float) * math.pi / r_max
        prefactor = torch.tensor(math.sqrt(2.0 / r_max), dtype=torch.float)
        self.register_buffer("freqs", freqs)
        self.register_buffer("prefactor", prefactor)

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        rbf = self.prefactor * torch.sin(args) / x[..., None]
        return rbf


class CosineCutoff(nn.Module):
    def __init__(self, r_max: float):
        super().__init__()
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float))

    def forward(self, x: torch.Tensor):
        x_cut = 0.5 * (1.0 + torch.cos(x * math.pi / self.r_max))
        x_cut = x_cut * (x < self.r_max).float()
        return x_cut


class DualMessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 4),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 4)

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
        radial_embeddings_1: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        radial_embeddings_2: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        f_cut_1: torch.Tensor,  # [n_edges, 1]
        f_cut_2: torch.Tensor,  # [n_edges, 1]
        unit_vectors_1: torch.Tensor,  # [n_edges, 3]
        unit_vectors_2: torch.Tensor,  # [n_edges, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = (
            self.linear_W(radial_embeddings_1) * f_cut_1[..., None]
            + self.linear_W(radial_embeddings_2) * f_cut_2[..., None]
        )
        x = phi[idx_j] * W
        x_s, x_vv, x_vs_1, x_vs_2 = torch.split(x, self.num_features, dim=-1)
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = (
            v[idx_j] * x_vv
            + x_vs_1 * unit_vectors_1[..., None]
            + x_vs_2 * unit_vectors_2[..., None]
        )
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class UpdateBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.mlp_a = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_UV = nn.Linear(num_features, num_features * 2, bias=False)

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        U_v, V_v = torch.split(self.linear_UV(v), self.num_features, dim=-1)
        a = self.mlp_a(torch.cat((s, V_v.norm(p=2, dim=-2, keepdim=True)), dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)
        dv = a_vv * U_v
        ds = a_ss + a_sv * torch.sum(U_v * V_v, dim=-2, keepdim=True)
        return s + ds, v + dv


class GatedEquivariantBlock(nn.Module):
    """Modified gated equivariant block to output a single vector.
    See PaiNN paper Fig. 3 or schnetpack.nn.equivariant module"""

    def __init__(
        self,
        num_scalar_inputs: int,
        num_vector_inputs: int,
    ):
        super().__init__()
        self.num_scalar_inputs = num_scalar_inputs
        self.num_vector_inputs = num_vector_inputs
        self.linear_v = nn.Linear(num_vector_inputs, 2, bias=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_scalar_inputs + 1, num_scalar_inputs + 1),
            nn.SiLU(),
            nn.Linear(num_scalar_inputs + 1, 1),
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        W_v1, W_v2 = torch.split(self.linear_v(v), 1, dim=-1)
        s_out = self.mlp_s(torch.cat((s, W_v2.norm(dim=-2, keepdim=True)), dim=-1))
        v_out = W_v1 * s_out
        return v_out
