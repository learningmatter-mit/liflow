import numpy as np
import torch
from vesin import NeighborList


def get_unit_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_i, idx_j = edge_index[0], edge_index[1]  # [n_edges]
    vectors = positions[idx_j] - positions[idx_i] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    unit_vectors = vectors / (lengths + eps)
    return unit_vectors, lengths


def get_neighbor_list_batch(
    positions_batch: np.ndarray | list[np.ndarray],  # [n_configs, n_atoms, 3]
    lattice: np.ndarray,  # [3, 3]
    cutoff: float = 5.0,
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct neighbor list for a batch of coordinates, as a union of all
    neighbor lists for each configuration in the batch."""
    nl = NeighborList(cutoff=cutoff, full_list=True)
    edge_attrs_all = []
    for positions in positions_batch:
        src, dst, shifts = nl.compute(
            quantities="ijS",
            points=positions,
            box=lattice,
            periodic=periodic,
        )
        edge_attrs = np.hstack((src[:, None], dst[:, None], shifts)).astype(int)
        edge_attrs_all.append(edge_attrs)
    edge_attrs = np.unique(np.vstack(edge_attrs_all), axis=0)
    edge_index = edge_attrs[:, :2].T
    shifts = edge_attrs[:, 2:] @ lattice  # convert shifts to Cartesian

    return edge_index, shifts


def unwrap_trajectory(
    positions: np.ndarray,  # [n_frames, n_atoms, 3]
    lattice: np.ndarray,  # [3, 3]
) -> np.ndarray:
    """Unwraps a trajectory of atomic positions. This function assumes that the
    positions do not jump more than half the box length between frames."""
    frac_diff = np.diff(positions, axis=0) @ np.linalg.inv(lattice)
    frac_diff_unwrap = frac_diff - np.floor(frac_diff + 0.5)
    diff_unwrap = frac_diff_unwrap @ lattice
    return np.cumsum(np.concatenate((positions[0, None], diff_unwrap), axis=0), axis=0)
