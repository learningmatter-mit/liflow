import numpy as np
from vesin import NeighborList


def calculate_msd(
    traj: np.ndarray,  # [n_frame, n_atom, 3]
    atom_mask: np.ndarray | None = None,  # [n_atom]
    final_only: bool = True,
) -> np.ndarray | float:
    squared_displacements = np.sum((traj - traj[0]) ** 2, axis=-1)
    if atom_mask is not None:
        squared_displacements = squared_displacements[:, atom_mask]
    if final_only:
        return np.mean(squared_displacements[-1])
    else:
        return np.mean(squared_displacements, axis=-1)


def calculate_average_rdf(
    traj: np.ndarray,  # [n_frame, n_atom, 3]
    lattice: np.ndarray,  # [3, 3]
    rmax: float = 5.0,
    nbins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:  # dist, rdf
    n_atoms = traj.shape[1]
    vol = np.linalg.det(lattice)
    r_bins = np.linspace(0, rmax, nbins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    dr = r_bins[1] - r_bins[0]
    den = 4 * np.pi * r_centers**2 * dr * (n_atoms / vol) * n_atoms
    nl = NeighborList(cutoff=5.0, full_list=True)
    rdf_list = []
    for positions in traj:
        (r,) = nl.compute(
            quantities="d",
            points=positions,
            box=lattice,
            periodic=True,
        )
        hist, _ = np.histogram(r, bins=r_bins)
        rdf = hist / den
        rdf_list.append(rdf)

    rdf = np.mean(rdf_list, axis=0)
    return r_centers, rdf
