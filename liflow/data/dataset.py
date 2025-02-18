from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from liflow.utils.geometry import get_neighbor_list_batch
from liflow.utils.prior import (
    AdaptiveMaxwellBoltzmannPrior,
    MaxwellBoltzmannPrior,
    NormalPrior,
    Prior,
    UniformScaleNormalPrior,
)


class NoisedFrameDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        df: pd.DataFrame,
        time_delay_steps: int = 100,
        noise: UniformScaleNormalPrior = UniformScaleNormalPrior(),
        prior: Prior = MaxwellBoltzmannPrior(scale=0.1),
        cutoff: float = 5.0,
        pbc: bool = True,
        in_memory: bool = False,
    ):
        super().__init__()
        self.root = Path(data_path)
        self.df = df
        self.skip_steps = time_delay_steps
        self.noise = noise
        self.prior = prior
        self.cutoff = cutoff
        self.pbc = pbc
        self.in_memory = in_memory

        self.load_data()

    def load_data(self):
        self.element_idx = np.load(self.root / "element_index.npy")
        self.atomic_numbers = np.load(
            self.root / "atomic_numbers.npy", allow_pickle=True
        ).item()
        self.positions = dict()
        # Load positions for each temperature
        temp_set = set(self.df["temp"])
        for temp in temp_set:
            self.positions[temp] = np.load(self.root / f"positions_{temp}K.npz")
            if self.in_memory:  # load all positions into memory
                self.positions[temp] = dict(self.positions[temp])
        self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        # Randomly select a timeframe
        time = np.random.randint(row["t_start"] + self.skip_steps, row["t_end"])

        # Get atomic numbers, positions, and lattice
        atomic_numbers = self.atomic_numbers[row["name"]]
        elements = self.element_idx[atomic_numbers]
        masses = ase.data.atomic_masses[atomic_numbers]
        positions = self.positions[row["temp"]][row["name"]][time]
        lattice = self.lattice[row["name"]]

        # Add noise to positions
        positions_noisy = positions + self.noise.sample(shape=positions.shape)

        # Get edge information
        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=[positions_noisy],
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.pbc,
        )
        ref_dist = ase.data.covalent_radii[atomic_numbers[edge_index]].sum(axis=0)

        # Add prior displacement
        if isinstance(self.prior, NormalPrior):
            prior = self.prior.sample(shape=positions.shape)
        elif isinstance(self.prior, MaxwellBoltzmannPrior):
            prior = self.prior.sample(
                temperature=row["temp"], masses=masses, shape=positions.shape
            )
        else:
            raise ValueError("Unsupported prior")

        return Data(
            num_nodes=len(elements),
            positions_1=torch.tensor(positions_noisy, dtype=torch.float),
            prior=torch.tensor(prior, dtype=torch.float),
            positions_2=torch.tensor(positions, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            shifts=torch.tensor(shifts, dtype=torch.float),
            ref_dist=torch.tensor(ref_dist, dtype=torch.float),
            elements=torch.tensor(elements, dtype=torch.long),
            temp=torch.full(elements.shape, row["temp"]).float(),
            lattice=torch.tensor(lattice, dtype=torch.float),
        )


class TimeDelayedPairDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        df: pd.DataFrame,
        time_delay_steps: int = 100,
        prior: Prior = AdaptiveMaxwellBoltzmannPrior(),
        cutoff: float = 5.0,
        pbc: bool = True,
        neighbor_list_both_ends: bool = False,
        in_memory: bool = False,
    ):
        super().__init__()
        self.root = Path(data_path)
        self.df = df
        self.time_delay_steps = time_delay_steps
        self.prior = prior
        self.cutoff = cutoff
        self.pbc = pbc
        self.neighbor_list_both_ends = neighbor_list_both_ends
        self.in_memory = in_memory

        self.load_data()

    def load_data(self):
        self.element_idx = np.load(self.root / "element_index.npy")
        self.atomic_numbers = np.load(
            self.root / "atomic_numbers.npy", allow_pickle=True
        ).item()
        self.positions = dict()
        # Load positions for each temperature
        temp_set = set(self.df["temp"])
        for temp in temp_set:
            self.positions[temp] = np.load(self.root / f"positions_{temp}K.npz")
            if self.in_memory:  # load all positions into memory
                self.positions[temp] = dict(self.positions[temp])
        self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        # Randomly select a temperature and timeframe
        start_time = np.random.randint(
            row["t_start"], row["t_end"] - self.time_delay_steps
        )
        end_time = start_time + self.time_delay_steps

        # Get atomic numbers, positions, and lattice
        atomic_numbers = self.atomic_numbers[row["name"]]
        elements = self.element_idx[atomic_numbers]
        masses = ase.data.atomic_masses[atomic_numbers]
        positions_1 = self.positions[row["temp"]][row["name"]][start_time]
        positions_2 = self.positions[row["temp"]][row["name"]][end_time]
        lattice = self.lattice[row["name"]]

        # Get edge information
        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=(
                [positions_1, positions_2]
                if self.neighbor_list_both_ends
                else [positions_1]
            ),
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.pbc,
        )
        ref_dist = ase.data.covalent_radii[atomic_numbers[edge_index]].sum(axis=0)

        # Add prior displacement
        if isinstance(self.prior, NormalPrior):
            prior = self.prior.sample(shape=positions_1.shape)
        elif isinstance(self.prior, MaxwellBoltzmannPrior):
            prior = self.prior.sample(
                temperature=row["temp"], masses=masses, shape=positions_1.shape
            )
        elif isinstance(self.prior, AdaptiveMaxwellBoltzmannPrior):
            prior = self.prior.sample(
                temperature=row["temp"],
                atomic_numbers=atomic_numbers,
                masses=masses,
                scale_Li_index=int(row["prior_Li"]),
                scale_frame_index=int(row["prior_frame"]),
                shape=positions_1.shape,
            )
        else:
            raise ValueError("Unsupported prior")

        return Data(
            num_nodes=len(elements),
            positions_1=torch.tensor(positions_1, dtype=torch.float),
            prior=torch.tensor(prior, dtype=torch.float),
            positions_2=torch.tensor(positions_2, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            shifts=torch.tensor(shifts, dtype=torch.float),
            ref_dist=torch.tensor(ref_dist, dtype=torch.float),
            elements=torch.tensor(elements, dtype=torch.long),
            temp=torch.full(elements.shape, row["temp"]).float(),
            lattice=torch.tensor(lattice, dtype=torch.float),
        )
