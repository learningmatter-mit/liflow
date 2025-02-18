import ase
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from liflow.model.modules import FlowModule
from liflow.utils.geometry import get_neighbor_list_batch
from liflow.utils.prior import (
    AdaptiveMaxwellBoltzmannPrior,
    MaxwellBoltzmannPrior,
    NormalPrior,
    Prior,
)


class FlowSimulator:
    def __init__(
        self,
        propagate_model: FlowModule,
        propagate_prior: Prior,
        atomic_numbers: np.ndarray,
        element_idx: np.ndarray,
        lattice: np.ndarray,
        temp: float,
        correct_model: FlowModule | None = None,
        correct_prior: Prior | None = None,
        pbc: bool = True,
        scale_Li_index: int | None = None,  # AdaptiveMaxwellBoltzmannPrior
        scale_frame_index: int | None = None,  # AdaptiveMaxwellBoltzmannPrior
    ):
        self.propagate_model = propagate_model
        self.propagate_prior = propagate_prior
        self.atomic_numbers = atomic_numbers
        # elements: atomic numbers -> element indices in the model
        self.elements = element_idx[atomic_numbers]
        self.masses = ase.data.atomic_masses[atomic_numbers]
        self.lattice = lattice
        self.temp = temp
        self.correct_model = correct_model
        self.correct_prior = correct_prior
        self.pbc = pbc
        self.device = propagate_model.device
        self.scale_Li_index = scale_Li_index
        self.scale_frame_index = scale_frame_index

    def initialize_batch(self, positions: np.ndarray, prior: Prior) -> Data:
        # Get edge information
        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=[positions],
            lattice=self.lattice,
            cutoff=self.propagate_model.cfg.data.cutoff,
            periodic=self.pbc,
        )

        # Sample prior
        if isinstance(prior, NormalPrior):
            _prior = prior.sample(shape=positions.shape)
        elif isinstance(prior, MaxwellBoltzmannPrior):
            _prior = prior.sample(
                temperature=self.temp, masses=self.masses, shape=positions.shape
            )
        elif isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
            _prior = prior.sample(
                temperature=self.temp,
                atomic_numbers=self.atomic_numbers,
                masses=self.masses,
                scale_Li_index=self.scale_Li_index,
                scale_frame_index=self.scale_frame_index,
                shape=positions.shape,
            )
        else:
            raise NotImplementedError

        # Return data object
        data = Data(
            num_nodes=len(self.elements),
            positions_1=torch.tensor(positions, dtype=torch.float),
            prior=torch.tensor(_prior, dtype=torch.float),
            positions_2=torch.tensor(positions, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            shifts=torch.tensor(shifts, dtype=torch.float),
            elements=torch.tensor(self.elements, dtype=torch.long),
            temp=torch.full(self.elements.shape, self.temp).float(),
        )
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data.to(self.device, non_blocking=True)

    @staticmethod
    @torch.no_grad()
    def propagate_flow(
        data: Data, model: FlowModule, flow_steps: int = 10, solver: str = "euler"
    ) -> torch.Tensor:
        # Time grid
        t_list = torch.linspace(
            0.0, 1.0, flow_steps + 1, device=model.device, dtype=torch.float
        )
        step_dt = t_list[1] - t_list[0]
        if hasattr(model.cfg.model, "prediction_mode"):
            prediction_mode = model.cfg.model.prediction_mode
        else:
            prediction_mode = "velocity"

        # Propagate flow
        for i, t in enumerate(t_list[:-1]):
            if prediction_mode == "velocity":
                pred_dx_dt = model(data, t_per_graph=t.unsqueeze(0))
            elif prediction_mode == "data":
                pred_disp = model(data, t_per_graph=t.unsqueeze(0))
                pred_dx_dt = (data["positions_1"] + pred_disp - data["positions_2"]) / (
                    1 - t
                )
            else:
                raise NotImplementedError
            if solver == "euler":
                data["positions_2"] += pred_dx_dt * step_dt  # Euler step
            elif solver == "heun":
                next_t = (t + step_dt).unsqueeze(0)
                next_data = data.clone()
                next_data["positions_2"] += pred_dx_dt * step_dt
                if prediction_mode == "velocity":
                    next_dx_dt = model(next_data, t_per_graph=next_t)
                elif prediction_mode == "data":
                    next_disp = model(next_data, t_per_graph=next_t)
                    if i == flow_steps - 1:
                        return next_data["positions_1"] + next_disp
                    next_dx_dt = (
                        next_data["positions_1"] + next_disp - next_data["positions_2"]
                    ) / (1 - next_t)
                data["positions_2"] += 0.5 * (pred_dx_dt + next_dx_dt) * step_dt
            else:
                raise NotImplementedError

        return data["positions_2"]

    @torch.no_grad()
    def step(
        self,
        positions: np.ndarray,
        flow_steps: int = 10,
        solver: str = "euler",
        use_corrector: bool = True,
    ) -> np.ndarray:
        # Propagator
        data = self.initialize_batch(positions, self.propagate_prior)
        if (
            hasattr(self.propagate_model, "use_scale")
            and self.propagate_model.use_scale
        ):
            scale = self.propagate_model.scale_model(data)
            data["prior"] = data["prior"] * scale[:, None]
        data["positions_2"] = data["positions_2"] + data["prior"]
        new_positions = self.propagate_flow(
            data, self.propagate_model, flow_steps, solver
        )
        new_positions = new_positions.detach().cpu().numpy()

        # Corrector
        if use_corrector:
            data = self.initialize_batch(new_positions, self.correct_prior)
            new_positions = self.propagate_flow(
                data, self.correct_model, flow_steps, solver
            )
            new_positions = new_positions.detach().cpu().numpy()

        return new_positions

    @torch.no_grad()
    def run(
        self,
        positions: np.ndarray,  # [n_atoms, 3]
        steps: int = 25,
        flow_steps: int = 10,
        solver: str = "euler",
        verbose: bool = False,
        fix_com: bool = True,
        num_retries: int = 2,
        corrector_every: int = 1,
    ) -> np.ndarray:
        traj = [positions]
        it = tqdm(range(steps)) if verbose else range(steps)
        for i_step in it:
            # Save previous center of mass
            positions_prev = positions.copy()
            # Propagate (and optionally correct)
            success = False
            for _ in range(num_retries + 1):
                use_corrector = ((i_step + 1) % corrector_every == 0) and (
                    self.correct_model is not None
                )
                positions = self.step(positions_prev, flow_steps, solver, use_corrector)
                # Fix center of mass
                if fix_com:
                    positions_diff = positions - positions_prev
                    com_diff = np.sum(
                        positions_diff * self.masses[:, None], axis=0
                    ) / np.sum(self.masses)
                    positions -= com_diff
                # Check for divergence
                if not (
                    np.isnan(positions).any()
                    or np.abs(positions - positions_prev).max() > 1e3
                ):
                    success = True
                    break
            if not success:
                break
            # Save trajectory
            traj.append(positions)
        return traj
