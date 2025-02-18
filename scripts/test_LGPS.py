from pathlib import Path
import argparse

import ase
import ase.io
import numpy as np
import torch

from liflow.model.modules import FlowModule
from liflow.utils.inference import FlowSimulator
from liflow.utils.prior import get_prior

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
# Input and output directories
parser.add_argument("--data-dir", type=str, default="data/LGPS")
parser.add_argument("--output-dir", type=str, default="traj_output_LGPS")

# Checkpoint paths (set None for corrector to disable)
parser.add_argument("--propagate-ckpt", type=str, default="ckpt/P_LGPS.ckpt")
parser.add_argument("--correct-ckpt", type=str, default="ckpt/C_LGPS_0.1.ckpt")

# Simulation parameters: number of steps, flow integration steps, solver
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--flow-steps", type=int, default=10)
parser.add_argument("--solver", type=str, default="euler")

# Note: the original structure is 2x2x1 supercell, so to simulate 4x4x4 supercell,
# the supercell should be [2, 2, 4]
parser.add_argument("--supercell", type=int, nargs=3, default=[1, 1, 1])
parser.add_argument("--temp-list", type=int, nargs="+", default=[650, 900, 1150, 1400])

# Miscellaneous parameters: seed, corrector application frequency, number of retries
# when the simulation step fails
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--corrector-every", type=int, default=1)
parser.add_argument("--num-retries", type=int, default=3)
args = parser.parse_args()

data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Prior scales for LGPS
scale_Li_index, scale_frame_index = 1, 0

# Load initial structure and create supercell
# Note: for LGPS, the initial structure is the same for all temperatures
atoms = ase.io.read(data_dir / "Li10GeS2P12_AIMD_stoichiometric_650K_POSCAR.sid")
atoms = atoms.repeat(args.supercell)

# Load element index
element_idx = np.load(data_dir / "element_index.npy")

# Load propagator model
propagate_model = FlowModule.load_from_checkpoint(args.propagate_ckpt).eval()
cfg_prior = propagate_model.cfg.propagate_prior
propagate_prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=args.seed)

# Load corrector model
if args.correct_ckpt is not None:
    correct_model = FlowModule.load_from_checkpoint(args.correct_ckpt).eval()
    cfg_prior = correct_model.cfg.correct_prior
    correct_prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=args.seed)
else:
    correct_model = None
    correct_prior = None

for temperature in args.temp_list:
    print(f"Running simulation at {temperature} K")
    # Initialize simulator
    atomic_numbers = atoms.get_atomic_numbers()
    simulator = FlowSimulator(
        propagate_model=propagate_model,
        propagate_prior=propagate_prior,
        atomic_numbers=atomic_numbers,
        element_idx=element_idx,
        lattice=atoms.cell.array,
        temp=temperature,
        correct_model=correct_model,
        correct_prior=correct_prior,
        scale_Li_index=scale_Li_index,
        scale_frame_index=scale_frame_index,
    )

    # Run simulation
    traj = simulator.run(
        positions=atoms.get_positions(),
        steps=args.steps,
        flow_steps=args.flow_steps,
        solver=args.solver,
        verbose=True,
        num_retries=args.num_retries,
        corrector_every=args.corrector_every,
    )

    # Save trajectory
    traj_atoms = [
        ase.Atoms(
            numbers=atoms.get_atomic_numbers(),
            positions=traj[i],
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        for i in range(len(traj))
    ]
    ase.io.write(output_dir / f"traj_{temperature}K_{args.seed}.xyz", traj_atoms)
