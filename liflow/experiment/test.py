from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from liflow.model.modules import FlowModule
from liflow.utils.analysis import calculate_average_rdf, calculate_msd
from liflow.utils.inference import FlowSimulator
from liflow.utils.prior import get_prior

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../config", config_name="test")
def main(cfg):
    # Load propagator model
    propagate_model = FlowModule.load_from_checkpoint(cfg.ckpt.propagate).eval()
    cfg_prior = propagate_model.cfg.propagate_prior
    propagate_prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=cfg.seed)

    # Load corrector model
    if cfg.ckpt.correct is not None:
        correct_model = FlowModule.load_from_checkpoint(cfg.ckpt.correct).eval()
        cfg_prior = correct_model.cfg.correct_prior
        correct_prior = get_prior(
            cfg_prior.class_name, **cfg_prior.params, seed=cfg.seed
        )
    else:
        correct_model = None
        correct_prior = None

    # Load data
    data_path = Path(cfg.data.data_path)
    df_test = pd.read_csv(data_path / cfg.data.index_file)
    element_idx = np.load(data_path / "element_index.npy")
    atomic_numbers_data = np.load(
        data_path / "atomic_numbers.npy", allow_pickle=True
    ).item()
    lattice_data = np.load(data_path / "lattice.npy", allow_pickle=True).item()
    positions_data = np.load(data_path / f"positions_{cfg.inference.temp}K.npz")

    msd_Li_list, msd_Li_ref_list = [], []
    msd_frame_list, msd_frame_ref_list = [], []
    rdf_mae_list = []
    final_step_list = []

    # Simulate for each item in the test set
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        name = row["name"]
        atomic_numbers = atomic_numbers_data[name]
        lattice = lattice_data[name]
        positions_ref = positions_data[name]

        if (
            propagate_model.cfg.propagate_prior.class_name
            == "AdaptiveMaxwellBoltzmannPrior"
        ):
            scale_Li_index = int(row["prior_Li"])
            scale_frame_index = int(row["prior_frame"])
        else:
            scale_Li_index = None
            scale_frame_index = None

        # Initialize simulator
        simulator = FlowSimulator(
            propagate_model=propagate_model,
            propagate_prior=propagate_prior,
            atomic_numbers=atomic_numbers,
            element_idx=element_idx,
            lattice=lattice,
            temp=cfg.inference.temp,
            correct_model=correct_model,
            correct_prior=correct_prior,
            scale_Li_index=scale_Li_index,
            scale_frame_index=scale_frame_index,
        )

        # Run simulation
        positions = positions_ref[0].copy()
        traj = simulator.run(
            positions,
            steps=cfg.inference.steps,
            flow_steps=cfg.inference.flow_steps,
            solver=cfg.inference.solver,
            num_retries=cfg.inference.num_retries,
        )

        # 1. MSD evaluation
        msd_Li = calculate_msd(traj, atomic_numbers == 3)
        msd_Li_ref = calculate_msd(positions_ref, atomic_numbers == 3)
        msd_frame = calculate_msd(traj, atomic_numbers != 3)
        msd_frame_ref = calculate_msd(positions_ref, atomic_numbers != 3)
        final_step = len(traj) - 1

        msd_Li_list.append(msd_Li)
        msd_Li_ref_list.append(msd_Li_ref)
        msd_frame_list.append(msd_frame)
        msd_frame_ref_list.append(msd_frame_ref)
        final_step_list.append(final_step)

        # 2. RDF evaluation
        if len(traj) <= 5:
            traj = np.array(traj[-1])[None]  # Use the last frame
        else:
            traj = np.array(traj[5:])
        traj_ref = positions_ref[500::100]

        _, rdf = calculate_average_rdf(traj, lattice, rmax=5.0, nbins=50)
        _, rdf_ref = calculate_average_rdf(traj_ref, lattice, rmax=5.0, nbins=50)
        rdf_mae = np.mean(np.abs(rdf - rdf_ref))
        rdf_mae_list.append(rdf_mae)

    df_test["msd_Li"] = msd_Li_list
    df_test["msd_Li_ref"] = msd_Li_ref_list
    df_test["msd_frame"] = msd_frame_list
    df_test["msd_frame_ref"] = msd_frame_ref_list
    df_test["rdf_mae"] = rdf_mae_list
    df_test["final_step"] = final_step_list

    df_test.to_csv(cfg.result.output_csv, index=False)


if __name__ == "__main__":
    main()
