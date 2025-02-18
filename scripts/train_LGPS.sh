conda activate liflow
export OMP_NUM_THREADS=1

python -m liflow.experiment.train \
    task=propagate \
    name=P_LGPS \
    data.data_path=data/LGPS \
    data.index_files="[train_25ps.csv]" \
    data.train_valid_split=False \
    data.sample_weight_comp=False \
    data.time_delay_steps=67 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"

python -m liflow.experiment.train \
    task=correct \
    name=C_LGPS_0.1 \
    data.data_path=data/LGPS \
    data.index_files="[train_25ps.csv]" \
    data.train_valid_split=False \
    data.sample_weight_comp=False \
    data.time_delay_steps=67 \
    data.in_memory=True \
    correct_noise.params.scale=0.1

python -m liflow.experiment.train \
    task=correct \
    name=C_LGPS_0.2 \
    data.data_path=data/LGPS \
    data.index_files="[train_25ps.csv]" \
    data.train_valid_split=False \
    data.sample_weight_comp=False \
    data.time_delay_steps=67 \
    data.in_memory=True \
    correct_noise.params.scale=0.2
