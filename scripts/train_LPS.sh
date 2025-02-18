conda activate liflow
export OMP_NUM_THREADS=1

python -m liflow.experiment.train \
    task=propagate \
    name=P_LPS \
    data.data_path=data/LPS \
    data.index_files="[train_25ps.csv]" \
    data.train_valid_split=False \
    data.sample_weight_comp=False \
    data.time_delay_steps=100 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[1.0,10.0],[1.0,1.0]]"

python -m liflow.experiment.train \
    task=correct \
    name=C_LPS \
    data.data_path=data/LPS \
    data.index_files="[train_25ps.csv]" \
    data.train_valid_split=False \
    data.sample_weight_comp=False \
    data.time_delay_steps=100 \
    data.in_memory=True \
    correct_noise.params.scale=0.1
