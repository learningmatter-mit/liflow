conda activate liflow
export OMP_NUM_THREADS=1

python -m liflow.experiment.train \
    task=propagate \
    name=P_universal \
    data.data_path=data/universal \
    data.index_files="[train_600K.csv,train_800K.csv,train_1000K.csv,train_1200K.csv]" \
    data.sample_weight_comp=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[1.0,10.0],[0.316,3.16]]"

python -m liflow.experiment.train \
    task=correct \
    name=C_universal \
    data.data_path=data/universal \
    data.index_files="[train_600K.csv,train_800K.csv,train_1000K.csv,train_1200K.csv]" \
    data.sample_weight_comp=True
