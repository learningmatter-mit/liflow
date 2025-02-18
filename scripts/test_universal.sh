conda activate liflow
export OMP_NUM_THREADS=1

for temp in 600 800 1000 1200; do
    for seed in 1 2 3; do
        # Propagator only
        python -m liflow.experiment.test \
            ckpt.propagate=ckpt/P_universal.ckpt \
            data.data_path=data/universal \
            data.index_file=test_${temp}K.csv \
            inference.temp=${temp} \
            inference.solver=euler \
            inference.flow_steps=10 \
            seed=${seed} \
            result.output_csv=eval_outputs/P_universal_${temp}_${seed}.csv

        # Propagator + Corrector
        python -m liflow.experiment.test \
            ckpt.propagate=ckpt/P_universal.ckpt \
            ckpt.correct=ckpt/C_universal.ckpt \
            data.data_path=data/universal \
            data.index_file=test_${temp}K.csv \
            inference.temp=${temp} \
            inference.solver=euler \
            inference.flow_steps=10 \
            seed=${seed} \
            result.output_csv=eval_outputs/PC_universal_${temp}_${seed}.csv
    done
done
