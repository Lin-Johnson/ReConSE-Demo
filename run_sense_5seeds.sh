#!/usr/bin/env bash

set -o pipefail

cd /yangliusha02/Project/SenSE-main

for seed in 0 1 2 3 4; do
    echo "===== START seed ${seed} ====="
    /opt/conda/envs/sense/bin/accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        src/sense/eval/eval_infer_batch.py \
        --seed "${seed}" \
        --llm_model SenSE_LLM_Base \
        --llm_ckpt_file /yangliusha02/Model/ASLP-lab/SenSE/SenSE_LLM.safetensors \
        --fm_model SenSE_CFM_Base \
        --fm_ckpt_file /yangliusha02/Model/ASLP-lab/SenSE/SenSE_CFM.safetensors \
        --exp_name voicefixer_gsr \
        --save_sample_rate 24000 \
        --testset custom \
        --test_dir /yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/simulated \
        --no_ref_audio \
        --nfestep 8 \
        --cfg_strength 0.5 \
        --swaysampling -1
    rc=$?
    echo "===== END seed ${seed}: rc=${rc} ====="
    if [ "${rc}" -ne 0 ]; then
        exit "${rc}"
    fi
done

echo "===== ALL FIVE SEEDS COMPLETED ====="
