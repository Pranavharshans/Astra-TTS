#!/bin/bash

# Train Astra-TTS Model B Enhanced 88M on LibriTTS.
# 88M variant: dim=512, ff=1792, text_encoder_dim=256, GQA, depthwise FFN,
# RoPE, ConvNeXt, 3-layer text refinement, center-heavy 9-layer decoder.

export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=${stage:-1}
stop_stage=${stop_stage:-3}
world_size=${world_size:-1}
use_fp16=${use_fp16:-1}
num_iters=${num_iters:-400000}
save_every_n=${save_every_n:-5000}
# 3090 24GB with fp16: 350s audio fits ~20GB, leaving headroom for backward
max_duration=${max_duration:-350}
# Gradient accumulation keeps effective batch size high without OOM
accum_grad_batches=${accum_grad_batches:-2}
base_lr=${base_lr:-0.02}
exp_dir=${exp_dir:-/workspace/astra_model_b_enhanced_88m}
start_epoch=${start_epoch:-1}
checkpoint=${checkpoint:-}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Data Preparation for LibriTTS dataset"
      bash local/prepare_libritts.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train Astra-TTS Model B Enhanced 88M"
      extra_args=()
      if [ "${start_epoch}" -gt 1 ]; then
          extra_args+=(--start-epoch "${start_epoch}")
      fi
      if [ -n "${checkpoint}" ]; then
          extra_args+=(--checkpoint "${checkpoint}")
      fi
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size ${world_size} \
            --use-fp16 ${use_fp16} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every_n} \
            --max-duration ${max_duration} \
            --accum-grad-batches ${accum_grad_batches} \
            --lr-epochs 10 \
            --base-lr ${base_lr} \
            --max-len 20 \
            --valid-by-epoch 0 \
            --model-config conf/astra_model_b_enhanced_80m.json \
            --tokenizer libritts \
            --token-file data/tokens_libritts.txt \
            --dataset libritts \
            --manifest-dir data/fbank \
            --exp-dir ${exp_dir} \
            "${extra_args[@]}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Average checkpoints for Astra-TTS Model B Enhanced 88M"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} \
            --avg 10 \
            --model-name zipvoice \
            --exp-dir ${exp_dir}
fi
