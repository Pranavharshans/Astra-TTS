#!/bin/bash

# Train Astra-TTS Model A (ZipVoice-Slim) on LibriTTS.

export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=1
stop_stage=3
world_size=${world_size:-1}
use_fp16=${use_fp16:-1}
num_iters=${num_iters:-500000}
save_every_n=${save_every_n:-5000}
max_duration=${max_duration:-100}
base_lr=${base_lr:-0.045}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Data Preparation for LibriTTS dataset"
      bash local/prepare_libritts.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train Astra-TTS Model A (ZipVoice-Slim)"
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size ${world_size} \
            --use-fp16 ${use_fp16} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every_n} \
            --max-duration ${max_duration} \
            --lr-epochs 10 \
            --base-lr ${base_lr} \
            --max-len 20 \
            --valid-by-epoch 0 \
            --model-config conf/astra_model_a_slim.json \
            --tokenizer libritts \
            --token-file data/tokens_libritts.txt \
            --dataset libritts \
            --manifest-dir data/fbank \
            --exp-dir exp/astra_model_a_slim
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Average checkpoints for Astra-TTS Model A"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} \
            --avg 10 \
            --model-name zipvoice \
            --exp-dir exp/astra_model_a_slim
fi
