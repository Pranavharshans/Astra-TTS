#!/bin/bash

# Train Astra-TTS Model A (ZipVoice-Slim) on LibriTTS.

export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Data Preparation for LibriTTS dataset"
      bash local/prepare_libritts.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train Astra-TTS Model A (ZipVoice-Slim)"
      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 8 \
            --use-fp16 0 \
            --num-iters 500000 \
            --save-every-n 5000 \
            --max-duration 250 \
            --lr-epochs 10 \
            --base-lr 0.045 \
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
            --iter 500000 \
            --avg 10 \
            --model-name zipvoice \
            --exp-dir exp/astra_model_a_slim
fi
