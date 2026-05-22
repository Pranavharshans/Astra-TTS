# Custom Fine-Tune

This folder contains a small clean-dataset fine-tuning workflow for Astra-TTS.

Default experiment:

- Base model: `Praha-Labs/astra-tts-model-b-enhanced-200k`
- Checkpoint: `checkpoint-200000.pt`
- Dataset: LJSpeech 1.1
- Fine-tune length: 30,000 steps
- Output directory: `/workspace/astra_model_b_enhanced_ljspeech_ft_30k`
- Safer defaults: `base_lr=0.0001`, `use_fp16=0`

Run from the repo root on a GPU VM:

```bash
cd /workspace/Astra-TTS
bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"
```

Useful overrides:

```bash
num_iters=10000 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"

base_lr=0.0002 max_duration=100 use_fp16=0 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"

hf_repo=Praha-Labs/astra-tts-model-b-enhanced-250k \
checkpoint_name=checkpoint-250000.pt \
bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"
```

Stages:

```bash
stage=1 stop_stage=4 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"
stage=5 stop_stage=6 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"
```

For TensorBoard:

```bash
tensorboard --logdir /workspace --host 0.0.0.0 --port 6006
```
