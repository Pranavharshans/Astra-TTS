# Custom Fine-Tune

This folder contains a small clean-dataset fine-tuning workflow for Astra-TTS.

Default experiment:

- Base model: `Praha-Labs/astra-tts-model-b-enhanced-200k`
- Checkpoint: `checkpoint-200000.pt`
- Dataset: LJSpeech 1.1
- Fine-tune length: 30,000 steps
- Output directory: `/workspace/astra_model_b_enhanced_ljspeech_ft_adamw_30k`
- Safer defaults: `optimizer=adamw`, `base_lr=1e-5`, `max_duration=40`,
  `use_fp16=0`, `condition_drop_ratio=0.0`,
  `disable_aux_grad_penalties=1`

The base HF artifact is a training checkpoint. The script exports a weights-only
checkpoint first, preferring `model_avg` when present, and fine-tunes from that file.
This avoids starting the fine-tune from a raw instantaneous training state and avoids
loading optimizer/scheduler/scaler state.

The script also disables Balancer/Whiten auxiliary gradient penalties during
fine-tuning. Those modules do not change the forward model output, but they can add
unstable backward-only gradients when adapting the smaller Model B checkpoint to a
small single-speaker dataset.

Run from the repo root on a GPU VM:

```bash
cd /workspace/Astra-TTS
bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"
```

Useful overrides:

```bash
num_iters=10000 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"

base_lr=5e-6 max_duration=30 use_fp16=0 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"

optimizer=scaledadam base_lr=1e-5 max_duration=30 use_fp16=0 bash "custom finetune/run_ljspeech_finetune_model_b_200k.sh"

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
