# Astra-TTS

Astra-TTS is a research implementation derived from
[k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice), an Apache-2.0 licensed
flow-matching TTS system.

This repository vendors ZipVoice as the base implementation and will add two
Astra model variants:

- Model A: ZipVoice-Slim, a conservative 55M parameter shrink.
- Model B: ZipVoice-Enhanced, a 55M parameter enhanced architecture.

The model specifications live in:

- `Arch+prd/model_a_slim.md`
- `Arch+prd/model_b_enhanced.md`

Training entry points:

- Model A: `ZipVoice/egs/zipvoice/run_astra_model_a_libritts.sh`
- Model B: `ZipVoice/egs/zipvoice/run_astra_model_b_libritts.sh`

The Astra training wrappers default to a single 24GB GPU profile:
`world_size=1`, `use_fp16=1`, `max_duration=40`, and
`accum_grad_batches=3`. This keeps GPU memory lower while preserving a larger
effective batch through gradient accumulation. You can override these without
editing the scripts, for example:

```bash
max_duration=30 accum_grad_batches=4 bash run_astra_model_a_libritts.sh
```

Experiment outputs default to `/workspace`:

- Model A: `/workspace/astra_model_a_slim`
- Model B: `/workspace/astra_model_b_enhanced`

TensorBoard logs are written under each experiment directory:

```bash
tensorboard --logdir /workspace --host 0.0.0.0 --port 6006
```

Model configs:

- Model A: `ZipVoice/egs/zipvoice/conf/astra_model_a_slim.json`
- Model B: `ZipVoice/egs/zipvoice/conf/astra_model_b_enhanced.json`

Model B optimized inference is available through `infer_zipvoice.py`:

```bash
python3 -m zipvoice.bin.infer_zipvoice \
  --model-name zipvoice \
  --model-dir exp/astra_model_b_enhanced \
  --checkpoint-name iter-500000-avg-10.pt \
  --tokenizer libritts \
  --test-list test.tsv \
  --res-dir results/astra_model_b_opt \
  --num-step 4 \
  --solver midpoint \
  --step-schedule epss \
  --smooth-cache True \
  --smooth-cache-stacks 0,1 \
  --smooth-cache-interval 2
```
