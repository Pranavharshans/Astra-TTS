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

The Astra training wrappers default to a single 40GB GPU profile:
`world_size=1`, `use_fp16=1`, and `max_duration=100`. You can override these
without editing the scripts, for example:

```bash
max_duration=80 bash run_astra_model_a_libritts.sh
```

Model configs:

- Model A: `ZipVoice/egs/zipvoice/conf/astra_model_a_slim.json`
- Model B: `ZipVoice/egs/zipvoice/conf/astra_model_b_enhanced.json`
