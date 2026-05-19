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

Model configs:

- Model A: `ZipVoice/egs/zipvoice/conf/astra_model_a_slim.json`
- Model B: `ZipVoice/egs/zipvoice/conf/astra_model_b_enhanced.json`
