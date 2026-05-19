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
