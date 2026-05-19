# Astra-TTS Model B: ZipVoice-Enhanced

## Overview

| Field | Value |
|-------|-------|
| **Model Name** | Astra-TTS Model B (ZipVoice-Enhanced) |
| **Target Parameters** | ~55M |
| **Base Architecture** | ZipVoice (Zipformer-based flow-matching TTS) |
| **Approach** | Same param budget as Model A, but with architectural improvements for better speed AND quality |
| **Purpose** | Prove that smart architecture changes outperform naive shrinking |
| **Training Data** | LibriTTS |
| **License** | Apache-2.0 |

---

## Design Philosophy

Model B spends the same ~55M parameter budget as Model A, but **redistributes and optimizes** it using proven techniques from recent papers. The goal is:

1. **Better quality** at the same size (through better inductive biases)
2. **Faster inference** (through architecture + inference-time optimizations)
3. **Future-ready** (style tokens for emotion, RoPE for variable length)

Every change is backed by published ablations or deployed systems (Supertonic 3, F5-TTS, FLY-TTS, M3-TTS).

---

## Architecture Changes Summary

| Change | What it does | Param cost | Expected gain | Source |
|--------|-------------|-----------|---------------|--------|
| **Grouped Query Attention (GQA)** | 4 Query heads share 2 KV heads | Saves ~2M | 1.3× attention speedup, same quality | Llama 2/3, GQA paper |
| **Depthwise Separable Conv FFN** | Replace standard FFN (dim→4×→dim) with depthwise separable conv | Saves ~5M per equivalent capacity | ~2× faster FFN, same quality for sequential data | FLY-TTS, Supertonic 3 |
| **Grouped Parameter Sharing** | Adjacent layers within a stack share base weights + low-rank residual (rank=32) | Saves ~8M (reused as bigger effective dim) | Same quality with fewer unique params | ResidualTransformer (3× reduction, no loss in speech) |
| **BiasNorm** | Keep Zipformer's BiasNorm (simpler than LayerNorm, no mean computation) | 0 | Faster than LayerNorm, already proven | Zipformer paper |
| **Dilated ConvNeXt in FM decoder** | Add dilated conv blocks [1,2,4,8] before self-attention in each stack | +3M | Capture local patterns cheaply (O(n) vs O(n²)), reduce attention workload | Supertonic 3, F5-TTS |
| **Remove NLA** | Drop Non-Linear Attention module | Saves ~5M | Negligible loss (ablation: UTMOS 4.16→3.99) | ZipVoice ablation Table V |
| **Add RoPE** | Rotary Position Embeddings replace Zipformer native pos enc | ~0 | Better variable-length generalization | F5-TTS, DiTTo-TTS, Supertonic 3 |
| **ConvNeXt text refinement** | 2-layer ConvNeXt after text encoder, before upsampling | +2M | Better alignment → lower WER | F5-TTS, Supertonic 3 |
| **Fewer text encoder layers** (4→3) | Compensate params for ConvNeXt refinement | -1.5M | Minimal loss (ConvNeXt compensates) | - |
| **Middle-heavy layer distribution** [1,2,3,2,1] | More capacity at lowest resolution (4× downsampled) | ~same | Better global pattern modeling (U-Net principle) | Diffusion model best practices |

---

## Architecture Specification

### FM Decoder (Vector Field Estimator)

| Parameter | Original (123M) | Model B (55M) | Change |
|-----------|-----------------|---------------|--------|
| `fm_decoder_dim` | 512 | **384** | -25% |
| `fm_decoder_feedforward_type` | Standard Linear | **Depthwise Separable Conv** | New |
| `fm_decoder_feedforward_dim` | 1536 | **1024** | -33% (but cheaper per FLOP) |
| `fm_decoder_num_heads` (Query) | 4 | **4** | unchanged |
| `fm_decoder_num_kv_heads` | 4 | **2 (GQA)** | New — halves KV computation |
| `fm_decoder_num_layers` | [2, 2, 4, 4, 4] = 16 | **[1, 2, 3, 2, 1] = 9** | Fewer but deeper at center |
| `fm_decoder_downsampling_factor` | [1, 2, 4, 2, 1] | [1, 2, 4, 2, 1] | unchanged |
| `fm_decoder_cnn_module_kernel` | [31, 15, 7, 15, 31] | [31, 15, 7, 15, 31] | unchanged |
| NLA module | Yes | **No (removed)** | Saves params |
| Attention weight sharing | Yes | **Yes + Grouped Parameter Sharing** | Enhanced |
| Bypass connections | Yes | Yes | unchanged |
| Position encoding | Zipformer native | **RoPE (rotary_base=10000)** | New |
| Normalization | BiasNorm | BiasNorm | unchanged (kept) |
| Pre-attention local modeling | CNN module only | **CNN module + Dilated ConvNeXt** | New |

#### Dilated ConvNeXt Block (per stack)

Inserted before self-attention in each Zipformer layer:

| Parameter | Value |
|-----------|-------|
| `kernel_size` | 7 |
| `intermediate_dim` | 768 (2× decoder dim) |
| `num_layers_per_stack` | [1, 1, 2, 1, 1] |
| `dilation_pattern` | [1, 2, 4, 8] (cycled) |
| `activation` | GELU |
| Residual connection | Yes |

#### Grouped Parameter Sharing

Within each stack, layers share a base weight matrix with a unique low-rank residual:

```
Layer_i_weight = Shared_Base_Weight + A_i @ B_i  (rank=32)
```

| Parameter | Value |
|-----------|-------|
| `sharing_mode` | within_stack |
| `residual_rank` | 32 |
| `shared_components` | FFN weights, attention projection weights |
| `unique_components` | Low-rank residuals, BiasNorm params, conv kernels |

#### Depthwise Separable Conv FFN

Replaces standard `Linear(dim, 4*dim) → GELU → Linear(4*dim, dim)`:

```
Depthwise Conv1d(dim, dim, kernel=7, groups=dim) → GELU → Pointwise Conv1d(dim, ff_dim) → GELU → Pointwise Conv1d(ff_dim, dim)
```

| Parameter | Value |
|-----------|-------|
| `depthwise_kernel_size` | 7 |
| `expansion_factor` | ~2.67 (1024/384) |
| FLOPs vs standard FFN | ~3× fewer |
| Quality impact | None for sequential audio data |

### Text Encoder

| Parameter | Original (123M) | Model B (55M) | Change |
|-----------|-----------------|---------------|--------|
| `text_encoder_dim` | 192 | 192 | unchanged |
| `text_encoder_feedforward_dim` | 512 | **384** | -25% |
| `text_encoder_num_layers` | 4 | **3** | -1 layer |
| `text_encoder_num_heads` | 4 | 4 | unchanged |
| `text_encoder_cnn_module_kernel` | 9 | 9 | unchanged |

#### ConvNeXt Text Refinement (New)

Applied after text encoder output, before average upsampling:

| Parameter | Value |
|-----------|-------|
| `num_layers` | 2 |
| `dim` | 192 |
| `intermediate_dim` | 512 |
| `kernel_size` | 7 |
| `dilation` | [1, 2] |
| `activation` | GELU |
| Residual connection | Yes |

### Shared Parameters

| Parameter | Value |
|-----------|-------|
| `query_head_dim` | 32 |
| `value_head_dim` | 12 |
| `pos_head_dim` | 4 (used only for RoPE dim calculation) |
| `pos_dim` | 48 |
| `time_embed_dim` | 192 |
| `text_embed_dim` | 192 |
| `feat_dim` | 100 |

### Feature Extraction

| Parameter | Value |
|-----------|-------|
| `sampling_rate` | 24000 |
| `type` | vocos |

---

## Parameter Budget Breakdown (Estimated)

| Component | Model A (naive) | Model B (enhanced) |
|-----------|----------------|-------------------|
| Text Encoder (Zipformer) | 5M | 3.5M (3 layers, smaller ff) |
| ConvNeXt Text Refinement | — | 2M |
| FM Decoder (Zipformer) | 47M | 35M (GQA + DepthSep + shared + no NLA) |
| Dilated ConvNeXt (decoder) | — | 5M |
| Grouped Sharing Residuals | — | 2M |
| Embeddings + RoPE | 3M | 3M |
| Style Encoder (32 GST) | — | 3.5M |
| **Total** | **~55M** | **~54M** |

> Note: Grouped Parameter Sharing means the 9 decoder layers only store ~5 unique full layers + 9 low-rank residuals, freeing params that are redistributed to ConvNeXt and style encoder.

---

## Style Encoder (Future-Ready)

Included in architecture but **optional during LibriTTS benchmark** (can be frozen/zeroed for fair comparison with Model A):

| Parameter | Value |
|-----------|-------|
| `n_style_tokens` | 32 |
| `style_dim` | 192 |
| `n_heads` | 2 |
| `convnext_layers` | 3 (dim=192, intermediate=512) |
| Input | Reference audio mel features |
| Output | Style embedding (added to time conditioning) |
| Expression tags supported | `<laugh>`, `<whisper>`, `<sad>`, `<breath>`, `<sigh>` |

For the LibriTTS benchmark, style encoder receives the same reference audio as ZipVoice's speaker prompt — no expression labels needed.

---

## Inference Configuration

### Standard Mode (For fair comparison with Model A)

| Parameter | Value |
|-----------|-------|
| ODE Solver | Euler (1st order) |
| NFE | 16 |
| Step Schedule | Uniform |
| Layer Caching | None |

### Optimized Mode (Demonstrates full speed potential)

| Parameter | Value |
|-----------|-------|
| ODE Solver | **Midpoint (2nd order)** |
| NFE | **8 function evaluations → 4 effective steps** |
| Step Schedule | **EPSS (non-uniform)** |
| EPSS Steps | `[0.0, 0.05, 0.18, 0.38, 0.62, 0.82, 0.95, 1.0]` |
| Layer Caching | **SmoothCache** |
| Cache Stacks | Stacks 0 and 1 (lowest change rate) |
| Cache Strategy | Reuse every 2nd step |

#### Midpoint ODE Solver

```
# Standard Euler (1st order):
x_{t+1} = x_t + dt * v(x_t, t)

# Midpoint (2nd order) — same cost as 2 Euler steps, but quality of 4:
x_mid = x_t + (dt/2) * v(x_t, t)
x_{t+1} = x_t + dt * v(x_mid, t + dt/2)
```

Effective quality of 8 Euler NFE using only 4 actual time-steps (8 model evaluations, but organized as 4 midpoint steps).

#### EPSS (Empirically Pruned Step Sampling)

Based on [Fast F5-TTS](https://arxiv.org/abs/2505.19931): flow-matching trajectories change most rapidly in t∈[0.2, 0.8]. Steps are concentrated there:

```
Uniform 8 steps:   [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
EPSS 8 steps:      [0, 0.05, 0.18, 0.38, 0.62, 0.82, 0.95, 1.0]
                         ↑ sparse here        ↑ dense here       ↑ sparse
```

#### SmoothCache

Based on [SmoothCache](https://arxiv.org/abs/2411.10510): layer outputs at low-resolution stacks change slowly between adjacent time-steps.

```
Step 1: Compute ALL stacks [0,1,2,3,4]
Step 2: Compute stacks [2,3,4] only; reuse cached [0,1]
Step 3: Compute ALL stacks [0,1,2,3,4]
Step 4: Compute stacks [2,3,4] only; reuse cached [0,1]
```

Stacks 0,1 operate at 1× and 2× downsampling (longest sequences, most compute) — caching them saves the most.

---

## Speed Breakdown (Optimized Mode)

| Source | Multiplier |
|--------|-----------|
| Midpoint solver (16 NFE Euler → 4 midpoint steps at same quality) | 4× |
| GQA (halve KV computation) | 1.2× |
| Depthwise Separable Conv FFN (3× fewer FFN FLOPs) | 1.3× |
| SmoothCache (skip stacks 0,1 every other step) | 1.4× |
| Dilated ConvNeXt absorbs local workload from attention | 1.1× |
| **Combined estimated speedup** | **~6-8×** |

---

## Expected Performance (Estimated)

### Quality (Standard inference — Euler 16 NFE, for fair comparison)

| Metric | Original ZipVoice (123M) | Model A Slim (55M) | Model B Enhanced (55M) |
|--------|-------------------------|--------------------|-----------------------|
| WER (LibriSpeech-PC) | 1.64% | ~2.2-2.5% | **~1.8-2.1%** |
| SIM-o | 0.668 | ~0.62-0.64 | **~0.64-0.66** |
| UTMOS | 3.98 | ~3.6-3.7 | **~3.7-3.85** |

### Speed (Optimized inference)

| Metric | Model A (Euler 16) | Model B (Euler 16) | Model B (Optimized) |
|--------|--------------------|--------------------|---------------------|
| NFE | 16 | 16 | **4 effective** |
| Relative speed | 1× | ~1.2× (cheaper layers) | **~6-8×** |
| RTF (GPU, estimated) | ~0.06 | ~0.05 | **~0.008** |

---

## Architectural Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ZipVoice-Enhanced (Model B)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │         TEXT ENCODER                 │                    │
│  │  3-layer Zipformer (dim=192)         │                    │
│  │  + 2-layer ConvNeXt refinement       │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────────────────┐                               │
│  │   AVERAGE UPSAMPLING     │ (text tokens → speech frames) │
│  └──────────────┬───────────┘                               │
│                 │                                            │
│  ┌──────────────┴───────────┐                               │
│  │     STYLE ENCODER        │                               │
│  │  ConvNeXt (3 layers)     │                               │
│  │  + 32 GST tokens         │──── style conditioning ───┐   │
│  └──────────────────────────┘                            │   │
│                                                          │   │
│  ┌───────────────────────────────────────────────────────┼──┐│
│  │         FM DECODER (Vector Field Estimator)           │  ││
│  │                                                       ▼  ││
│  │  Stack 0 (1×, 1 layer):  DilConvNeXt → GQA → DepSepFFN ││
│  │  Stack 1 (2×, 2 layers): DilConvNeXt → GQA → DepSepFFN ││
│  │  Stack 2 (4×, 3 layers): DilConvNeXt → GQA → DepSepFFN ││
│  │  Stack 3 (2×, 2 layers): DilConvNeXt → GQA → DepSepFFN ││
│  │  Stack 4 (1×, 1 layer):  DilConvNeXt → GQA → DepSepFFN ││
│  │                                                         ││
│  │  Each layer:                                            ││
│  │  ┌─────────────────────────────────────────────────┐    ││
│  │  │ Input                                           │    ││
│  │  │   ↓                                             │    ││
│  │  │ Dilated ConvNeXt (k=7, dilation=[1,2,4,8])     │    ││
│  │  │   ↓                                             │    ││
│  │  │ GQA Self-Attention (4Q/2KV heads, RoPE)        │    ││
│  │  │   ↓                                             │    ││
│  │  │ BiasNorm                                        │    ││
│  │  │   ↓                                             │    ││
│  │  │ Depthwise Separable Conv FFN (dim→1024→dim)    │    ││
│  │  │   ↓                                             │    ││
│  │  │ BiasNorm + Bypass                               │    ││
│  │  │   ↓                                             │    ││
│  │  │ + Time conditioning + Style conditioning        │    ││
│  │  └─────────────────────────────────────────────────┘    ││
│  │                                                         ││
│  │  Grouped Parameter Sharing:                             ││
│  │  Layers within same stack share base weights            ││
│  │  Each layer has unique rank-32 residual                 ││
│  └─────────────────────────────────────────────────────────┘│
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────────────────┐                               │
│  │     VOCOS VOCODER        │ (mel → waveform, 24kHz)       │
│  └──────────────────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Configuration (Planned)

| Parameter | Value |
|-----------|-------|
| Dataset | LibriTTS (train-clean-100 + train-clean-360 + train-other-500) |
| Total hours | ~585 hours |
| Optimizer | ScaledAdam (Zipformer native) |
| Learning rate | 0.045 (with warmup) |
| Batch strategy | Dynamic batching, max duration per batch |
| Training steps | 500,000 (match Model A for fair comparison) |
| Style encoder training | Joint — receives reference audio segment from same utterance |
| CFG | Unconditional drop: text=0.01, style=0.04, both=0.04 |
| Loss | Flow matching loss (same as ZipVoice) |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Grouped parameter sharing hurts at small scale | Medium | Moderate | Ablate: train one version without sharing |
| GQA with only 2 KV heads insufficient | Low | Low | Can fall back to 4 KV heads (marginal param increase) |
| Depthwise sep conv less expressive than standard FFN | Low | Low | Kernel size 7 covers enough local context for speech |
| ConvNeXt + dilated conv adds complexity without gain | Medium | Low | Monitor WER specifically — alignment quality |
| RoPE doesn't help for fixed-length training | Low | None | Won't hurt; will help at inference with variable lengths |

---

## References

- ZipVoice: [arXiv:2506.13053](https://arxiv.org/abs/2506.13053)
- Zipformer: [arXiv:2310.11230](https://arxiv.org/abs/2310.11230)
- GQA: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- FLY-TTS (DepthSep Conv): [arXiv:2407.00753](https://arxiv.org/abs/2407.00753)
- ResidualTransformer (Parameter Sharing): [arXiv:2310.02489](https://arxiv.org/abs/2310.02489)
- F5-TTS (ConvNeXt, Sway Sampling): [arXiv:2410.06885](https://arxiv.org/abs/2410.06885)
- Fast F5-TTS / EPSS: [arXiv:2505.19931](https://arxiv.org/abs/2505.19931)
- SmoothCache: [arXiv:2411.10510](https://arxiv.org/abs/2411.10510)
- M3-TTS (Joint-DiT): [arXiv:2512.04720](https://arxiv.org/abs/2512.04720)
- Supertonic 3: [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
- DiTTo-TTS: [arXiv:2406.11427](https://arxiv.org/abs/2406.11427)
