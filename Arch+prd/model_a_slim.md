# Astra-TTS Model A: ZipVoice-Slim

## Overview

| Field | Value |
|-------|-------|
| **Model Name** | Astra-TTS Model A (ZipVoice-Slim) |
| **Target Parameters** | ~55M |
| **Base Architecture** | ZipVoice (Zipformer-based flow-matching TTS) |
| **Approach** | Naive dimensional shrink — same architecture, smaller config |
| **Purpose** | Baseline for comparison against enhanced architecture (Model B) |
| **Training Data** | LibriTTS |
| **License** | Apache-2.0 |

---

## Design Philosophy

Model A is a **direct shrink** of the original 123M ZipVoice model to ~55M parameters. No architectural innovations are introduced. All Zipformer components (NLA, attention weight sharing, convolution modules, U-Net downsampling, bypass connections) are preserved exactly as in the original.

The sole changes are:
- Reduced embedding dimensions
- Reduced feedforward dimensions
- Fewer layers per stack

This model serves as the **null hypothesis** — establishing what performance you get at 55M params with no architectural changes.

---

## Architecture Specification

### FM Decoder (Vector Field Estimator)

| Parameter | Original (123M) | Model A (55M) | Change |
|-----------|-----------------|---------------|--------|
| `fm_decoder_dim` | 512 | **384** | -25% |
| `fm_decoder_feedforward_dim` | 1536 | **1024** | -33% |
| `fm_decoder_num_heads` | 4 | 4 | unchanged |
| `fm_decoder_num_layers` | [2, 2, 4, 4, 4] = 16 total | **[2, 2, 3, 2, 2] = 11 total** | -31% |
| `fm_decoder_downsampling_factor` | [1, 2, 4, 2, 1] | [1, 2, 4, 2, 1] | unchanged |
| `fm_decoder_cnn_module_kernel` | [31, 15, 7, 15, 31] | [31, 15, 7, 15, 31] | unchanged |
| NLA module | Yes | Yes | unchanged |
| Attention weight sharing | Yes | Yes | unchanged |
| Bypass connections | Yes | Yes | unchanged |
| Position encoding | Zipformer native | Zipformer native | unchanged |

### Text Encoder

| Parameter | Original (123M) | Model A (55M) | Change |
|-----------|-----------------|---------------|--------|
| `text_encoder_dim` | 192 | 192 | unchanged |
| `text_encoder_feedforward_dim` | 512 | 512 | unchanged |
| `text_encoder_num_layers` | 4 | 4 | unchanged |
| `text_encoder_num_heads` | 4 | 4 | unchanged |
| `text_encoder_cnn_module_kernel` | 9 | 9 | unchanged |

### Shared Parameters

| Parameter | Value |
|-----------|-------|
| `query_head_dim` | 32 |
| `value_head_dim` | 12 |
| `pos_head_dim` | 4 |
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

| Component | Params |
|-----------|--------|
| Text Encoder (4 layers, dim=192, ff=512) | ~5M |
| FM Decoder (11 layers, dim=384, ff=1024) | ~47M |
| Embeddings (time, text, positional) | ~3M |
| **Total** | **~55M** |

---

## Inference Configuration

Standard ZipVoice inference — no optimizations applied:

| Parameter | Value |
|-----------|-------|
| ODE Solver | Euler (1st order) |
| NFE (Number of Function Evaluations) | 16 |
| Step Schedule | Uniform `[0, 1/16, 2/16, ..., 1]` |
| CFG (Classifier-Free Guidance) | Time-dependent (text-only early, both late) |
| Layer Caching | None |

---

## What Is Preserved From Original ZipVoice

All of these are **critical** per the ZipVoice ablation study and are kept intact:

| Component | Reason to keep |
|-----------|---------------|
| Convolution modules | Removing causes WER 1.69% → 9.79% |
| U-Net downsampling [1,2,4,2,1] | Removing drops SIM-o 0.610 → 0.557 |
| Bypass connections | Removing causes total collapse (WER 98%) |
| NLA module | Contributes to quality (UTMOS +0.17) |
| Attention weight sharing | Improves efficiency at no quality cost |
| Average upsampling (text→speech alignment) | Removing causes WER 1.69% → 20.19% |

---

## Training Configuration (Planned)

| Parameter | Value |
|-----------|-------|
| Dataset | LibriTTS (train-clean-100 + train-clean-360 + train-other-500) |
| Total hours | ~585 hours |
| Optimizer | ScaledAdam (Zipformer native) |
| Learning rate | 0.045 (with warmup) |
| Batch strategy | Dynamic batching, max duration per batch |
| Training steps | 500,000 (match Model B for fair comparison) |

---

## Expected Performance (Estimated)

| Metric | Original ZipVoice (123M) | Model A (55M) |
|--------|-------------------------|---------------|
| WER (LibriSpeech-PC) | 1.64% | ~2.2-2.5% |
| SIM-o | 0.668 | ~0.62-0.64 |
| UTMOS | 3.98 | ~3.6-3.7 |
| Inference RTF (GPU) | baseline | ~1.5× faster (fewer params) |

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Quality drops more than expected at 55M | Medium | Can try intermediate size (70M) |
| Training instability with smaller model | Low | ScaledAdam handles this well |
| Diminishing returns from shared attention at smaller dim | Low | Monitor training curves |

---

## References

- ZipVoice paper: [arXiv:2506.13053](https://arxiv.org/abs/2506.13053)
- Zipformer paper: [arXiv:2310.11230](https://arxiv.org/abs/2310.11230)
- Original checkpoint: [k2-fsa/ZipVoice](https://huggingface.co/k2-fsa/ZipVoice)
