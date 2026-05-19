# Astra-TTS Benchmark PRD: Architecture Comparison

## Overview

| Field | Value |
|-------|-------|
| **Project** | Astra-TTS Architecture Evaluation |
| **Goal** | Determine whether architectural improvements (Model B) outperform naive shrinking (Model A) at ~55M params |
| **Dataset** | LibriTTS (English) |
| **Baseline** | Original ZipVoice 123M (k2-fsa/ZipVoice) |
| **Models Under Test** | Model A (Slim 55M), Model B (Enhanced 55M) |
| **Evaluation Configurations** | 4 total (see below) |

---

## Hypothesis

> At the same parameter budget (~55M), architectural improvements (GQA, Depthwise Separable Conv, Grouped Parameter Sharing, Dilated ConvNeXt, RoPE, ConvNeXt text refinement, removed NLA) will yield **equal or better quality** than naive shrinking, while enabling **significantly faster inference** through EPSS + Midpoint + SmoothCache.

---

## Models & Configurations to Evaluate

| ID | Model | Params | Inference Mode | Purpose |
|----|-------|--------|---------------|---------|
| **Baseline** | ZipVoice Original | 123M | Euler 16 NFE uniform | Reference — published numbers |
| **A-std** | Model A (Slim) | 55M | Euler 16 NFE uniform | Naive shrink baseline |
| **B-std** | Model B (Enhanced) | 55M | Euler 16 NFE uniform | **Quality comparison** (fair, same inference as A) |
| **B-opt** | Model B (Enhanced) | 55M | Midpoint 4-step + EPSS + SmoothCache | **Speed comparison** (full optimized stack) |

### Why these 4?

- **Baseline vs A-std**: How much does shrinking from 123M→55M cost in quality?
- **A-std vs B-std**: Do arch improvements help at same size and same inference? (Quality ablation)
- **B-std vs B-opt**: How much speed do inference optimizations add? (Speed ablation)
- **A-std vs B-opt**: The real comparison — same params, but B is faster AND better?

---

## Training Protocol

All models (A and B) must be trained under **identical conditions** for fair comparison:

| Parameter | Value |
|-----------|-------|
| **Dataset** | LibriTTS (train-clean-100 + train-clean-360 + train-other-500) |
| **Total hours** | ~585 hours |
| **Audio preprocessing** | Resample to 24kHz, trim silence, normalize volume |
| **Text preprocessing** | IPA phonemization via eSpeak-ng (same as ZipVoice) |
| **Optimizer** | ScaledAdam |
| **Learning rate** | 0.045 (linear warmup 5000 steps) |
| **Batch strategy** | Dynamic batching, max 300s total duration per batch |
| **Training steps** | 500,000 steps (both models, same count) |
| **Gradient clipping** | 1.0 |
| **EMA** | 0.9999 (for evaluation) |
| **Random seed** | Fixed (42) for reproducibility |
| **Mixed precision** | bf16 |
| **Checkpoint selection** | Best validation loss OR step 500k (whichever is reported) |

### Baseline Model

The original ZipVoice 123M checkpoint from [k2-fsa/ZipVoice](https://huggingface.co/k2-fsa/ZipVoice) is used directly. The `zipvoice_libritts/model.pt` variant (trained on LibriTTS) is the correct baseline since our models are also trained on LibriTTS.

---

## Evaluation Protocol

### Test Sets

| Test Set | Samples | Purpose |
|----------|---------|---------|
| **LibriSpeech-PC test-clean** | Standard partition | Primary benchmark (matches ZipVoice paper) |
| **Seed-TTS test-en** | Standard partition | Cross-domain zero-shot evaluation |

### Evaluation Procedure

For each test utterance:
1. Select a reference audio clip (3-10 seconds) from the same speaker
2. Provide the reference audio + reference transcription + target text to the model
3. Generate speech
4. Measure metrics against ground truth

### Metrics

#### Quality Metrics

| Metric | What it measures | Tool | Target range |
|--------|-----------------|------|-------------|
| **WER** (Word Error Rate) | Intelligibility — can you understand the words? | Whisper-large-v3 transcription → WER vs ground truth text | Lower is better. ZipVoice baseline: 1.64% |
| **SIM-o** (Speaker Similarity - original) | Voice cloning quality — does it sound like the target speaker? | WavLM-TDNN speaker verification model, cosine similarity between generated and original target audio | Higher is better. ZipVoice baseline: 0.668 |
| **UTMOS** | Naturalness/quality — does it sound like real speech? | UTMOS predictor (pretrained MOS estimator) | Higher is better. ZipVoice baseline: 3.98 |

#### Speed Metrics

| Metric | What it measures | How |
|--------|-----------------|-----|
| **RTF** (Real-Time Factor) | Time to generate / duration of generated audio | Measure wall-clock inference time, divide by audio length |
| **NFE** (Number of Function Evaluations) | Model forward passes per utterance | Count |
| **Latency (s)** | Absolute time for a 10-second utterance | Measure on fixed hardware |
| **Peak Memory (MB)** | Maximum GPU/CPU memory during inference | torch.cuda.max_memory_allocated() |

#### Speed Evaluation Hardware

All speed metrics measured on:
- **GPU**: Single NVIDIA A100 80GB (for GPU RTF)
- **CPU**: Single-threaded Intel Xeon (for CPU RTF)
- **Batch size**: 1 (real-world latency scenario)
- **Warm-up**: 10 utterances discarded before timing
- **Measurement**: Mean of 50 utterances ± std dev

---

## Success Criteria

### Primary (Must achieve to validate Model B)

| Criterion | Condition | Rationale |
|-----------|-----------|-----------|
| **B-std quality ≥ A-std quality** | B-std WER ≤ A-std WER AND B-std UTMOS ≥ A-std UTMOS | Arch changes must not hurt quality |
| **B-opt quality ≈ B-std quality** | B-opt WER within +0.3% of B-std AND B-opt UTMOS within -0.1 of B-std | Inference optimizations must be near-lossless |
| **B-opt speed > A-std speed** | B-opt RTF < 0.5 × A-std RTF | Must be at least 2× faster |

### Stretch Goals

| Criterion | Condition | What it would prove |
|-----------|-----------|---------------------|
| B-std matches Baseline quality | B-std WER ≤ 2.0% AND UTMOS ≥ 3.8 | Enhanced 55M achieves near-123M quality |
| B-opt achieves 5×+ speedup | B-opt RTF < 0.2 × A-std RTF | Full optimization stack works at scale |
| B-std WER < A-std WER by >0.3% | Statistical significance (p<0.05) | ConvNeXt/GQA/RoPE genuinely help alignment |

### Failure Criteria (Abort/revise)

| Condition | Action |
|-----------|--------|
| B-std quality < A-std on ALL metrics | Arch changes hurt → revert to simpler model, investigate which change caused regression |
| B-opt quality degrades >10% vs B-std | Inference optimizations too aggressive → relax cache schedule or increase NFE |
| Both A-std and B-std WER > 4% | 55M is too small for this task → increase param budget to 70-80M |

---

## Ablation Matrix (Optional, if time allows)

To understand **which specific change** helped or hurt, run these single-change ablations from Model B:

| Ablation | Change from Model B | What it tests |
|----------|--------------------| --------------|
| B - GQA | Use 4 KV heads instead of 2 | Is GQA actually free at this scale? |
| B - DepSepConv | Use standard Linear FFN | Is depthwise sep conv as good as linear? |
| B - ParamSharing | No weight sharing between layers | Does sharing work at 55M? |
| B - DilatedConvNeXt | Remove dilated conv, attention only | Does local conv help? |
| B - RoPE | Use Zipformer native pos enc | Does RoPE matter? |
| B - ConvNeXt Text | Remove text refinement, use 4 encoder layers | Does ConvNeXt help WER? |
| B + NLA | Add NLA back | Was removing NLA actually fine? |

Each ablation trains for same 500k steps. Report WER + UTMOS + SIM-o for each.

---

## Inference Optimization Ablation

To understand which inference trick contributes most:

| Config | Solver | NFE | Steps | Cache | Expected speed |
|--------|--------|-----|-------|-------|---------------|
| B-std | Euler | 16 | Uniform | None | 1× |
| B + EPSS only | Euler | 8 | EPSS | None | ~2× |
| B + Midpoint only | Midpoint | 8 (4 steps) | Uniform | None | ~2× |
| B + Cache only | Euler | 16 | Uniform | SmoothCache | ~1.4× |
| B + EPSS + Midpoint | Midpoint | 8 (4 steps) | EPSS | None | ~4× |
| B-opt (all) | Midpoint | 8 (4 steps) | EPSS | SmoothCache | ~6× |

Report quality (WER, UTMOS, SIM-o) AND speed (RTF) for each.

---

## Reporting Format

### Results Table (Template)

```markdown
| Model | Config | WER↓ | SIM-o↑ | UTMOS↑ | RTF↓ | NFE | Mem (MB) |
|-------|--------|------|--------|--------|------|-----|----------|
| ZipVoice (Baseline) | Euler 16 | 1.64 | 0.668 | 3.98 | X.XX | 16 | XXX |
| Model A (Slim) | Euler 16 | X.XX | X.XXX | X.XX | X.XX | 16 | XXX |
| Model B (Enhanced) | Euler 16 | X.XX | X.XXX | X.XX | X.XX | 16 | XXX |
| Model B (Enhanced) | Optimized | X.XX | X.XXX | X.XX | X.XX | 4eff | XXX |
```

### Visualizations Required

1. **Bar chart**: WER comparison (Baseline vs A vs B-std vs B-opt)
2. **Bar chart**: UTMOS comparison (same)
3. **Scatter plot**: Quality (UTMOS) vs Speed (RTF) — Pareto frontier
4. **Training curves**: Loss vs steps for Model A and Model B (convergence comparison)
5. **Ablation heatmap**: Each change → metric delta

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **1. Implementation** | 1-2 weeks | Model A and B code, training scripts |
| **2. Training** | 2-3 weeks | Both models trained for 500k steps |
| **3. Evaluation** | 3-5 days | All metrics computed |
| **4. Ablations** (optional) | 1-2 weeks | Per-change ablation results |
| **5. Report** | 2-3 days | Final comparison document with conclusions |

---

## Decision Framework

After results are in:

```
IF B-std > A-std (quality) AND B-opt >> A-std (speed):
    → Model B wins. Proceed to Malayalam training with Model B architecture.
    
ELIF B-std ≈ A-std (quality within noise) AND B-opt >> A-std (speed):
    → Model B wins on speed alone. Still proceed with B.
    
ELIF B-std < A-std (quality regression):
    → Investigate via ablations. Remove harmful changes.
    → Create Model B' with only beneficial changes.
    → Re-evaluate B' vs A.
    
ELIF both A and B unacceptably bad (WER > 4%):
    → 55M is too small. Scale up to 70-80M.
    → Or revisit training config (more steps, different lr).
```

---

## Dependencies

| Dependency | Source | Status |
|-----------|--------|--------|
| ZipVoice training code | [github.com/k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | Available |
| LibriTTS dataset | [OpenSLR](https://www.openslr.org/60/) | Available |
| ZipVoice LibriTTS checkpoint (baseline) | [k2-fsa/ZipVoice](https://huggingface.co/k2-fsa/ZipVoice) `zipvoice_libritts/model.pt` | Available |
| Whisper-large-v3 (WER eval) | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | Available |
| UTMOS predictor | [sarulab-speech/UTMOS](https://github.com/sarulab-speech/UTMOS22) | Available |
| WavLM-TDNN (speaker similarity) | [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) | Available |
| Vocos vocoder | Bundled with ZipVoice | Available |

---

## References

- ZipVoice: [arXiv:2506.13053](https://arxiv.org/abs/2506.13053)
- Zipformer: [arXiv:2310.11230](https://arxiv.org/abs/2310.11230)
- Fast F5-TTS / EPSS: [arXiv:2505.19931](https://arxiv.org/abs/2505.19931)
- SmoothCache: [arXiv:2411.10510](https://arxiv.org/abs/2411.10510)
- F5-TTS: [arXiv:2410.06885](https://arxiv.org/abs/2410.06885)
- M3-TTS: [arXiv:2512.04720](https://arxiv.org/abs/2512.04720)
- GQA: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- FLY-TTS: [arXiv:2407.00753](https://arxiv.org/abs/2407.00753)
- ResidualTransformer: [arXiv:2310.02489](https://arxiv.org/abs/2310.02489)
- Supertonic 3: [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
