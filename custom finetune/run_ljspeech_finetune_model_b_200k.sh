#!/usr/bin/env bash

# Fine-tune Astra-TTS Model B from the 200k HF checkpoint on LJSpeech.
# Run from anywhere inside the Astra-TTS repo or directly from this folder.

set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
zipvoice_dir="${repo_dir}/ZipVoice"

export PYTHONPATH="${zipvoice_dir}:${PYTHONPATH:-}"

stage="${stage:-1}"
stop_stage="${stop_stage:-6}"

hf_repo="${hf_repo:-Praha-Labs/astra-tts-model-b-enhanced-200k}"
checkpoint_name="${checkpoint_name:-checkpoint-200000.pt}"
checkpoint_state="${checkpoint_state:-model_avg}"

work_dir="${work_dir:-/workspace/astra_ljspeech_finetune}"
data_dir="${data_dir:-${work_dir}/data}"
raw_dir="${raw_dir:-${data_dir}/raw}"
manifest_dir="${manifest_dir:-${data_dir}/manifests}"
fbank_dir="${fbank_dir:-${data_dir}/fbank}"
model_dir="${model_dir:-${work_dir}/base_model}"
exp_dir="${exp_dir:-/workspace/astra_model_b_enhanced_ljspeech_ft_adamw_30k}"

nj="${nj:-8}"
num_iters="${num_iters:-30000}"
save_every_n="${save_every_n:-1000}"
max_duration="${max_duration:-30}"
max_len="${max_len:-12}"
base_lr="${base_lr:-5e-6}"
use_fp16="${use_fp16:-0}"
accum_grad_batches="${accum_grad_batches:-1}"
dev_size="${dev_size:-500}"
optimizer="${optimizer:-adamw}"
adamw_weight_decay="${adamw_weight_decay:-0.01}"
grad_clip="${grad_clip:-1.0}"
finetune_checkpoint="${finetune_checkpoint:-${model_dir}/model-avg-${checkpoint_name}}"
condition_drop_ratio="${condition_drop_ratio:-0.0}"
disable_aux_grad_penalties="${disable_aux_grad_penalties:-1}"
disable_finetune_stochastic_modules="${disable_finetune_stochastic_modules:-1}"
finetune_batch_count_offset="${finetune_batch_count_offset:-0}"
freeze_modules="${freeze_modules:-}"
unfreeze_modules="${unfreeze_modules:-fm_decoder}"
debug_nonfinite_grads="${debug_nonfinite_grads:-1}"
max_nonfinite_grad_steps="${max_nonfinite_grad_steps:-5}"

ljspeech_url="${ljspeech_url:-https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2}"
ljspeech_archive="${raw_dir}/LJSpeech-1.1.tar.bz2"
ljspeech_dir="${raw_dir}/LJSpeech-1.1"

mkdir -p "${raw_dir}" "${manifest_dir}" "${fbank_dir}" "${model_dir}" "${exp_dir}"

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "Stage 1: Download and extract LJSpeech"
  if [ ! -f "${ljspeech_archive}" ]; then
    wget -O "${ljspeech_archive}" "${ljspeech_url}"
  fi
  if [ ! -d "${ljspeech_dir}" ]; then
    tar -xjf "${ljspeech_archive}" -C "${raw_dir}"
  fi
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "Stage 2: Create custom train/dev TSV files"
  python3 - <<PY
from pathlib import Path

ljspeech_dir = Path("${ljspeech_dir}")
out_dir = Path("${raw_dir}")
dev_size = int("${dev_size}")

metadata = ljspeech_dir / "metadata.csv"
rows = []
with metadata.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 3:
            continue
        utt_id, raw_text, normalized_text = parts[:3]
        text = (normalized_text or raw_text).strip()
        wav = (ljspeech_dir / "wavs" / f"{utt_id}.wav").resolve()
        if wav.is_file() and text:
            rows.append((utt_id, text, wav))

if len(rows) <= dev_size:
    raise RuntimeError(f"Not enough LJSpeech rows ({len(rows)}) for dev_size={dev_size}")

train_rows = rows[:-dev_size]
dev_rows = rows[-dev_size:]

def write_tsv(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for utt_id, text, wav in items:
            safe_text = " ".join(text.replace("\t", " ").split())
            f.write(f"{utt_id}\t{safe_text}\t{wav}\n")

write_tsv(out_dir / "ljspeech_train.tsv", train_rows)
write_tsv(out_dir / "ljspeech_dev.tsv", dev_rows)
print(f"train rows: {len(train_rows)}")
print(f"dev rows: {len(dev_rows)}")
PY
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  echo "Stage 3: Prepare Lhotse manifests"
  for subset in train dev; do
    python3 -m zipvoice.bin.prepare_dataset \
      --tsv-path "${raw_dir}/ljspeech_${subset}.tsv" \
      --prefix custom-finetune \
      --subset "raw_${subset}" \
      --num-jobs "${nj}" \
      --output-dir "${manifest_dir}"
  done

  echo "Stage 3b: Add LibriTTS tokens"
  for subset in train dev; do
    python3 -m zipvoice.bin.prepare_tokens \
      --input-file "${manifest_dir}/custom-finetune_cuts_raw_${subset}.jsonl.gz" \
      --output-file "${manifest_dir}/custom-finetune_cuts_${subset}.jsonl.gz" \
      --tokenizer libritts \
      --lang en-us \
      --num-jobs "${nj}"
  done
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
  echo "Stage 4: Compute Vocos fbank features"
  for subset in train dev; do
    python3 -m zipvoice.bin.compute_fbank \
      --source-dir "${manifest_dir}" \
      --dest-dir "${fbank_dir}" \
      --dataset custom-finetune \
      --subset "${subset}" \
      --num-jobs "${nj}"
  done
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  echo "Stage 5: Download base Model B checkpoint from Hugging Face"
  hf download "${hf_repo}" \
    --repo-type model \
    --local-dir "${model_dir}"

  for file in "${checkpoint_name}" model.json tokens.txt; do
    test -f "${model_dir}/${file}" || {
      echo "Missing ${model_dir}/${file}. Check hf_repo/checkpoint_name." >&2
      exit 1
    }
  done

  echo "Stage 5b: Export weights-only checkpoint for fine-tuning"
  python3 - <<PY
import torch

src = "${model_dir}/${checkpoint_name}"
dst = "${finetune_checkpoint}"

ckpt = torch.load(src, map_location="cpu", weights_only=False)
requested = "${checkpoint_state}"
if requested == "model_avg" and "model_avg" in ckpt and ckpt["model_avg"] is not None:
    state = ckpt["model_avg"]
    source = "model_avg"
elif requested == "model":
    state = ckpt["model"]
    source = "model"
elif "model_avg" in ckpt and ckpt["model_avg"] is not None:
    state = ckpt["model_avg"]
    source = "model_avg"
else:
    state = ckpt["model"]
    source = "model"

torch.save({"model": state}, dst)
print(f"Saved stripped weights-only checkpoint: {dst} from {source}")
print("Removed keys:", sorted(k for k in ckpt.keys() if k != source))
PY
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
  echo "Stage 6: Fine-tune Model B on LJSpeech"
  python3 -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 "${use_fp16}" \
    --finetune 1 \
    --optimizer "${optimizer}" \
    --adamw-weight-decay "${adamw_weight_decay}" \
    --grad-clip "${grad_clip}" \
    --disable-aux-grad-penalties "${disable_aux_grad_penalties}" \
    --disable-finetune-stochastic-modules "${disable_finetune_stochastic_modules}" \
    --finetune-batch-count-offset "${finetune_batch_count_offset}" \
    --freeze-modules "${freeze_modules}" \
    --unfreeze-modules "${unfreeze_modules}" \
    --debug-nonfinite-grads "${debug_nonfinite_grads}" \
    --max-nonfinite-grad-steps "${max_nonfinite_grad_steps}" \
    --num-iters "${num_iters}" \
    --save-every-n "${save_every_n}" \
    --max-duration "${max_duration}" \
    --accum-grad-batches "${accum_grad_batches}" \
    --base-lr "${base_lr}" \
    --condition-drop-ratio "${condition_drop_ratio}" \
    --max-len "${max_len}" \
    --valid-by-epoch 0 \
    --model-config "${model_dir}/model.json" \
    --checkpoint "${finetune_checkpoint}" \
    --tokenizer libritts \
    --token-file "${model_dir}/tokens.txt" \
    --dataset custom \
    --train-manifest "${fbank_dir}/custom-finetune_cuts_train.jsonl.gz" \
    --dev-manifest "${fbank_dir}/custom-finetune_cuts_dev.jsonl.gz" \
    --exp-dir "${exp_dir}"
fi

echo "Done. Fine-tune output: ${exp_dir}"
