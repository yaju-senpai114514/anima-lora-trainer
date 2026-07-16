#!/usr/bin/env bash
set -euo pipefail

REPO="$( cd "$( dirname "$0" )" && pwd -P )"
cd "$REPO"

# 2-GPU 버전: 2개 GPU 에 로컬 배치 2씩 분배해 글로벌 배치 4 를 만든다
# (1-GPU train_batch_size=4 와 동등).
#
# ★ 로컬 배치를 정하는 것은 dataset toml 의 batch_size 다. --train_batch_size 가 아니다.
#   sd-scripts 는 --dataset_config 가 있으면 toml 값을 우선한다
#   (library/config_util.py search_value: [dataset, general, argparse] 순으로 첫 non-None).
#   --train_batch_size 는 로그와 LoRA 메타데이터에만 쓰인다
#   (train_network.py 의 ss_batch_size_per_device / total_batch_size).
#   그래서 여기서는 toml 을 읽어 실제 글로벌 배치를 검증하고, 같은 값을
#   --train_batch_size 로도 넘겨 로그/메타데이터가 실제와 일치하게 한다.
NUM_GPUS=2

usage() {
  cat <<'EOF'
Usage: ./3_run_training_2gpu.sh <trigger>

  <trigger>  학습할 데이터셋/설정 이름. 다음 두 파일이 있어야 한다:
               - config.<trigger>.env    (LoRA/학습 하이퍼파라미터)
               - dataset/<trigger>.toml  (dataset_config; 2_make_config.py 산출)
             모델 경로는 config_model.env 에서 공통으로 읽는다.

             config.<trigger>.env 의 선택 변수(설정 시 사용, 1gpu 스크립트와 공통 API):
               - DATASET_TOML   커스텀 dataset_config 경로(멀티해상도 toml 등). 기본 dataset/<trigger>.toml
               - SEED           난수 시드 고정(멀티GPU 캡션 셔플 일관성). 미설정 시 랜덤
               - OUT_DIR        출력 디렉토리 오버라이드(절대/REPO상대). 미설정 시 기본 규칙

  이 스크립트는 2-GPU 전용이다:
               - GPU당 배치 = toml 의 batch_size (--train_batch_size 가 아님!)
               - toml batch_size × 2 == config 의 BATCH_SIZE(글로벌) 를 검증한다.
               - 글로벌 4 로 돌리려면:  2_make_config.py <trigger> --batch-size 2 --force

  예:  ./3_run_training_2gpu.sh mychar

  파이프라인: 0_dedup_raw → 1_tag_dataset → 2_make_config → 3_run_training
EOF
}

# --- 트리거 인자 검사 ---
if [ "$#" -ge 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  usage
  exit 0
fi
if [ "$#" -ne 1 ] || [ -z "${1:-}" ]; then
  echo "[error] 트리거 인자 1개가 필요합니다." >&2
  usage >&2
  exit 1
fi
TRIGGER="$1"

CONFIG="$REPO/config.${TRIGGER}.env"
if [ ! -f "$CONFIG" ]; then
  echo "[error] 설정 파일 없음: config.${TRIGGER}.env" >&2
  avail="$(ls config.*.env 2>/dev/null | sed 's/^config\.//; s/\.env$//' | tr '\n' ' ')"
  echo "        사용 가능한 트리거: ${avail:-<없음>}" >&2
  exit 1
fi
if [ ! -f "$REPO/config_model.env" ]; then
  echo "[error] 모델 경로 설정 없음: config_model.env" >&2
  exit 1
fi

echo "Loading config.${TRIGGER}.env + config_model.env ..."
source "$CONFIG"
source "$REPO/config_model.env"

VENV="$REPO/.venv"

# --- dataset_config 결정: config 의 DATASET_TOML 우선(멀티해상도 등 커스텀 toml), 없으면 dataset/<trigger>.toml ---
if [ -n "${DATASET_TOML:-}" ]; then
  case "$DATASET_TOML" in
    /*) TOML="$DATASET_TOML" ;;
    *)  TOML="$REPO/$DATASET_TOML" ;;
  esac
else
  TOML="$REPO/dataset/${TRIGGER}.toml"
fi
if [ ! -f "$TOML" ]; then
  echo "[error] dataset_config 없음: $TOML" >&2
  echo "        (config 의 DATASET_TOML 로 커스텀 지정 가능; 미지정 시 dataset/${TRIGGER}.toml)" >&2
  echo "        먼저: uv run python 2_make_config.py ${TRIGGER}" >&2
  exit 1
fi

# --- 실제 로컬 배치를 toml 에서 읽는다 (sd-scripts 의 해석 순서를 그대로 흉내낸다) ---
TOML_BATCH="$("$VENV/bin/python" - "$TOML" <<'PY'
import sys, tomllib

with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
general = cfg.get("general", {})
datasets = cfg.get("datasets", [])
if not datasets:
    print("EMPTY")
    sys.exit()
# sd-scripts 우선순위: [[datasets]].batch_size > [general].batch_size > --train_batch_size
vals = {d.get("batch_size", general.get("batch_size")) for d in datasets}
if len(vals) > 1:
    print("MIXED:" + ",".join(str(v) for v in sorted(vals, key=str)))
else:
    v = vals.pop()
    print("NONE" if v is None else v)
PY
)"

if [ -z "${BATCH_SIZE:-}" ]; then
  echo "[error] config.${TRIGGER}.env 에 BATCH_SIZE(글로벌 배치)가 없습니다." >&2
  exit 1
fi
WANT=$(( BATCH_SIZE / NUM_GPUS ))
if [ $(( WANT * NUM_GPUS )) -ne "$BATCH_SIZE" ]; then
  echo "[error] BATCH_SIZE(글로벌)=$BATCH_SIZE 는 GPU ${NUM_GPUS}개로 나누어떨어지지 않습니다." >&2
  exit 1
fi

case "$TOML_BATCH" in
  EMPTY)
    echo "[error] toml 에 [[datasets]] 가 없습니다: $TOML" >&2
    exit 1 ;;
  MIXED:*)
    echo "[error] toml 의 [[datasets]] 마다 batch_size 가 다릅니다: ${TOML_BATCH#MIXED:}" >&2
    echo "        $TOML" >&2
    echo "        (모든 블록이 같아야 글로벌 배치를 계산할 수 있습니다)" >&2
    exit 1 ;;
  NONE)
    # 권장 경로: toml 이 batch_size 를 안 정하면 --train_batch_size 가 실제 배치가 된다.
    # 같은 toml 로 1/2/4 GPU 를 다 돌릴 수 있고 로그/메타데이터도 자동으로 맞는다.
    LOCAL_BATCH="$WANT" ;;
  *)
    LOCAL_BATCH="$TOML_BATCH" ;;
esac

# --- 글로벌 배치 검증: toml 의 GPU당 배치 × GPU 수 == config 의 BATCH_SIZE ---
if [ $(( LOCAL_BATCH * NUM_GPUS )) -ne "$BATCH_SIZE" ]; then
  echo "[error] 글로벌 배치가 맞지 않습니다." >&2
  echo "        toml 의 batch_size (= GPU당 배치) : $LOCAL_BATCH   ($(basename "$TOML"))" >&2
  echo "        × GPU ${NUM_GPUS}개                       : $(( LOCAL_BATCH * NUM_GPUS ))" >&2
  echo "        config.${TRIGGER}.env 의 BATCH_SIZE  : $BATCH_SIZE  (글로벌 배치)" >&2
  echo "" >&2
  echo "        toml 에 batch_size 가 적혀 있으면 sd-scripts 가 그걸 우선하고" >&2
  echo "        --train_batch_size 는 무시됩니다 (로그/메타데이터에만 반영)." >&2
  echo "" >&2
  echo "        → toml 에서 batch_size 를 빼면 GPU 수와 무관하게 맞습니다 (권장):" >&2
  echo "          uv run python 2_make_config.py ${TRIGGER} --force" >&2
  echo "        → toml 에 고정해야 한다면 GPU당 배치로 맞추세요:" >&2
  echo "          uv run python 2_make_config.py ${TRIGGER} --batch-size ${WANT} --force" >&2
  exit 1
fi

# --- gradient checkpointing 토글 (config.<trigger>.env 의 GRADIENT_CHECKPOINTING; 기본 on) ---
# VRAM 절약(느려짐) vs 속도. 미설정 시 기존 동작 유지(활성화).
GC_ARG=()
case "${GRADIENT_CHECKPOINTING:-1}" in
  1|true|True|yes|on)   GC_ARG=(--gradient_checkpointing) ;;
  0|false|False|no|off) : ;;
  *) echo "[warn] GRADIENT_CHECKPOINTING='${GRADIENT_CHECKPOINTING}' 해석 불가 → 활성화" >&2
     GC_ARG=(--gradient_checkpointing) ;;
esac

# --- seed: config 의 SEED 설정 시 고정. 멀티GPU 캡션 셔플 일관성 등. 미설정 시 미전달(랜덤). ---
SEED_ARG=()
if [ -n "${SEED:-}" ]; then
  SEED_ARG=(--seed="$SEED")
fi

echo "Starting LoRA training for <$TRIGGER> on ${NUM_GPUS} GPUs  " \
     "(global_batch=$BATCH_SIZE = ${NUM_GPUS}×${LOCAL_BATCH}, grad_acc=$GRADIENT_ACC_STEPS," \
     "epochs=$EPOCHS lr=$LR save_every=$SAVE_EVERY_N_EPOCHS gc=${GRADIENT_CHECKPOINTING:-1}" \
     "seed=${SEED:-random} toml=$(basename "$TOML"))..."

# 출력 디렉토리: config 의 OUT_DIR 로 오버라이드 가능(절대경로 또는 REPO 기준 상대). 미설정 시 기본 규칙.
# 기본 규칙은 글로벌 배치(gb${BATCH_SIZE})를 쓰므로 1-GPU 실행과 동일 → resume 호환.
if [ -n "${OUT_DIR:-}" ]; then
  case "$OUT_DIR" in
    /*) OUT="$OUT_DIR" ;;
    *)  OUT="$REPO/$OUT_DIR" ;;
  esac
else
  OUT="$REPO/output/${TRIGGER}_gb${BATCH_SIZE}_lr${LR}_ep${EPOCHS}"
fi
OUT_LOG="$OUT/logs"
NAME="anima_${TRIGGER}_r${NETWORK_DIM}"

mkdir -p "$OUT" "$OUT_LOG"
cd "$REPO/sd-scripts"

# 최종 LoRA 가 이미 있으면 스킵
if [ -f "$OUT/$NAME.safetensors" ]; then
  echo "[$TRIGGER] Final LoRA already exists ($OUT/$NAME.safetensors). Skipping."
  exit 0
fi

# 자동 resume: 가장 최신 *-state 있으면 이어서
RESUME_ARG=()
STATE=$(ls -dt "$OUT/"*-state 2>/dev/null | head -1 || true)
if [ -n "${STATE:-}" ]; then
  echo "[$TRIGGER] Resuming from training state: $STATE"
  RESUME_ARG=(--resume "$STATE")
else
  echo "[$TRIGGER] No prior state found. Starting fresh."
fi

"$VENV/bin/accelerate" launch \
  --num_cpu_threads_per_process 2 \
  --multi_gpu \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  anima_train_network.py \
  --pretrained_model_name_or_path="$PATH_MODEL" \
  --qwen3="$PATH_QWEN" \
  --vae="$PATH_VAE" \
  --dataset_config="$TOML" \
  --output_dir="$OUT" \
  --output_name="$NAME" \
  --save_model_as="safetensors" \
  --network_module="networks.lora_anima" \
  --network_train_unet_only \
  --network_dim="$NETWORK_DIM" \
  --network_alpha="$NETWORK_ALPHA" \
  --train_batch_size="$LOCAL_BATCH" \
  --gradient_accumulation_steps="$GRADIENT_ACC_STEPS" \
  --optimizer_type="AdamW8bit" \
  --learning_rate="$LR" \
  --lr_scheduler="$LR_SCHED" \
  --lr_warmup_steps="$LR_WARMUP_STEPS" \
  --max_train_epochs="$EPOCHS" \
  --save_every_n_epochs="$SAVE_EVERY_N_EPOCHS" \
  --save_state \
  --save_last_n_epochs_state=2 \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --cache_latents \
  --cache_latents_to_disk \
  --qwen_image_vae_2d \
  --caption_extension=".txt" \
  --timestep_sampling="sigmoid" \
  --console_log_simple \
  --logging_dir="$OUT_LOG" \
  "${SEED_ARG[@]}" \
  "${GC_ARG[@]}" \
  "${RESUME_ARG[@]}"
