# anima-lora-trainer

**Anima (Qwen-Image) 캐릭터/스타일 LoRA 학습 파이프라인.**

WD14 태깅 → blacklist 제거 / trigger 부착 → 데이터셋 config(toml) 생성 → [kohya-ss/sd-scripts]
의 `anima_train_network.py` 로 LoRA 학습까지 한 번에 이어지는 3-스텝 파이프라인이다.

```
이미지 폴더  ──1──▶  *.txt 캡션  ──2──▶  dataset/<name>.toml  ──3──▶  output/<name>/anima_<name>_r32.safetensors
            태깅            설정 생성                  학습
```

태깅 로직(`WD14Tagger`)은 `app.py` 한 곳에 있고 `1_tag_dataset.py` 가 이를 재사용하므로
전처리/모델/threshold 동작이 일관된다.

---

## 1. 사전 요구사항

| 항목 | 비고 |
|------|------|
| Linux + NVIDIA GPU | RTX 3090(24GB)에서 검증. 학습은 24GB에 여유 있게 들어간다. |
| NVIDIA 드라이버 + CUDA 12.x | `pyproject.toml` 은 **CUDA 12.4용 torch(cu124)** 를 받는다. 다른 CUDA면 아래 참고. |
| [uv](https://docs.astral.sh/uv/) | 파이썬/의존성 관리. `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Python 3.12 | `.python-version` 으로 고정. uv 가 자동으로 받아준다. |
| Anima 모델 3종 | ComfyUI 모델 디렉터리에 있어야 한다(아래). |

### 필요한 모델 파일

`config_model.env` 의 `PATH_MODELS_BASE` 기준 경로에 다음 3개가 존재해야 한다(기본값은 `$HOME/workspace/ComfyUI/models`):

```
$PATH_MODELS_BASE/diffusion_models/anima_baseV10.safetensors   # Anima base (DiT)
$PATH_MODELS_BASE/text_encoders/qwen_3_06b_base.safetensors     # Qwen3 0.6B text encoder
$PATH_MODELS_BASE/vae/qwen_image_vae.safetensors                # Qwen-Image VAE
```

---

## 2. 설치 (clone 직후 1회)

```bash
# 1) 서브모듈(sd-scripts)까지 함께 클론
git clone --recurse-submodules <this-repo-url>
cd anima-lora-trainer
# 이미 --recurse-submodules 없이 받았다면:
git submodule update --init --recursive

# 2) 가상환경 + 전체 의존성 설치 (.venv 생성)
#    태깅 스택(onnxruntime/pandas/pillow) + 학습 스택(torch-cu124/accelerate/diffusers 등)
#    + sd-scripts(editable) 를 한 번에 설치한다.
uv sync
```

`uv sync` 가 끝나면 프로젝트 루트에 `.venv/` 가 생기고, `3_run_training.sh` 는 이 `.venv` 를
그대로 사용한다. 설치 확인:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 예: 2.5.1+cu124 True
```

> **다른 CUDA 버전을 쓴다면** `pyproject.toml` 의 `[[tool.uv.index]]` url(`.../whl/cu124`)을
> 본인 환경에 맞게 바꾼 뒤(`cu121`, `cu118` 등) `uv sync` 를 다시 실행한다.

---

## 3. 설정 (`config_model.env` + `config.<trigger>.env`)

설정 파일은 `.gitignore` 되어 있고, 저장소에는 `*.example` 템플릿만 들어 있다. 복사해서 채운다:

```bash
cp config_model.env.example   config_model.env    # 모델 경로 (모든 스크립트 공통)
cp config.trigger.env.example config.mychar.env   # 트리거별 하이퍼파라미터 (config.<trigger>.env)
```

- **학습**(`3_run_training*.sh <trigger>`)은 `config_model.env`(경로) + `config.<trigger>.env`(하이퍼파라미터)를 읽는다.
- **스윕**(`4_sweep_grid.py`)은 `config_model.env`(경로)만 읽는다. rank·에폭 등은 출력 폴더에서 자동 추출한다.

`config_model.env` 예시 (모델 경로. 큰따옴표 안 `~` 는 확장 안 되므로 `$HOME` 사용):

```bash
PATH_MODELS_BASE="$HOME/workspace/ComfyUI/models"
PATH_MODEL="$PATH_MODELS_BASE/diffusion_models/anima_baseV10.safetensors"
PATH_QWEN="$PATH_MODELS_BASE/text_encoders/qwen_3_06b_base.safetensors"
PATH_VAE="$PATH_MODELS_BASE/vae/qwen_image_vae.safetensors"
```

`config.<trigger>.env` 예시 (LoRA 하이퍼파라미터. `BATCH_SIZE` 는 글로벌 배치):

```bash
NETWORK_DIM=32
NETWORK_ALPHA=32
BATCH_SIZE=4
GRADIENT_ACC_STEPS=1
LR="2e-5"
LR_SCHED="cosine"
LR_WARMUP_STEPS=100
EPOCHS=24
SAVE_EVERY_N_EPOCHS=2
GRADIENT_CHECKPOINTING=1
```

---

## 4. 파이프라인 실행

아래는 데이터셋 이름이 `mychar` 인 예시다. `<name>` 은 `dataset/<name>/` 폴더명이자
LoRA 출력 이름의 기준이 된다.

### 0단계 — 데이터셋 준비 (`0_dedup_raw.py`)

이미 정리된 평면 이미지라면 `dataset/<name>/` 에 그대로 넣으면 된다.

```
dataset/
└── mychar/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

**원시(중첩) 데이터셋이라면 → `0_dedup_raw.py`.** 앨범/출처별 폴더가 섞여 있거나 같은 그림의
변형(차분)이 많은 `dataset_raw/<원시>/` 를, DINOv2-base 의 CLS 임베딩으로 웹 UI 에서
그룹핑/제외/강조한 뒤 `dataset/<name>/repeat_<N>/` 심볼릭 링크 데이터셋까지 **한 번에
익스포트**한다. 같은 그림의 변형을 하나로 묶지 않으면 그 그림만 K배 과대표집되는데, 그걸
이름 규칙이 아니라 이미지 내용(임베딩)으로 잡는다. **원본 파일은 절대 건드리지 않고**
(read-only), 현황은 `.dedup/<name>/state.json` 메타데이터로만 관리한다.

```bash
uv run python 0_dedup_raw.py                # → http://127.0.0.1:8765 (UI 에서 데이터셋 선택)
uv run python 0_dedup_raw.py mychar     # 미리 임베딩해서 요약을 찍고 그 데이터셋으로 시작
uv run python 0_dedup_raw.py mychar --print # 서버 없이 현황/후보만 출력
```

**워크플로우**: 처음엔 모든 이미지가 **개별 그룹(싱글턴)**. 임계값을 천천히 낮춰가며 뜨는
**후보 쌍**을 하나씩 확인한다 — 자동 병합은 하지 않는다(연쇄 병합 방지).

| 컨트롤 | 동작 |
|--------|------|
| **데이터셋** | `dataset_raw/` 목록에서 선택. 임베딩이 없으면 시작 버튼이 뜬다 |
| **임계값** | 낮출수록 후보 쌍이 많이 뜬다. 그룹은 **centroid** 로 대표되어 다시 후보가 된다 |
| **임베딩 재계산** | 캐시를 버리고 DINOv2 를 다시 돌린다(백그라운드, 진행률 표시) |

| 조작 | 결과 |
|------|------|
| **병합** | 후보 쌍(또는 찍어둔 두 그룹)을 한 그룹으로. 그룹 centroid = 멤버 평균(정규화) |
| **다르다** | 두 그룹은 별개라고 명시(cannot_link) → 임계값을 더 낮춰도 다시 후보로 안 뜬다 |
| **강조(2R)** | 그룹(또는 싱글턴) 단위로 반복수 2배 |
| **그룹 해체 / 분리** | 그룹을 싱글턴들로 풀거나, 멤버 한 장만 빼서 싱글턴으로 |
| **제외 / 복구** | 익스포트에서 빼거나 되돌린다(원본은 그대로, 메타데이터만) |

화면은 **후보 쌍 → 묶인 그룹 → 개별 이미지 → 제외됨** 순. 개별 이미지의 대표를 클릭해
찍어두고 다른 그룹의 대표를 하나 더 찍으면 상단에 병합/다르다 확인 바가 뜬다(수동 병합).

**익스포트**(상단 `익스포트…`): `--name`/`repeat`(R)/`rounding` 을 지정하면 미리보기(분포/충돌
검사) 후 `dataset/<name>/repeat_<N>/` 에 **상대 심볼릭 링크**를 만든다. 반복수 규칙은
표준 싱글턴 = R, 강조 = 2R, 그룹은 **전체 합이 R(강조 2R)** 이 되도록 각 멤버
`round(합/k)`(`ceil`/`floor`/`round`, 최소 1). 제외 이미지는 빠지고, 플래트닝(`/`→`__`) 후
파일명이 충돌하면 막는다.

- `--threshold`(기본 **0.85**, UI 에서도 조절): centroid 코사인 유사도. 차분 쌍은 대개 0.95↑,
  무관한 쌍은 0.72 부근이라 0.85 에서 잘 갈린다. 낮출수록 후보가 급증하니 위에서부터 훑는다.
- 임베딩은 `.dedup/<name>/embeddings.npz` 에 캐시되고(파일 mtime+size 로 무효화), 그룹핑은
  임베딩과 무관하므로 재계산이 없다. DINOv2 모델은 프로세스당 한 번만 올라간다.
- 상태(`state.json`)는 비-기본 그룹(멤버>1 또는 강조)·제외·cannot_link 만 저장하고, 나머지
  싱글턴은 로드 때 복원한다. 원본이 read-only 라 rel 경로가 안정적 키가 된다.
- **속도**: 임베딩의 실제 병목은 GPU forward(수백 ms)가 아니라 PIL 디코드/리사이즈다 → 로딩을
  스레드풀로 병렬화(항상 켜짐). `--workers`(기본 auto=min(16, CPU수))로 조절한다. 실측 246장
  54s(순차)→18s(16스레드). `--fast`(또는 `DINOV2_FAST=1`)는 bf16+fast 프로세서로 forward 를
  더 줄이지만 이미 짧아 이득은 미미하다. fast 임베딩은 값이 fp32 와 미세하게 달라(코사인 |Δ|
  ~2e-3, 후보 쌍 집합은 사실상 동일) 캐시를 `full`/`fast` 로 태깅해 **모드 전환 시 자동 무효화**한다
  — 수동 삭제 불필요.
- 기타: `--refresh`(임베딩 캐시 무시), `--batch-size`(기본 16, `--fast` 면 32), `--host/--port`, `--open`.
- 익스포트 결과(`dataset/<name>/repeat_<N>/`)는 이후 단계가 그대로 소비한다. 태깅은 하위 폴더까지:
  `1_tag_dataset.py <name> --trigger @<name> --recursive`. `2_make_config.py` 는 repeat_<N> 마다
  `num_repeats=N` 서브셋을 찍는다.

### 1단계 — 태깅 (`1_tag_dataset.py`)

WD14 태거로 각 이미지를 태깅하고, `blacklist.txt` 의 태그를 제거한 뒤, 맨 앞에 trigger 를 붙여
`<이미지>.txt` 를 생성한다. (첫 실행 시 WD 모델을 HuggingFace 에서 자동 다운로드)

```bash
uv run python 1_tag_dataset.py mychar --trigger @mychar
```

주요 옵션:

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--trigger @mychar` | (필수) | 캡션 맨 앞 고정 토큰. `keep_tokens=1` 과 맞물려 셔플에서 고정된다. |
| `--general-threshold` | 0.35 | general 태그 임계값 |
| `--character-threshold` | 0.85 | character 태그 임계값 |
| `--include-rating` | off | rating 태그(general/sensitive…)를 trigger 다음에 삽입 (이 경우 toml 의 `keep_tokens=2` 로) |
| `--overwrite` | off | 기존 `.txt` 덮어쓰기 |
| `--dry-run` | off | 파일을 쓰지 않고 결과만 출력 |

생성 결과 예 (`@mychar` 이 항상 첫 토큰):

```
@mychar, 1girl, swimsuit, red hair, outdoors, smile, ...
```

`blacklist.txt` 는 스타일/캐릭터 학습에 방해되는 태그(메타/소스 노이즈 등)를 한 줄에 하나씩
적는다. `#` 주석/빈 줄은 무시되며, `_` 는 공백으로 정규화되어 매칭된다.

### 2단계 — 데이터셋 config 생성 (`2_make_config.py`)

sd-scripts 가 읽는 `dataset/<name>.toml` 을 생성한다. (ML 의존성 없이 동작)

```bash
uv run python 2_make_config.py mychar
# → dataset/mychar.toml 생성, image_dir = "../dataset/mychar"
```

주요 옵션: `--resolution 1024`(또는 `1024,768`), `--num-repeats 2`, `--batch-size 2`,
`--keep-tokens 1`, `--min-bucket-reso 512`, `--max-bucket-reso 1536`,
`--no-shuffle-caption`, `--no-bucket`, `--force`(덮어쓰기).

> 캡션(`.txt`)이 아직 없으면 경고를 출력한다. 1단계를 먼저 돌렸는지 확인할 것.

### 3단계 — 학습 (`3_run_training.sh`)

`config_model.env`(경로) + `config.<trigger>.env`(하이퍼파라미터) + `dataset/<name>.toml` 을 사용해 `accelerate launch` 로 LoRA 를 학습한다.

```bash
./3_run_training.sh mychar
```

- 결과: `output/mychar/anima_mychar_r32.safetensors` (`r32` 은 `NETWORK_DIM`)
- **자동 스킵**: 최종 `.safetensors` 가 이미 있으면 건너뛴다.
- **자동 resume**: `output/mychar/*-state` 가 있으면 가장 최신 state 에서 이어서 학습한다.
- 2 에폭마다 중간 체크포인트와 state 를 저장한다(`--save_every_n_epochs=2`).

참고 — `mychar`(227장, repeats 2 → 454, batch 2)의 경우 총 24 에폭 = 5448 스텝,
RTX 3090 기준 약 3.3s/it (전체 ~5시간).

#### 멀티 GPU (`3_run_training_2gpu.sh` / `3_run_training_4gpu.sh`)

`config.<trigger>.env` 의 `BATCH_SIZE` 는 **글로벌 배치**다. 스크립트가 이걸 GPU 수로 나눠
`--train_batch_size`(= GPU당 배치)로 넘긴다. **toml 은 손댈 필요 없다** — 같은 toml 로 셋 다 돈다.

```bash
./3_run_training.sh      mychar   # 1 GPU × 4 = 글로벌 4
./3_run_training_2gpu.sh mychar   # 2 GPU × 2 = 글로벌 4
./3_run_training_4gpu.sh mychar   # 4 GPU × 1 = 글로벌 4
```

그래서 `2_make_config.py` 는 **`batch_size` 를 toml 에 적지 않는 게 기본**이다. 배치는
GPU 수에 딸린 실행 시점 설정이지 데이터셋 속성이 아니기 때문이다.

> **toml 에 `batch_size` 를 적으면 그게 `--train_batch_size` 를 이긴다.**
> sd-scripts 는 `--dataset_config` 가 있으면 toml 값을 우선한다
> (`library/config_util.py` 의 `search_value`: `[dataset, general, argparse]` 순 첫 non-None).
> 그때 `--train_batch_size` 는 **로그와 LoRA 메타데이터에만** 반영되므로
> (`train_network.py` 의 `total_batch_size` / `ss_batch_size_per_device`),
> toml 이 `batch_size = 4` 인데 4-GPU 로 돌리면 실제로는 **글로벌 16** 이면서
> 콘솔엔 `total batch size: 4` 로 찍힌다 — 조용히 틀린다.
> 세 스크립트는 toml 에 `batch_size` 가 있으면 `× GPU수 == BATCH_SIZE` 인지 검증하고
> 어긋나면 실행을 거부한다. (`--batch-size N` 으로 일부러 박은 경우에만 해당)

출력 디렉토리는 `gb<글로벌배치>` 규칙이라 GPU 수가 달라도 같은 곳을 쓴다 → resume 호환.

### 4단계 — 에폭 그리드 스윕 (`4_sweep_grid.py`)

학습이 끝나면 `output/<run>/`(예: `output/mychar_gb4_lr4e-5_ep18`)에 에폭별 체크포인트가
쌓인다. 이 스크립트는 **출력 폴더 자체에서 trigger·rank·에폭을 자동 추출**해(심링크·별도
config 불필요), 최종 포함 N개 에폭을 골라 같은 프롬프트로 생성하고 "행=에폭 × 열=프롬프트"
그리드 PNG 로 묶는다. 에폭들은 **시스템의 모든 GPU 에 분산돼 병렬로** 돈다.

```bash
# 출력 폴더 직접 지정 (또는 트리거만 줘도 output/ 에서 유일하면 자동으로 찾는다)
uv run python 4_sweep_grid.py mychar_gb4_lr4e-5_ep18

# in-distribution: 데이터셋 캡션에서 6개 프롬프트 샘플링, 후반 5개 에폭
uv run python 4_sweep_grid.py mychar --mode tags --num-prompts 6 --num-epochs 5

# 특정 GPU 만 / 에폭 직접 지정 / 생성 건너뛰고 그리드만 재합성
uv run python 4_sweep_grid.py mychar --gpus 0,1 --epochs 14,16,18 --grid-only
```

- 결과: `output/<run>/sweep_<mode>/grid_<trigger>.png` (+ 에폭별 이미지 `ep<NN>/`, 에폭별 로그 `ep<NN>.log`)
- **인자**: 출력 폴더 이름/경로, 또는 트리거(= `output/<트리거>*` 유일 매칭이면 자동 해석).
- **프롬프트 소스 2종**:
  - `--mode samples`(기본): `samples.txt` 의 줄을 컬럼으로. `<trigger_placeholder>` 는
    `--trigger`(기본 `@<자동추출>`)로 치환. 줄 끝 `--d/--s/--g/--w/--h/--n` 으로 컬럼별 오버라이드.
  - `--mode tags`: `dataset/<trigger>/` 캡션에서 샘플링(= **in-distribution** 프롬프트).
    `--num-tags`(기본 10)개 태그, `--rng-seed` 로 재현 가능.
- **에폭 선택**: 기본은 **최종 포함 뒤에서 `--num-epochs`(기본 4)개**. `--epochs 12,16,20,24` 로 직접 지정.
- **병렬**: `--gpus`(기본 시스템 전체, 예 `0,1,2`, CPU 는 `cpu`). 한 에폭 = 한 서브프로세스가
  한 GPU 를 점유하고, 끝나면 대기 중인 다음 에폭이 그 GPU 를 이어받는다. 한 에폭이 실패해도
  나머지는 계속 돌고 그리드의 해당 행만 `missing` 으로 표시된다.
- 기타: `--steps 24 --guidance 3.5 --width/--height --multiplier 1.0`,
  `--dry-run`(계획만), `--grid-only`(그리드만 재합성), `--out`(출력 폴더).

---

## 5. 레포 구조

```
app.py                WD14Tagger (ONNX 태깅 로직). 1_tag_dataset.py 가 import.
dedup.py              DINOv2 임베딩 + centroid 그룹핑 상태/익스포트. 0_dedup_raw.py 가 import.
0_dedup_raw.py        [0] 웹 UI 그룹핑/제외/강조 → dataset/<name>/repeat_<N>/ 심볼릭 링크 익스포트
dataset_raw/          원시(중첩) 데이터셋 (read-only) + README.md
.dedup/<name>/        0a 캐시: DINOv2 임베딩 / 썸네일 / state.json (그룹핑 현황) (gitignore)
1_tag_dataset.py      [1] 태깅 → blacklist → trigger → *.txt (--recursive)
2_make_config.py      [2] dataset/<name>.toml 생성
3_run_training.sh     [3] accelerate 로 sd-scripts 학습 실행 (1 GPU)
3_run_training_2gpu.sh  [3] 같은 글로벌 배치를 2 GPU 로 분할 (BATCH_SIZE/2 씩)
3_run_training_4gpu.sh  [3] 같은 글로벌 배치를 4 GPU 로 분할 (BATCH_SIZE/4 씩)
4_sweep_grid.py       [4] 에폭별 LoRA 적용 → 그리드 스윕 PNG (다중 GPU 병렬)
samples.txt           [4] samples 모드 프롬프트 목록 (열 = 한 줄)
config_model.env      모델 경로 (모든 스크립트 공통; *.example 템플릿 제공, 실제는 gitignore)
config.<trigger>.env  트리거별 LoRA 하이퍼파라미터 (gitignore)
blacklist.txt         제외할 WD 태그 목록 (스타일/캐릭터 LoRA 용)
dataset/<name>/       학습 이미지 + 생성된 *.txt 캡션
dataset/<name>.toml   생성된 sd-scripts dataset_config
sd-scripts/           kohya-ss/sd-scripts 서브모듈 (anima/qwen 지원)
output/<name>/        학습 산출물 (LoRA, 중간 체크포인트, state)
pyproject.toml        uv 프로젝트 정의 (태깅 + 학습 의존성 일체)
```

`dataset/`, `output/`, `.venv/` 등 대용량 산출물은 `.gitignore` 로 제외된다.

---

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------|
| `No module named 'numpy'` 등 | `uv sync` 를 안 했거나 `uv run` 없이 시스템 파이썬으로 실행. `uv run python ...` 또는 `.venv/bin/python ...` 사용. |
| `anima_train_network.py: No such file` / 학습 모듈 없음 | 서브모듈 미초기화. `git submodule update --init --recursive`. |
| 모델 파일 못 찾음 (`~/workspace/...` 리터럴) | `config_model.env` 에서 `~` 대신 `$HOME` 사용 (큰따옴표 안 `~` 는 확장되지 않음 — 기본값은 이미 `$HOME`). |
| `torch.cuda.is_available()` 가 False | GPU/드라이버 미인식이거나 CUDA 버전 불일치. `pyproject.toml` 의 cu124 인덱스를 본인 CUDA 에 맞추고 `uv sync` 재실행. |
| 첫 태깅이 느림 / 멈춘 듯 | 첫 실행 시 WD 모델을 HuggingFace 에서 다운로드한다. 이후엔 캐시 사용. |
| 학습 로그가 파일로 리다이렉트하면 갱신이 느림 | 파일 출력은 블록 버퍼링된다. 실시간 확인은 터미널에서 직접 실행하거나 `PYTHONUNBUFFERED=1`. |

---

[kohya-ss/sd-scripts]: https://github.com/kohya-ss/sd-scripts
