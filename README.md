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

`config.env` 의 `PATH_COMFYUI` 기준 경로에 다음 3개가 존재해야 한다(기본값은 `$HOME/workspace/ComfyUI`):

```
$PATH_COMFYUI/models/diffusion_models/anima_baseV10.safetensors   # Anima base (DiT)
$PATH_COMFYUI/models/text_encoders/qwen_3_06b_base.safetensors     # Qwen3 0.6B text encoder
$PATH_COMFYUI/models/vae/qwen_image_vae.safetensors                # Qwen-Image VAE
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

## 3. 설정 (`config.env`)

설정 파일은 `.gitignore` 되어 있고, 저장소에는 `*.example` 템플릿만 들어 있다. 복사해서 채운다:

```bash
cp config_model.env.example  config_model.env      # 모델 경로 (학습 스크립트 공통)
cp config.trigger.env.example config.mychar.env     # 트리거별 하이퍼파라미터 (config.<trigger>.env)
cp config.env.example        config.env            # 4_sweep_grid.py 전용 통합 설정
```

- **학습**(`3_run_training*.sh <trigger>`)은 `config_model.env`(경로) + `config.<trigger>.env`(하이퍼파라미터)를 읽는다.
- **스윕**(`4_sweep_grid.py`)은 `config.env` 하나를 읽는다.

아래는 `config.env` 의 예시다. 모델 경로와 LoRA 하이퍼파라미터를 한 파일에서 관리한다.

```bash
PATH_COMFYUI="$HOME/workspace/ComfyUI"   # ← 큰따옴표 안 ~ 는 확장 안 되므로 반드시 $HOME 사용
PATH_MODEL="$PATH_COMFYUI/models/diffusion_models/anima_baseV10.safetensors"
PATH_QWEN="$PATH_COMFYUI/models/text_encoders/qwen_3_06b_base.safetensors"
PATH_VAE="$PATH_COMFYUI/models/vae/qwen_image_vae.safetensors"

NETWORK_DIM=32          # LoRA rank
NETWORK_ALPHA=32
BATCH_SIZE=2
GRADIENT_ACC_STEPS=1
LR="2e-5"
LR_SCHED="cosine"
LR_WARMUP_STEPS=100
MAX_TRAIN_EPOCHS=24
```

---

## 4. 파이프라인 실행

아래는 데이터셋 이름이 `mychar` 인 예시다. `<name>` 은 `dataset/<name>/` 폴더명이자
LoRA 출력 이름의 기준이 된다.

### 0단계 — 이미지 배치

학습할 이미지를 `dataset/<name>/` 에 넣는다.

```
dataset/
└── mychar/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

> **원시(중첩) 데이터셋이 있다면 → `0_process_raw.py` 로 플래트닝.** 앨범/차분/강조가 폴더로
> 나뉜 `dataset_raw/<원시>/` 를 반복수별 서브셋(`dataset/<name>/repeat_<N>/`)으로 **심볼릭 링크**
> 플래트닝한다(자세한 규칙은 `dataset_raw/PROCESS_RAW.md`).
>
> ```bash
> uv run python 0_process_raw.py mystyle_raw --repeat 10        # 표준 반복 R=10
> # → dataset/mystyle/repeat_{5,7,10,20}/... (경로구분자 / → __ 로 플래트닝된 심볼릭 링크)
> ```
>
> - **분류/반복수**: 표준 = R, 강조(`_special_`, 원시의 `special/`·`*_special` 폴더) = 2R,
>   차분 그룹(`_sabun`/`_sabun_<>` 변형 + 원본) = 그룹 합이 R(강조면 2R)이 되도록 각 이미지 R/K.
> - 나눠떨어지지 않으면 `--rounding ceil|floor`(기본 ceil, 최소 1). 차분 그룹에 원본이 없으면 오류로 중단.
> - `--name`(출력 이름, 기본 `_raw` 제거), `--absolute`(절대경로 링크), `--force`, `--dry-run`.
> - 이후 태깅은 하위 폴더까지: `1_tag_dataset.py mystyle --trigger @mystyle --recursive`
>   (repeat_ 구조는 자동 재귀). `2_make_config.py` 는 repeat_<N> 마다 `num_repeats=N` 서브셋을 찍는다.

### 0a단계 — 차분 탐지 & 리네임 (`0a_dedup_raw.py`) — 선택

0단계의 차분(`_sabun`)/강조(`_special_`) 규칙은 **순전히 파일명에만 의존한다.** 같은 그림의
변형을 이름으로 묶어주지 않으면 그 그림만 K배 과대표집된다. 이 스텝은 DINOv2-base 의 CLS
임베딩으로 변형 후보를 찾아 웹 UI 에서 일괄 리네임한다. `0_process_raw.py` **앞에** 온다.

```bash
uv run python 0a_dedup_raw.py                    # → http://127.0.0.1:8765 (UI 에서 데이터셋 선택)
uv run python 0a_dedup_raw.py mychar         # 미리 임베딩해서 결과를 찍고 그 데이터셋으로 시작
uv run python 0a_dedup_raw.py mychar --print # 서버 없이 탐지 결과만 (이름 필수)
uv run python 0a_dedup_raw.py mychar --undo  # 마지막 적용 되돌리기 (이름 필수)
```

웹 UI(의존성 추가 없이 stdlib `http.server`)의 상단에서:

| 컨트롤 | 동작 |
|--------|------|
| **데이터셋** | `dataset_raw/` 목록에서 선택. 임베딩이 없으면 안내와 함께 시작 버튼이 뜬다 |
| **임계값** | 바꾸면 **즉시** 재그룹핑 (임베딩과 무관하므로 재계산 없음) |
| **임베딩** | 캐시를 버리고 DINOv2 를 다시 돌린다. 백그라운드로 돌며 진행률이 표시된다 |

그룹별로:

| 조작 | 결과 |
|------|------|
| **원본으로** | 그 이미지를 `<base>`, 나머지를 `<base>_sabun_1..N` 으로 (그룹 일괄 정리) |
| **유지** | 그 이미지는 손대지 않는다. **기본값** — 잘못 엮인 멤버를 빼는 용도 |
| **강조(2R)** | 이름에 `_special_` 마커를 넣어 반복수 2배로 |
| **제외** | `dataset_raw/<name>_excluded/` 로 이동 (UI 에서 복구 가능) |
| 이름 직접 편집 | 자유 입력. 규칙 위반은 적용 전에 막힌다 |

**이미 정리된 그룹은 접혀서 나온다.** 그룹 헤더의 뱃지와 타일 뱃지는 "무슨 역할을 줬나"가
아니라 **적용 후 0단계가 그 파일을 뭘로 볼지**를 보여준다. 그래서 이미 `_sabun` 으로 묶어 둔
그룹은 손대지 않아도 `원본`/`차분` 으로 보이고, 안 묶인 중복은 `표준`(= 각자 R 회 = 과대표집)
으로 드러난다. 뱃지는 편집하는 즉시 따라오므로 `이미 차분 그룹` 으로 바뀌는 게 곧 완료 신호다.

| 그룹 상태 | 뜻 |
|-----------|-----|
| **이미 차분 그룹** | 원본 1장 + 나머지 전부 `_sabun_N`. 할 일 없음 → **접힌 채로 시작** |
| **일부만 표기됨** | 차분 그룹에 아직 판단 안 한 이미지가 섞여 있다 |
| **미표기** | 전부 표준 취급 중 — 묶어야 할 후보 |
| **비규격 차분 이름** | `_sabun1` 류. 0단계가 못 알아본다 (아래 참고) |
| **원본 없음** | 차분만 있고 원본이 없다 → 이대로 두면 0단계가 멈춘다 |

상단 바에 `이미 차분 그룹 N개 · 확인 필요 M개` 가 뜨고 접기/펼치기를 일괄로 할 수 있다.
연결요소가 서로 다른 원본을 엮어놓은 경우(예: `_p0_l` + `_p0_r`) 무관한 멤버를 **유지**로
빼면 그 그룹도 완료로 넘어간다 — 그래야 `확인 필요` 카운터가 0 에 도달한다.

- **적용 전 전량 검증**: 이름 충돌, 원본 없는 차분 그룹(0단계가 중단하는 조건), 확장자 변경 등은
  `0_process_raw.py` 와 **같은 정규식**으로 서버가 검사해 오류면 적용을 막는다. 적용은 즉시
  반영되고 `.dedup/<name>/rename_log.json` 에 기록되어 `--undo` 로 되돌릴 수 있다.
- `--threshold`(기본 **0.85**, UI 에서도 조절): CLS 코사인 유사도. `mychar` 의 기존 `_sabun`
  표기를 정답으로 놓고 실측한 값이다 — 차분 쌍 중앙값 0.95 / 무관한 쌍 상위 5% 0.72 로 분포가
  거의 겹치지 않고, 0.85 에서 재현율 92.9%. 0.80 으로 낮추면 재현율은 그대로인데 오탐만 8배로 는다.
- 임베딩은 `.dedup/<name>/embeddings.npz` 에 캐시되고(파일 mtime+size 로 무효화), 리네임해도
  키만 갱신하므로 다시 계산하지 않는다. DINOv2 모델은 프로세스당 한 번만 올라가므로 데이터셋을
  오가도 재로딩이 없다.
- 기타: `--refresh`(임베딩 캐시 무시), `--batch-size`, `--host/--port`, `--open`.

> **`_sabun1` 은 차분이 아니다.** 0단계의 정규식은 `_sabun` 또는 `_sabun_<>` 만 인식한다.
> `_sabun1` 처럼 밑줄 없이 숫자가 붙으면 **독립 표준 이미지로 처리된다**(= 과대표집).
> 이 스텝이 그런 이름을 `규칙위반` 으로 표시하고 `_sabun_N` 으로 교정해 준다.

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

`config.env` 와 `dataset/<name>.toml` 을 사용해 `accelerate launch` 로 LoRA 를 학습한다.

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

학습이 끝나면 `output/<name>/` 에 에폭별 체크포인트
(`anima_<name>_r<dim>-NNNNNN.safetensors` + 최종 `anima_<name>_r<dim>.safetensors`)가 쌓인다.
이 스크립트는 그중 **최종 에폭을 포함해 N개(기본 4)** 를 골라, 같은 프롬프트 묶음을 각 에폭에
적용해 생성하고 "행=에폭 × 열=프롬프트" 그리드 PNG 로 묶어 어느 에폭이 좋은지 한눈에 비교한다.
(생성은 `sd-scripts/anima_minimal_inference.py` 배치 모드를 그대로 사용)

```bash
# samples.txt 의 프롬프트로, 최종 포함 4개 에폭 스윕
uv run python 4_sweep_grid.py mychar

# 데이터셋 캡션 태그에서 5개 프롬프트를 샘플링해 스윕
uv run python 4_sweep_grid.py mychar --mode tags --num-prompts 5
```

- 결과: `output/<name>/sweep4/grid_<name>.png` (+ 에폭별 이미지 `sweep4/ep<NN>/`)
- **프롬프트 소스 2종**:
  - `--mode samples`(기본): `samples.txt` 의 줄을 그대로 컬럼으로 사용. trigger 자리에는
    `<trigger_placeholder>` 를 적으면 `--trigger`(기본 `@<name>`)로 치환된다. 줄 끝에
    `--d/--s/--g/--w/--h/--n` 으로 컬럼별 오버라이드 가능(미지정 항목은 기본값/자동 시드).
  - `--mode tags`: `dataset/<name>/` 캡션에서 무작위로 캡션을 골라 trigger 고정 + 나머지 태그를
    `--num-tags`(기본 10)개 샘플링해 프롬프트를 만든다. `--rng-seed` 로 재현 가능.
- **에폭 선택**: 기본은 가용 에폭에서 **최종 포함 뒤에서 `--num-epochs`(기본 4)개**(예: `18,20,22,24`).
  `--epochs 12,16,20,24` 로 직접 지정 가능.
- 기타 옵션: `--steps 24 --guidance 3.5 --width/--height --multiplier 1.0 --device cuda`,
  `--dry-run`(계획/prompts.txt 만), `--grid-only`(생성 건너뛰고 그리드만 재합성), `--out`(출력 폴더).

---

## 5. 레포 구조

```
app.py                WD14Tagger (ONNX 태깅 로직). 1_tag_dataset.py 가 import.
dedup.py              DINOv2 임베딩 + _sabun 이름 규칙/검증. 0a_dedup_raw.py 가 import.
0a_dedup_raw.py       [0a] 차분/중복 탐지 → 웹 UI 리네임 (0단계 전, 선택)
0_process_raw.py      [0] dataset_raw/<원시> → dataset/<name>/repeat_<N>/ 심볼릭 플래트닝
dataset_raw/          원시(중첩) 데이터셋 + PROCESS_RAW.md (플래트닝 규칙)
.dedup/<name>/        0a 캐시: DINOv2 임베딩 / 썸네일 / undo 로그 (gitignore)
1_tag_dataset.py      [1] 태깅 → blacklist → trigger → *.txt (--recursive)
2_make_config.py      [2] dataset/<name>.toml 생성
3_run_training.sh     [3] accelerate 로 sd-scripts 학습 실행 (1 GPU)
3_run_training_2gpu.sh  [3] 같은 글로벌 배치를 2 GPU 로 분할 (BATCH_SIZE/2 씩)
3_run_training_4gpu.sh  [3] 같은 글로벌 배치를 4 GPU 로 분할 (BATCH_SIZE/4 씩)
4_sweep_grid.py       [4] 에폭별 LoRA 적용 → 그리드 스윕 PNG
samples.txt           [4] samples 모드 프롬프트 목록 (열 = 한 줄)
config.env            모델 경로 + LoRA 하이퍼파라미터
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
| 모델 파일 못 찾음 (`~/workspace/...` 리터럴) | `config.env` 에서 `~` 대신 `$HOME` 사용 (큰따옴표 안 `~` 는 확장되지 않음 — 기본값은 이미 `$HOME`). |
| `torch.cuda.is_available()` 가 False | GPU/드라이버 미인식이거나 CUDA 버전 불일치. `pyproject.toml` 의 cu124 인덱스를 본인 CUDA 에 맞추고 `uv sync` 재실행. |
| 첫 태깅이 느림 / 멈춘 듯 | 첫 실행 시 WD 모델을 HuggingFace 에서 다운로드한다. 이후엔 캐시 사용. |
| 학습 로그가 파일로 리다이렉트하면 갱신이 느림 | 파일 출력은 블록 버퍼링된다. 실시간 확인은 터미널에서 직접 실행하거나 `PYTHONUNBUFFERED=1`. |

---

[kohya-ss/sd-scripts]: https://github.com/kohya-ss/sd-scripts
