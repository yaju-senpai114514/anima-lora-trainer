`anime-base-v1.0`은 검색상으로는 **Anima-Base v1.0**을 말하는 것으로 보입니다. Anima는 SDXL U-Net 계열이 아니라 **2B DiT / Rectified Flow 계열**이고, Qwen3-0.6B text encoder, LLM Adapter, Qwen-Image VAE를 쓰므로 일반 `train_network.py`가 아니라 `sd-scripts`의 **`anima_train_network.py`**를 쓰는 쪽이 정석입니다. ([GitHub][1])

## 핵심 요약

Anima LoRA는 “가볍게” 학습하는 쪽이 좋습니다. 공식 모델 카드도 **LLM Adapter는 학습하지 말고**, rank 32 LoRA 기준 **LR 2e-5부터 시작**하라고 권장합니다. 이유는 Anima-Base 자체가 이미 꽤 넓은 시각 개념을 갖고 있고, 강한 aesthetic tuning을 이겨내야 하는 모델이 아니기 때문입니다. ([Hugging Face][2])

일반 시작점은 다음 정도입니다.

| 목적         |   이미지 수 |     rank/alpha |        LR | 총 step 감각 | 메모                                        |
| ---------- | ------: | -------------: | --------: | --------: | ----------------------------------------- |
| 캐릭터 LoRA   |  20–80장 | 16/16 또는 32/32 | 1e-5–3e-5 |     1k–3k | 과적합이 빠름. 포즈/표정/구도 다양성 중요                  |
| 의상/소품 LoRA |  20–60장 | 16/16 또는 32/32 | 1e-5–3e-5 |   1k–2.5k | 캐릭터보다 의상 태그 정리가 중요                        |
| 스타일 LoRA   | 80–300장 | 32/32 또는 64/64 | 1e-5–2e-5 |   2.5k–6k | style bleed 방지를 위해 subject captioning 철저히 |
| 컨셉/구도 LoRA | 50–200장 |          16–32 | 1e-5–3e-5 |   1.5k–4k | 배경/구도/소품을 caption에 명시                     |

커뮤니티 레시피에서도 Anima LoRA는 대략 **2500 step 근방**을 기준으로 repeats를 맞추고, 의상은 rank 16/32, 스타일은 rank 32/64 정도를 제안하는 사례가 있습니다. 다만 이건 경험적 시작점이고 데이터셋 품질에 더 민감합니다. ([note（ノート）][3])

## 1. 필요한 파일

공식 Anima 구조 기준으로 최소 3개가 필요합니다.

```text
models/
  anima-base-v1.0.safetensors          # DiT / diffusion model
  qwen_3_06b_base.safetensors          # text encoder
  qwen_image_vae.safetensors           # VAE
```

ComfyUI 기준 배치도는 각각 `diffusion_models`, `text_encoders`, `vae` 폴더로 안내되어 있습니다. ([Hugging Face][2])
학습 스크립트 쪽에서는 `--pretrained_model_name_or_path`, `--qwen3`, `--vae`를 명시합니다. ([GitHub][1])

## 2. 데이터셋 준비

권장 해상도는 일단 **1024×1024 bucket** 기준으로 잡는 것이 무난합니다. Anima 공식 생성 설정은 512²–1536² 범위를 지원한다고 되어 있고, 1024는 LoRA 학습/검증에서 현실적인 기본값입니다. ([Hugging Face][2])

폴더 구조 예:

```text
dataset/
  my_char/
    0001.png
    0001.txt
    0002.png
    0002.txt
```

캐릭터 LoRA caption 예:

```text
mychar_token, 1girl, white hair, red eyes, witch hat, black dress, solo, looking at viewer, upper body, simple background
```

스타일 LoRA caption 예:

```text
mystyle_token, 1girl, school uniform, standing, city background, soft lighting, detailed eyes
```

caption 원칙은 간단합니다.

고정하고 싶은 대상은 trigger token에 묶고, **묶고 싶지 않은 요소는 caption에 적습니다.** 예를 들어 특정 캐릭터 LoRA에서 “항상 흰 배경만 나오는 문제”를 피하려면 `white background`, `simple background`, `outdoors`, `bedroom` 같은 배경 정보를 caption에 제대로 넣어야 합니다.

Anima는 Danbooru-style tag와 자연어 caption을 모두 학습한 모델이고, 태그는 소문자와 공백 사용이 권장됩니다. score tag만 underscore를 씁니다. 기본 positive prefix는 `masterpiece, best quality, score_7, safe,`이고 negative는 `worst quality, low quality, score_1, score_2, score_3, artist name` 쪽이 권장됩니다. ([Hugging Face][2])

## 3. dataset config 예시

`anima_dataset.toml`:

```toml
[general]
caption_extension = ".txt"
shuffle_caption = false
keep_tokens = 1

[[datasets]]
resolution = [1024, 1024]
batch_size = 2
enable_bucket = true
bucket_no_upscale = false
min_bucket_reso = 512
max_bucket_reso = 1536

  [[datasets.subsets]]
  image_dir = "/data/dataset/my_char"
  num_repeats = 10
```

대략 step 계산은:

```text
steps_per_epoch = ceil(num_images * repeats / batch_size / grad_accum)
total_steps = steps_per_epoch * epochs
```

예를 들어 40장, repeats 10, batch 2, grad accum 2, epoch 10이면:

```text
ceil(40 * 10 / 2 / 2) * 10 = 1000 steps
```

캐릭터는 1000–2500 step부터 보고, 스타일은 2500–6000 step 정도까지 늘려보는 식이 무난합니다.

## 4. 학습 커맨드 시작점

24GB VRAM 기준으로 보수적인 레시피입니다.

```bash
accelerate launch --num_cpu_threads_per_process 2 ./anima_train_network.py \
  --pretrained_model_name_or_path="/models/anima-base-v1.0.safetensors" \
  --qwen3="/models/qwen_3_06b_base.safetensors" \
  --vae="/models/qwen_image_vae.safetensors" \
  --dataset_config="/data/anima_dataset.toml" \
  --output_dir="/data/output" \
  --output_name="my_char_anima_lora_r32" \
  --save_model_as="safetensors" \
  --network_module="networks.lora_anima" \
  --network_train_unet_only \
  --network_dim=32 \
  --network_alpha=32 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --optimizer_type="AdamW8bit" \
  --learning_rate=2e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --caption_extension=".txt" \
  --timestep_sampling="sigmoid" \
  --console_log_simple
```

`sd-scripts` 공식 예제는 `--network_module=networks.lora_anima`, `--timestep_sampling=sigmoid`, `--mixed_precision=bf16`, `--gradient_checkpointing`, `--cache_latents`, `--cache_text_encoder_outputs` 같은 구성을 보여줍니다. 또한 Anima 전용 인자로 `--qwen3`, `--vae`가 필요하고, SD v1/v2용 `--clip_skip`, `--v2`, `--v_parameterization`, `--fp8_base`는 맞지 않습니다. ([GitHub][1])

최신 `sd-scripts`라면 latent caching 단계에서 다음을 추가하는 것도 좋습니다.

```bash
  --qwen_image_vae_2d
```

공식 문서상 이 옵션은 단일 이미지 기준 3D VAE와 수치적으로 동등한 2D 변환을 쓰며, 예시 기준 RTX 3090에서 VAE encode/decode peak VRAM과 시간이 줄어드는 것으로 설명되어 있습니다. ([GitHub][1])

## 5. 모듈 선택 레시피

기본적으로 Anima LoRA는 DiT block 내부의 self-attention, cross-attention, MLP 쪽에 붙고, modulation/norm/embedder/final layer는 제외됩니다. `network_reg_dims`, `network_reg_lrs`, `include_patterns`, `exclude_patterns`로 regex 기반 제어가 가능합니다. ([GitHub][1])

보수적 캐릭터 LoRA:

```bash
--network_dim=16 \
--network_alpha=16 \
--learning_rate=2e-5
```

표현력 조금 더 필요한 캐릭터/의상:

```bash
--network_dim=32 \
--network_alpha=32 \
--learning_rate=2e-5
```

스타일 LoRA:

```bash
--network_dim=64 \
--network_alpha=64 \
--learning_rate=1e-5
```

self-attn은 높게, cross-attn은 낮게 주고 싶으면:

```bash
--network_args "network_reg_dims=.*self_attn.*=32,.*cross_attn.*=16,.*mlp.*=32" \
--network_args "network_reg_lrs=.*self_attn.*=2e-5,.*cross_attn.*=1e-5,.*mlp.*=2e-5"
```

LLM Adapter는 기본적으로 건드리지 않는 편이 좋습니다. 공식 모델 카드도 LLM Adapter가 text embedding을 diffusion model로 넘기기 전에 처리하는 위치라 영향이 크고 쉽게 망가질 수 있다고 경고합니다. ([Hugging Face][2])

## 6. 검증 프롬프트

매 epoch 저장 후 같은 seed/grid로 비교합니다. Anima 공식 생성 설정은 대략 **30–50 steps, CFG 4–5**가 기본 범위입니다. ([Hugging Face][2])

캐릭터 LoRA 검증:

```text
masterpiece, best quality, score_7, safe, mychar_token, 1girl, solo, standing, full body, white background
```

다양성 검증:

```text
masterpiece, best quality, score_7, safe, mychar_token, 1girl, cafe, sitting, smile, casual clothes
```

negative:

```text
worst quality, low quality, score_1, score_2, score_3, artist name, bad anatomy, extra fingers
```

LoRA weight는 `0.6 / 0.8 / 1.0 / 1.2`를 같이 봅니다. `1.0`에서만 겨우 나오면 underfit, `0.6`에서도 포즈/배경까지 훈련셋처럼 고정되면 overfit 쪽입니다.

## 7. 문제별 조정

**캐릭터가 잘 안 나옴**
`steps`를 늘리거나 `rank 16 → 32`, repeats 증가. trigger token이 너무 일반적인 단어면 더 유니크하게 변경.

**얼굴은 나오는데 의상/장식이 빠짐**
caption에 빠진 장식 태그를 넣고, 해당 장식이 잘 보이는 이미지를 추가. 전신/상반신/클로즈업 비율을 섞기.

**훈련 이미지 포즈만 반복됨**
steps/repeats를 줄이고, caption에 포즈/배경/구도를 더 명시. 너무 비슷한 이미지는 제거.

**스타일 LoRA가 모든 subject를 같은 얼굴로 만듦**
caption에서 subject와 composition을 더 자세히 적고, rank/LR/steps를 낮춤. 스타일 데이터셋에 특정 캐릭터가 과다하면 샘플 균형을 맞춤.

**색감만 약하게 먹음**
style LoRA는 rank 32보다 64가 나을 때가 많음. 다만 LR은 1e-5 근방으로 낮게.

## 8. 라이선스 주의

Anima 모델 자체는 CircleStone Labs Non-Commercial License이고, 모델/derivative model의 상업적 사용에는 제한이 있습니다. 다만 공식 모델 카드는 생성 이미지 output의 상업적 사용은 허용된다고 설명합니다. LoRA를 공개/판매/서비스에 붙일 때는 모델 라이선스 조건을 별도로 확인해야 합니다. ([Hugging Face][2])

[1]: https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md "sd-scripts/docs/anima_train_network.md at main · kohya-ss/sd-scripts · GitHub"
[2]: https://huggingface.co/circlestone-labs/Anima "circlestone-labs/Anima · Hugging Face"
[3]: https://note.com/yumihari2025/n/n4dc165fdc3f2 "画像生成モデルAnimaのLoRA学習およびFinetuningコード｜yumihari"
