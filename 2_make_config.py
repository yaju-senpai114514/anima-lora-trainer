#!/usr/bin/env python3
"""dataset/<name>.toml (kohya sd-scripts dataset_config) 를 생성하는 파이프라인 스크립트.

1_tag_dataset.py 로 캡션(.txt)을 만든 뒤, 이 스크립트로 학습 설정 toml 을 찍어내고,
3_run_training.sh <name> 으로 학습을 돌리는 흐름이다.

image_dir 은 학습이 sd-scripts/ 에서 실행되는 것을 기준으로 "../dataset/<name>" 상대경로로
기록한다(기존 trigger_tag.toml 과 동일한 규약).

사용 예:
    # dataset/mychar 을 가리키는 dataset/mychar.toml 생성
    .venv/bin/python 2_make_config.py mychar

    # 해상도/리피트/배치 조정 + 기존 toml 덮어쓰기
    .venv/bin/python 2_make_config.py mychar --resolution 1024 --num-repeats 4 --batch-size 2 --force
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# app.py 와 동일한 경로/확장자 규약. 이 스텝은 toml 만 찍어내므로 ML 스택(numpy/onnx)을
# 끌어오지 않도록 app 을 import 하지 않고 상수만 로컬에 둔다.
ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def parse_resolution(value: str) -> list[int]:
    """'1024' -> [1024, 1024], '1024,768' -> [1024, 768]."""
    parts = [p.strip() for p in value.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        return [n, n]
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    raise argparse.ArgumentTypeError(f"invalid resolution: {value!r}")


def count_images(folder: Path) -> int:
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def render_toml(
    *,
    subsets: list[tuple[str, int]],
    resolution: list[int],
    batch_size: int | None,
    keep_tokens: int,
    caption_extension: str,
    shuffle_caption: bool,
    enable_bucket: bool,
    bucket_no_upscale: bool,
    min_bucket_reso: int,
    max_bucket_reso: int,
) -> str:
    """subsets: (image_dir, num_repeats) 목록. 플래트닝 전엔 1개, repeat_<N> 구조면 여러 개.

    batch_size=None 이면 toml 에 적지 않는다(기본). 그러면 학습 스크립트가 넘기는
    --train_batch_size 가 GPU당 배치가 되어, 같은 toml 로 1/2/4 GPU 를 다 돌릴 수 있다.
    toml 에 batch_size 를 적으면 sd-scripts 가 그걸 **우선**해 --train_batch_size 를
    무시하므로(library/config_util.py 의 search_value), GPU 수를 바꿀 때마다 toml 을
    다시 만들어야 한다. 그래서 기본은 안 적는 쪽이다.
    """
    res = f"[{resolution[0]}, {resolution[1]}]"
    body = [
        "[general]\n",
        f'caption_extension = "{caption_extension}"\n',
        f"shuffle_caption = {str(shuffle_caption).lower()}\n",
        f"keep_tokens = {keep_tokens}\n",
        "\n",
        "[[datasets]]\n",
        f"resolution = {res}\n",
    ]
    if batch_size is not None:
        body.append(f"batch_size = {batch_size}\n")
    body += [
        f"enable_bucket = {str(enable_bucket).lower()}\n",
        f"bucket_no_upscale = {str(bucket_no_upscale).lower()}\n",
        f"min_bucket_reso = {min_bucket_reso}\n",
        f"max_bucket_reso = {max_bucket_reso}\n",
    ]
    for image_dir, num_repeats in subsets:
        body += [
            "\n",
            "  [[datasets.subsets]]\n",
            f'  image_dir = "{image_dir}"\n',
            f"  num_repeats = {num_repeats}\n",
        ]
    return "".join(body)


def collect_subsets(data_dir: Path, name: str, default_repeats: int) -> tuple[list[tuple[str, int]], int]:
    """repeat_<N> 서브셋 구조면 폴더별 (image_dir, N) 목록을, 아니면 단일 서브셋을 반환.

    returns: (subsets, 총 이미지 수)
    """
    repeat_dirs = sorted(
        (p for p in data_dir.iterdir()
         if p.is_dir() and re.fullmatch(r"repeat_\d+", p.name)),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if repeat_dirs:
        subsets, total = [], 0
        for p in repeat_dirs:
            total += count_images(p)
            subsets.append((f"../dataset/{name}/{p.name}", int(p.name.split("_")[1])))
        return subsets, total
    return [(f"../dataset/{name}", default_repeats)], count_images(data_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="dataset/<name>.toml 생성")
    ap.add_argument("name", help="dataset/<name> 의 데이터셋 이름 (예: mychar)")
    ap.add_argument("--resolution", type=parse_resolution, default=parse_resolution("1024"),
                    help="해상도. '1024' 또는 '1024,768' (기본: 1024)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="toml 에 batch_size 를 **박아넣는다**. 기본은 안 박는 것 — 그러면 GPU당 배치를 "
                         "학습 스크립트의 --train_batch_size(= BATCH_SIZE/GPU수)가 정하므로 같은 toml 로 "
                         "1/2/4 GPU 를 다 돌릴 수 있다. 박으면 sd-scripts 가 그걸 우선해 "
                         "--train_batch_size 를 무시하니, GPU 수를 바꿀 때마다 다시 만들어야 한다.")
    ap.add_argument("--num-repeats", type=int, default=2)
    ap.add_argument("--keep-tokens", type=int, default=1,
                    help="셔플에서 고정할 앞쪽 토큰 수 (trigger 1개 → 1)")
    ap.add_argument("--caption-extension", default=".txt")
    ap.add_argument("--no-shuffle-caption", action="store_true",
                    help="caption 셔플 비활성화")
    ap.add_argument("--no-bucket", action="store_true",
                    help="aspect ratio bucketing 비활성화")
    ap.add_argument("--bucket-no-upscale", action="store_true",
                    help="bucket 시 업스케일 금지")
    ap.add_argument("--min-bucket-reso", type=int, default=512)
    ap.add_argument("--max-bucket-reso", type=int, default=1536)
    ap.add_argument("--output", default=None,
                    help="출력 toml 경로 (기본: dataset/<name>.toml)")
    ap.add_argument("--force", action="store_true", help="기존 toml 덮어쓰기")
    args = ap.parse_args()

    data_dir = (IMAGE_ROOT / args.name).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"[error] dataset dir not found: {data_dir}")

    subsets, n_img = collect_subsets(data_dir, args.name, args.num_repeats)
    if n_img == 0:
        raise SystemExit(f"[error] no images in {data_dir}")

    # repeat_<N> 구조면 하위까지, 아니면 top-level 캡션 수를 센다.
    multi = len(subsets) > 1 or subsets[0][0] != f"../dataset/{args.name}"
    caption_glob = data_dir.rglob(f"*{args.caption_extension}") if multi else data_dir.glob(f"*{args.caption_extension}")
    n_txt = sum(1 for _ in caption_glob)
    if n_txt == 0:
        print(f"[warn] {data_dir} 에 {args.caption_extension} 캡션이 없다. "
              f"먼저 1_tag_dataset.py {args.name} 을 돌렸는지 확인하라.")

    out_path = Path(args.output).resolve() if args.output else (IMAGE_ROOT / f"{args.name}.toml")
    if out_path.exists() and not args.force:
        raise SystemExit(f"[error] already exists (use --force): {out_path}")

    toml = render_toml(
        subsets=subsets,
        resolution=args.resolution,
        batch_size=args.batch_size,
        keep_tokens=args.keep_tokens,
        caption_extension=args.caption_extension,
        shuffle_caption=not args.no_shuffle_caption,
        enable_bucket=not args.no_bucket,
        bucket_no_upscale=args.bucket_no_upscale,
        min_bucket_reso=args.min_bucket_reso,
        max_bucket_reso=args.max_bucket_reso,
    )
    out_path.write_text(toml, encoding="utf-8")

    print(f"[done] wrote {out_path}")
    if len(subsets) > 1:
        print(f"  images={n_img}  captions={n_txt}  subsets={len(subsets)} "
              f"(repeat: {', '.join(str(r) for _d, r in subsets)})  resolution={args.resolution}")
    else:
        bs_note = "toml 미기재 (학습 스크립트의 --train_batch_size 가 결정)" \
            if args.batch_size is None else f"{args.batch_size} (toml 에 고정 → GPU 수 바꾸면 재생성 필요)"
        print(f"  images={n_img}  captions={n_txt}  resolution={args.resolution}  "
              f"repeats={subsets[0][1]}  batch_size: {bs_note}")
    # 반복수 반영 총 학습이미지 + 배치별 에폭당 스텝수 (0_dedup_raw 와 동일 규약)
    # 총 스텝 = 이 값 × epochs. enable_bucket 시 버킷 반올림으로 실제 스텝은 조금 더 클 수 있음.
    def _subset_dir(image_dir: str) -> Path:
        tail = Path(image_dir).name
        return data_dir if tail == args.name else data_dir / tail
    total_reps = sum(count_images(_subset_dir(d)) * r for d, r in subsets)
    steps_line = "  ".join(f"b{b}:{math.ceil(total_reps / b)}" for b in (1, 2, 4, 8, 16))
    print(f"  에폭당 학습이미지(반복반영) ≈ {total_reps}  |  스텝수(batch별, ceil): {steps_line}")
    print(f"  next: ./3_run_training.sh {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
