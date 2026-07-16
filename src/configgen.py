"""sd-scripts dataset_config(toml) 생성 로직 (공용, ML 의존성 없음).

웹 서버(src/webui/server.py)의 config 생성 엔드포인트가 make_config() 를 호출한다.
numpy/onnx 등 ML 스택을 끌어오지 않도록 paths 외에는 표준 라이브러리만 쓴다.

batch_size 는 toml 에 **절대 적지 않는다**. sd-scripts 는 toml 의 batch_size 가 있으면
--train_batch_size 를 무시하므로(library/config_util.py 의 search_value), 배치는 항상
학습 스크립트의 --train_batch_size(= BATCH_SIZE/GPU수)가 정한다. 그래야 같은 toml 로
1/2/4 GPU 를 다 돌릴 수 있다.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from .paths import DATASET_ROOT, IMAGE_EXTENSIONS


def parse_resolution(value: str) -> list[int]:
    """'1024' -> [1024, 1024], '1024,768' -> [1024, 768]."""
    parts = [p.strip() for p in value.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        return [n, n]
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    raise ValueError(f"invalid resolution: {value!r}")


def count_images(folder: Path) -> int:
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def render_toml(
    *,
    subsets: list[tuple[str, int]],
    resolution: list[int],
    keep_tokens: int,
    caption_extension: str,
    shuffle_caption: bool,
    enable_bucket: bool,
    bucket_no_upscale: bool,
    min_bucket_reso: int,
    max_bucket_reso: int,
) -> str:
    """subsets: (image_dir, num_repeats) 목록. 플래트닝 전엔 1개, repeat_<N> 구조면 여러 개.

    batch_size 는 적지 않는다(모듈 docstring 참고) — 학습 스크립트의 --train_batch_size 가
    GPU당 배치를 정한다.
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


def total_repeats(data_dir: Path, name: str, subsets: list[tuple[str, int]]) -> int:
    """반복수 반영 에폭당 학습 이미지 수. 총 스텝 = 이 값 × epochs / batch (ceil)."""
    def _subset_dir(image_dir: str) -> Path:
        tail = Path(image_dir).name
        return data_dir if tail == name else data_dir / tail
    return sum(count_images(_subset_dir(d)) * r for d, r in subsets)


def make_config(
    name: str,
    *,
    resolution: list[int],
    num_repeats: int = 2,
    keep_tokens: int = 1,
    caption_extension: str = ".txt",
    shuffle_caption: bool = True,
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
    min_bucket_reso: int = 512,
    max_bucket_reso: int = 1536,
) -> dict:
    """dataset/<name> 의 toml 텍스트 + 요약 통계를 만든다. 파일은 쓰지 않는다 — 호출자 몫.

    raises ValueError: 폴더 없음 / 이미지 없음.
    """
    data_dir = (DATASET_ROOT / name).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"dataset dir not found: dataset/{name}")

    subsets, n_img = collect_subsets(data_dir, name, num_repeats)
    if n_img == 0:
        raise ValueError(f"no images in dataset/{name}")

    # repeat_<N> 구조면 하위까지, 아니면 top-level 캡션 수를 센다.
    multi = len(subsets) > 1 or subsets[0][0] != f"../dataset/{name}"
    caption_glob = (data_dir.rglob(f"*{caption_extension}") if multi
                    else data_dir.glob(f"*{caption_extension}"))
    n_txt = sum(1 for _ in caption_glob)

    toml = render_toml(
        subsets=subsets,
        resolution=resolution,
        keep_tokens=keep_tokens,
        caption_extension=caption_extension,
        shuffle_caption=shuffle_caption,
        enable_bucket=enable_bucket,
        bucket_no_upscale=bucket_no_upscale,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
    )
    # 에폭당 학습이미지(반복 반영)와 배치별 스텝수. enable_bucket 시 실제 스텝은 조금 더 클 수 있다.
    total = total_repeats(data_dir, name, subsets)
    return {
        "name": name,
        "toml": toml,
        "out_path": str(DATASET_ROOT / f"{name}.toml"),
        "images": n_img,
        "captions": n_txt,
        "subsets": [[d, r] for d, r in subsets],
        "total_reps": total,
        "steps": {str(b): math.ceil(total / b) for b in (1, 2, 4, 8, 16)},
    }
