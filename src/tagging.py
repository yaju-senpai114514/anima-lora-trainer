"""WD14 ONNX 태깅 + blacklist/trigger 캡션 조립 (공용 로직).

웹 서버(src/webui/server.py)의 태깅 잡이 tag_folder() 를 호출한다.
WD14Tagger 는 모델/전처리/threshold 동작의 단일 소스다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from .paths import IMAGE_EXTENSIONS

MODEL_REPO = os.environ.get(
    "WD_TAGGER_MODEL", "SmilingWolf/wd-eva02-large-tagger-v3"
)
MODEL_FILE = os.environ.get("WD_TAGGER_MODEL_FILE", "model.onnx")
TAGS_FILE = os.environ.get("WD_TAGGER_TAGS_FILE", "selected_tags.csv")


def _select_providers() -> list[str]:
    """실행 프로바이더 선택: 기본은 CUDA 가능 시 GPU, 아니면 CPU 폴백.

    WD_TAGGER_PROVIDERS 로 강제 지정 가능 (쉼표 구분, 예: "CPUExecutionProvider").
    onnxruntime-gpu 미설치 시 CUDAExecutionProvider 가 목록에 없으므로 자동 CPU 폴백.
    """
    override = os.environ.get("WD_TAGGER_PROVIDERS")
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in preferred if p in available]
    return selected or ["CPUExecutionProvider"]


class WD14Tagger:
    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._input_height = 448
        self._tags: pd.DataFrame | None = None

    def _load(self) -> None:
        if self._session is not None and self._tags is not None:
            return

        model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        tags_path = hf_hub_download(MODEL_REPO, TAGS_FILE)

        providers = _select_providers()
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            # torch(cu12)의 CUDA/cuDNN 라이브러리를 프로세스에 선로드해야
            # onnxruntime 가 CUDAExecutionProvider 를 실제로 초기화할 수 있다.
            ort.preload_dlls()
        self._session = ort.InferenceSession(model_path, providers=providers)
        print(f"[wd14] onnxruntime providers: {self._session.get_providers()}")
        input_info = self._session.get_inputs()[0]
        self._input_name = input_info.name
        shape = input_info.shape
        if len(shape) >= 3 and isinstance(shape[1], int):
            self._input_height = shape[1]
        elif len(shape) >= 3 and isinstance(shape[2], int):
            self._input_height = shape[2]

        self._tags = pd.read_csv(tags_path)

    def tag_image(
        self, image_path: Path, general_threshold: float, character_threshold: float
    ) -> dict[str, Any]:
        self._load()
        assert self._session is not None
        assert self._tags is not None

        image = self._prepare_image(image_path)
        output = self._session.run(None, {self._input_name: image})[0][0]

        rows: list[dict[str, Any]] = []
        for tag_row, score in zip(self._tags.itertuples(index=False), output):
            name = str(getattr(tag_row, "name")).replace("_", " ")
            category = int(getattr(tag_row, "category"))
            rows.append({"name": name, "category": category, "score": float(score)})

        rating = sorted(
            (row for row in rows if row["category"] == 9),
            key=lambda row: row["score"],
            reverse=True,
        )[:1]
        general = [
            row
            for row in rows
            if row["category"] == 0 and row["score"] >= general_threshold
        ]
        character = [
            row
            for row in rows
            if row["category"] == 4 and row["score"] >= character_threshold
        ]
        selected = sorted(character, key=lambda row: row["score"], reverse=True)
        selected += sorted(general, key=lambda row: row["score"], reverse=True)

        return {
            "tags": ", ".join(row["name"] for row in selected),
            "rating": rating[0] if rating else None,
            "count": len(selected),
        }

    def _prepare_image(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        width, height = image.size
        side = max(width, height)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - width) // 2, (side - height) // 2))
        square = square.resize((self._input_height, self._input_height), Image.BICUBIC)

        array = np.asarray(square, dtype=np.float32)
        array = array[:, :, ::-1]  # RGB -> BGR (WD 1.4 입력 규약)
        array = np.ascontiguousarray(array)  # 음수 stride 제거 (onnxruntime 안전)
        return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# 캡션 조립: blacklist 제거 → trigger/rating prepend → 중복 제거
# ---------------------------------------------------------------------------
def normalize(tag: str) -> str:
    """매칭용 정규화: 소문자 + '_'->공백 + 공백 축약."""
    return " ".join(tag.replace("_", " ").lower().split())


def load_blacklist(path: Path) -> set[str]:
    """blacklist.txt 파싱. '#' 주석/빈 줄 무시, 정규화해서 set 으로."""
    if not path.exists():
        raise FileNotFoundError(f"blacklist not found: {path}")
    banned: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        banned.add(normalize(line))
    return banned


def build_caption(
    raw_tags: list[str],
    trigger: str,
    banned: set[str],
    rating: str | None,
    removed_counter: dict[str, int],
) -> str:
    """blacklist 제거 + trigger/rating prepend + 중복 제거(순서 보존)."""
    trig_norm = normalize(trigger)
    kept: list[str] = []
    for tag in raw_tags:
        tag = " ".join(tag.split())
        if not tag:
            continue
        norm = normalize(tag)
        if norm in banned:
            removed_counter[norm] = removed_counter.get(norm, 0) + 1
            continue
        if norm == trig_norm:  # WD 가 우연히 trigger 를 뱉으면 중복 방지
            continue
        kept.append(tag)

    head = [trigger]
    if rating:
        head.append(rating)
    ordered = head + kept
    # 순서 보존 중복 제거
    return ", ".join(dict.fromkeys(ordered))


# ---------------------------------------------------------------------------
# 데이터셋 이미지 수집 (dataset/<name>/ 및 repeat_<N> 서브셋 구조)
# ---------------------------------------------------------------------------
def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def has_repeat_subsets(folder: Path) -> bool:
    """dedup 익스포트가 만든 repeat_<N> 서브셋 구조인지."""
    return any(p.is_dir() and p.name.startswith("repeat_") for p in folder.iterdir())


# ---------------------------------------------------------------------------
# 폴더 태깅 엔진 (+ WD 추론 캐시)
# ---------------------------------------------------------------------------
_TAGGER: WD14Tagger | None = None


def get_tagger() -> WD14Tagger:
    """프로세스당 하나만 만든다(잡마다 ONNX 모델 재로딩 방지)."""
    global _TAGGER
    if _TAGGER is None:
        _TAGGER = WD14Tagger()
    return _TAGGER


def _sig(path: Path) -> list:
    st = path.stat()
    return [int(st.st_mtime), st.st_size]


def _load_cache(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] WD 캐시 무시: {exc}")
        return {}


def tag_folder(
    folder: Path,
    trigger: str,
    banned: set[str],
    *,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    include_rating: bool = False,
    overwrite: bool = False,
    cache_path: Path | None = None,
    progress=None,
    on_ready=None,
) -> dict:
    """폴더의 이미지를 태깅해 옆에 .txt 캡션을 쓴다. repeat_<N> 구조면 자동 재귀.

    WD 추론 결과는 cache_path(JSON)에 **심링크를 푼 실경로**의 mtime+size + 모델 +
    threshold 를 키로 캐시한다 — dedup 재익스포트가 .txt 를 지워도(심링크 폴더를 갈아엎는다)
    재태깅은 추론 없이 캡션 조립만 다시 하므로 몇 초에 끝난다.

    progress(done, total) 은 스킵 포함 매 이미지마다, on_ready() 는 모델 로딩 완료 시 불린다.
    returns: {"images","processed","skipped","failed","cached","removed"}
    """
    recursive = has_repeat_subsets(folder)
    images = list_images(folder, recursive)
    if not images:
        raise ValueError(f"이미지가 없다: {folder}")

    tagger = get_tagger()
    tagger._load()
    if on_ready:
        on_ready()

    cache = _load_cache(cache_path)
    removed: dict[str, int] = {}
    processed = skipped = failed = cached_hits = 0

    try:
        for i, img in enumerate(images, 1):
            txt = img.with_suffix(".txt")
            if txt.exists() and not overwrite:
                skipped += 1
                if progress:
                    progress(i, len(images))
                continue

            real = img.resolve()
            key = str(real)
            entry = cache.get(key)
            sig = _sig(real)
            if (entry and entry.get("sig") == sig and entry.get("model") == MODEL_REPO
                    and entry.get("gth") == general_threshold
                    and entry.get("cth") == character_threshold):
                tags_str, rating_name = entry["tags"], entry.get("rating")
                cached_hits += 1
            else:
                try:
                    result = tagger.tag_image(img, general_threshold, character_threshold)
                except Exception as exc:  # 이미지 손상 등은 건너뛰고 계속
                    print(f"[warn] {img.name}: {exc}")
                    failed += 1
                    if progress:
                        progress(i, len(images))
                    continue
                tags_str = result["tags"]
                rating_name = result["rating"]["name"] if result.get("rating") else None
                cache[key] = {"sig": sig, "model": MODEL_REPO,
                              "gth": general_threshold, "cth": character_threshold,
                              "tags": tags_str, "rating": rating_name}

            caption = build_caption(
                tags_str.split(","), trigger, banned,
                rating_name if include_rating else None, removed)
            txt.write_text(caption, encoding="utf-8")
            processed += 1
            if progress:
                progress(i, len(images))
    finally:
        if cache_path and cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    return {"images": len(images), "processed": processed, "skipped": skipped,
            "failed": failed, "cached": cached_hits, "removed": removed}
