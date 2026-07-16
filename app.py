from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset"
MODEL_REPO = os.environ.get(
    "WD_TAGGER_MODEL", "SmilingWolf/wd-eva02-large-tagger-v3"
)
MODEL_FILE = os.environ.get("WD_TAGGER_MODEL_FILE", "model.onnx")
TAGS_FILE = os.environ.get("WD_TAGGER_TAGS_FILE", "selected_tags.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


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
