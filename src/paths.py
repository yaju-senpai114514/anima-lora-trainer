"""레포 공통 경로/확장자 규약. src/ 의 모든 모듈과 루트 스크립트가 이걸 쓴다."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 레포 루트 (src/ 의 부모)

DATASET_ROOT = ROOT / "dataset"        # 학습용 데이터셋 (익스포트/태깅/toml 대상)
RAW_ROOT = ROOT / "dataset_raw"        # 원시(중첩) 데이터셋 — 완전 read-only
DEDUP_CACHE_ROOT = ROOT / ".dedup"     # DINOv2 임베딩 / 썸네일 / state.json 캐시
BLACKLIST_PATH = ROOT / "blacklist.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
