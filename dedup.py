"""DINOv2 CLS 임베딩 기반 차분/중복 후보 탐지 + `_sabun` 리네임 규칙 (공용 로직).

`0a_dedup_raw.py` 가 import 한다. `app.py` 가 WD14Tagger 를 담고 `1_tag_dataset.py` 가
재사용하는 것과 같은 구조다.

여기서 다루는 이름은 두 종류다.

  rel   : raw_dir 기준 상대경로 (예: `C99_FAKE DOLL_2120939/07_7.png`)
  flat  : 그 상대경로의 `/` 를 `__` 로 치환한 것 (`C99_FAKE DOLL_2120939__07_7.png`)

`0_process_raw.py` 는 **flat** 이름으로 차분/강조를 판정하므로, 그룹핑과 검증도 flat 기준으로
한다. 다만 실제 리네임은 basename 만 바꾼다(파일은 원래 디렉토리에 그대로 둔다).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "dataset_raw"
CACHE_ROOT = ROOT / ".dedup"

# app.py 와 동일 규약. 이 모듈은 onnxruntime 태깅 스택과 무관하므로 app 을 import 하지 않고
# 상수만 로컬에 둔다 (2_make_config.py 가 같은 이유로 복제하는 것과 동일).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

MODEL_REPO = os.environ.get("DINOV2_MODEL", "facebook/dinov2-base")

# ---------------------------------------------------------------------------
# 이름 규칙
#
# 아래 두 상수는 0_process_raw.py 의 것과 **반드시 동일해야 한다**. 숫자로 시작하는 모듈은
# import 할 수 없어(`import 0_process_raw` = SyntaxError) 복제한다.
# ---------------------------------------------------------------------------
SABUN_RE = re.compile(r"_sabun(?:_.*)?$")
SPECIAL_MARKER = "_special_"

# 0_process_raw.py 가 인식하지 못하는 `_sabun1` 류까지 포함해 잡아내는 느슨한 패턴.
# 그룹 시드(기존 표기로 이미 묶여 있던 것)와 규칙 위반 경고에 쓴다.
LOOSE_SABUN_RE = re.compile(r"_sabun(?:\d+|_.*)?$")

SABUN_SUFFIX = "_sabun_{n}"  # 규칙에 맞는 차분 접미사. web UI 의 JS 와 같은 규칙.

# 제외된 이미지는 raw_dir 의 **형제** 폴더 `dataset_raw/<name>_excluded/` 로 옮긴다.
# raw_dir 안에 두면 0_process_raw.py 의 rglob 이 제외한 이미지를 그대로 다시 긁어간다.
# 내부적으로는 이 접두사를 붙인 rel 문자열 하나로 다루고(`..` 없는 순수 상대경로 유지),
# 실제 경로 변환은 resolve_rel() 이 담당한다.
EXCLUDE_PREFIX = "_excluded/"


def sabun_base(stem: str) -> str:
    """차분 접미사를 제거한 그룹 base. 0_process_raw.sabun_base 와 동일."""
    return SABUN_RE.sub("", stem)


def loose_base(stem: str) -> str:
    """`_sabun1` 같은 비규격 표기까지 벗겨낸 그룹 base (그룹 시드용)."""
    return LOOSE_SABUN_RE.sub("", stem)


def is_sabun(stem: str) -> bool:
    return SABUN_RE.search(stem) is not None


def is_special(stem: str) -> bool:
    """0_process_raw 판정과 동일: 차분 접미사를 벗긴 base 에 `_special_` 이 있는가."""
    return SPECIAL_MARKER in sabun_base(stem)


def malformed_sabun(stem: str) -> bool:
    """`_sabun1` 처럼 차분 의도로 보이나 0_process_raw 가 인식 못 하는 이름인가."""
    return LOOSE_SABUN_RE.search(stem) is not None and not is_sabun(stem)


def flat_name(rel: PurePosixPath | str) -> str:
    """rel 경로를 0_process_raw 와 같은 규약으로 플래트닝."""
    return str(rel).replace("/", "__").replace(os.sep, "__")


def flat_stem(rel: PurePosixPath | str) -> str:
    name = flat_name(rel)
    suffix = PurePosixPath(name).suffix
    return name[: -len(suffix)] if suffix else name


def dir_prefix(rel: PurePosixPath | str) -> str:
    """flat 이름 중 디렉토리에서 온 앞부분. 최상위 파일이면 빈 문자열."""
    parent = PurePosixPath(str(rel)).parent
    return "" if str(parent) == "." else flat_name(parent) + "__"


def inherits_special(rel: PurePosixPath | str) -> bool:
    """원시 폴더명(`*_special/`) 때문에 이미 강조로 판정되는가.

    mystyle_raw/album1_special/ 처럼 앨범 폴더가 강조인 경우, basename 을
    건드리지 않아도 flat 이름에 `_special_` 이 들어간다. 이 경우 UI 의 강조 토글은 의미가 없다.
    """
    return SPECIAL_MARKER in dir_prefix(rel)


# ---------------------------------------------------------------------------
# 이미지 수집
# ---------------------------------------------------------------------------
def resolve_raw(arg: str) -> Path:
    """0_process_raw.resolve_raw 와 동일 규약: 경로거나 dataset_raw/<이름>."""
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    candidate = (RAW_ROOT / arg).resolve()
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"[error] raw dataset not found: {arg} (dataset_raw/{arg} 확인)")


def excluded_dir(raw_dir: Path) -> Path:
    return raw_dir.parent / f"{raw_dir.name}_excluded"


def resolve_rel(raw_dir: Path, rel: str) -> Path:
    """rel 문자열 → 실제 경로. `_excluded/` 접두사는 형제 폴더로 보낸다."""
    if rel.startswith(EXCLUDE_PREFIX):
        return excluded_dir(raw_dir) / rel[len(EXCLUDE_PREFIX) :]
    return raw_dir / rel


def _scan(base: Path, prefix: str = "") -> list[str]:
    if not base.is_dir():
        return []
    return [
        prefix + p.relative_to(base).as_posix()
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_images(raw_dir: Path) -> list[PurePosixPath]:
    """raw_dir 아래 모든 이미지의 상대경로(POSIX). 제외 폴더는 형제라 잡히지 않는다."""
    return [PurePosixPath(r) for r in _scan(raw_dir)]


def collect_excluded(raw_dir: Path) -> list[str]:
    """이미 제외해 둔 이미지의 rel (`_excluded/...`). 복구용."""
    return _scan(excluded_dir(raw_dir), EXCLUDE_PREFIX)


def collect_all(raw_dir: Path) -> list[str]:
    """리네임 대상이 될 수 있는 전부 (제외 폴더 포함)."""
    return _scan(raw_dir) + collect_excluded(raw_dir)


def cache_dir(raw_dir: Path) -> Path:
    """임베딩/썸네일/undo 로그 위치.

    raw_dir **바깥**에 둔다. 안에 두면 0_process_raw.py 의 rglob 이 썸네일 .jpg 까지
    학습 이미지로 긁어간다.
    """
    d = CACHE_ROOT / raw_dir.name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# DINOv2 임베딩
# ---------------------------------------------------------------------------
def load_square_rgb(path: Path, size: int) -> Image.Image:
    """알파를 흰 배경에 합성 → 정사각 패딩 → size x size.

    app.py 의 _prepare_image 와 같은 규약(정사각 패딩)이다. DINOv2 기본 전처리는
    shortest-edge 256 → center-crop 224 라서 세로로 긴 일러스트의 위아래가 잘려나간다.
    차분 판별은 전체 구도를 봐야 하므로 크롭 대신 패딩한다. (224 = 16*14, DINOv2 patch=14)
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    w, h = image.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(image, ((side - w) // 2, (side - h) // 2))
    return square.resize((size, size), Image.BICUBIC)


class Dinov2Embedder:
    """facebook/dinov2-base 의 CLS 토큰을 L2 정규화해서 반환한다 (코사인 = 내적)."""

    def __init__(self, repo: str = MODEL_REPO, device: str | None = None, batch_size: int = 16):
        self.repo = repo
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._proc = None
        self._size = 224

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._proc = AutoImageProcessor.from_pretrained(self.repo)
        self._model = AutoModel.from_pretrained(self.repo).to(self.device).eval()
        print(f"[dinov2] {self.repo} on {self.device} (hidden={self._model.config.hidden_size})")

    def embed(self, paths: list[Path], progress=None, on_ready=None) -> np.ndarray:
        """on_ready: 모델 로딩이 끝나고 실제 임베딩이 시작되는 시점에 불린다.
        첫 호출은 모델을 올리느라 수 초 걸리는데, 그 구간을 진행률 0% 로 보여주면
        멈춘 것처럼 보인다."""
        self._load()
        if on_ready:
            on_ready()
        import torch

        dim = self._model.config.hidden_size
        out = np.zeros((len(paths), dim), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(paths), self.batch_size):
                chunk = paths[i : i + self.batch_size]
                images = [load_square_rgb(p, self._size) for p in chunk]
                # 이미 정사각 224 로 맞췄으므로 프로세서는 rescale/normalize 만 하게 한다.
                inputs = self._proc(
                    images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
                ).to(self.device)
                hidden = self._model(**inputs).last_hidden_state  # (B, 1+patches, D)
                cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
                out[i : i + len(chunk)] = cls.float().cpu().numpy()
                if progress:
                    progress(min(i + self.batch_size, len(paths)), len(paths))
        return out


_EMBEDDER: Dinov2Embedder | None = None


def get_embedder(batch_size: int = 16) -> Dinov2Embedder:
    """프로세스당 하나만 만든다. 웹 UI 에서 데이터셋을 갈아탈 때마다 모델을 다시 올리면
    전환이 매번 수 초씩 느려진다."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Dinov2Embedder(batch_size=batch_size)
    _EMBEDDER.batch_size = batch_size
    return _EMBEDDER


def _sig(path: Path) -> list:
    st = path.stat()
    return [int(st.st_mtime), st.st_size]


def _read_cache(raw_dir: Path, sigs: dict[str, list]) -> dict[str, np.ndarray]:
    """캐시에서 파일이 그대로인(mtime+size 일치) 임베딩만 골라 온다."""
    cache = cache_dir(raw_dir) / "embeddings.npz"
    if not cache.exists():
        return {}
    try:
        blob = np.load(cache, allow_pickle=False)
        out = {}
        for i, k in enumerate(blob["keys"]):
            k = str(k)
            if k in sigs and list(blob["sigs"][i]) == sigs[k]:
                out[k] = blob["emb"][i]
        return out
    except Exception as exc:  # 캐시가 깨졌으면 조용히 버리고 다시 만든다
        print(f"[warn] 임베딩 캐시 무시: {exc}")
        return {}


def pending_count(raw_dir: Path, rels: list[PurePosixPath]) -> int:
    """아직 임베딩되지 않은 이미지 수. 0 이면 바로 그룹핑할 수 있다."""
    sigs = {str(r): _sig(raw_dir / r) for r in rels}
    return len(rels) - len(_read_cache(raw_dir, sigs))


def cached_embeddings(raw_dir: Path, rels: list[PurePosixPath]) -> np.ndarray | None:
    """전부 캐시돼 있으면 스택해서, 하나라도 없으면 None. (임베딩을 새로 돌리지 않는다)"""
    sigs = {str(r): _sig(raw_dir / r) for r in rels}
    cached = _read_cache(raw_dir, sigs)
    if len(cached) != len(rels):
        return None
    return np.stack([cached[str(r)] for r in rels]).astype(np.float32)


def build_embeddings(
    raw_dir: Path,
    rels: list[PurePosixPath],
    *,
    force: bool = False,
    batch_size: int = 16,
    progress=None,
    on_ready=None,
) -> np.ndarray:
    """임베딩 캐시(.dedup/<name>/embeddings.npz)를 채우고 전체를 반환한다.

    progress(done, total) 콜백은 **아직 임베딩 안 된 것** 기준으로 불린다.
    """
    keys = [str(r) for r in rels]
    sigs = {k: _sig(raw_dir / k) for k in keys}
    cached = {} if force else _read_cache(raw_dir, sigs)

    todo = [k for k in keys if k not in cached]
    if todo:
        print(f"[dinov2] 임베딩 {len(todo)}장 (캐시 재사용 {len(cached)}장)")
        emb = get_embedder(batch_size).embed([raw_dir / k for k in todo], progress, on_ready)
        for i, k in enumerate(todo):
            cached[k] = emb[i]
    else:
        print(f"[dinov2] 캐시 재사용 {len(cached)}장")

    stacked = np.stack([cached[k] for k in keys]).astype(np.float32)
    np.savez(
        cache_dir(raw_dir) / "embeddings.npz",
        keys=np.array(keys),
        sigs=np.array([sigs[k] for k in keys], dtype=np.int64),
        emb=stacked,
    )
    return stacked


def list_raw_datasets() -> list[str]:
    """dataset_raw/ 아래의 원시 데이터셋 이름들. 제외 폴더는 뺀다."""
    if not RAW_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in RAW_ROOT.iterdir()
        if p.is_dir() and not p.name.endswith("_excluded") and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# 그룹핑
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_groups(
    rels: list[PurePosixPath], emb: np.ndarray, threshold: float, *, seed_by_name: bool = True
) -> list[list[int]]:
    """코사인 유사도 >= threshold 를 잇는 연결요소 + 기존 파일명 시드.

    반환: 크기 2 이상인 그룹들의 인덱스 목록 (그룹 내부는 rel 순).
    """
    n = len(rels)
    uf = _UnionFind(n)

    if seed_by_name:
        # 이미 `_sabun*` 로 묶여 있던 것은 유사도와 무관하게 한 그룹으로 본다.
        # (`_sabun1` 같은 비규격 표기를 교정 대상으로 UI 에 올리기 위해서도 필요)
        by_base: dict[str, list[int]] = defaultdict(list)
        for i, rel in enumerate(rels):
            by_base[loose_base(flat_stem(rel))].append(i)
        for idxs in by_base.values():
            for j in idxs[1:]:
                uf.union(idxs[0], j)

    # 블록 단위 행렬곱 (수천 장까지 무리 없음)
    block = 1024
    for s in range(0, n, block):
        sim = emb[s : s + block] @ emb.T  # (b, n)
        for local, i in enumerate(range(s, min(s + block, n))):
            hits = np.nonzero(sim[local] >= threshold)[0]
            for j in hits:
                if int(j) > i:
                    uf.union(i, int(j))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# 리네임 계획 / 검증 / 적용
# ---------------------------------------------------------------------------
@dataclass
class Check:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_flat_names(flats: list[str]) -> Check:
    """최종 flat 이름 집합이 0_process_raw.py 를 통과할지 검증한다.

    errors 가 있으면 적용을 막는다. 0_process_raw 가 SystemExit 하는 조건이 곧 error 다.
    """
    out = Check()
    counts: dict[str, int] = defaultdict(int)
    for f in flats:
        counts[f] += 1
    for name, c in sorted(counts.items()):
        if c > 1:
            out.errors.append(f"플래트닝 이름 충돌: {name} ({c}개)")

    groups: dict[str, list[str]] = defaultdict(list)
    for f in flats:
        stem = f[: -len(PurePosixPath(f).suffix)] if PurePosixPath(f).suffix else f
        groups[sabun_base(stem)].append(stem)

    for base, stems in sorted(groups.items()):
        has_sabun = any(is_sabun(s) for s in stems)
        has_original = any(not is_sabun(s) for s in stems)
        if has_sabun and not has_original:
            # 0_process_raw.py:126 이 여기서 중단한다.
            out.errors.append(f"차분 그룹에 원본이 없다 (원본 {base}.* 누락): {', '.join(stems)}")
        if (SPECIAL_MARKER not in base) and ("special" in base.lower()):
            # 0_process_raw.py:124 와 같은 경고.
            out.warnings.append(f"'special' 문자열이 있으나 강조 마커(_special_)가 아님 → 표준 취급: {base}")

    for f in flats:
        stem = f[: -len(PurePosixPath(f).suffix)] if PurePosixPath(f).suffix else f
        if malformed_sabun(stem):
            out.warnings.append(
                f"차분처럼 보이나 0_process_raw 가 인식 못 하는 이름 (`_sabun_N` 으로 고칠 것): {f}"
            )
    return out


def plan_moves(raw_dir: Path, targets: dict[str, str]) -> tuple[list[tuple[str, str]], Check]:
    """targets: {현재 rel: 새 rel}. 실제로 바뀌는 것만 추려 검증한다.

    반환: (moves, check). moves 는 [(src_rel, dst_rel)].
    """
    current = collect_all(raw_dir)
    moves = [(src, dst) for src, dst in targets.items() if src != dst]

    check = Check()
    for src, dst in moves:
        if src not in current:
            check.errors.append(f"원본 파일이 없다: {src}")
        if not dst or dst.startswith("/") or ".." in PurePosixPath(dst).parts:
            check.errors.append(f"허용되지 않는 경로: {dst}")
        if PurePosixPath(src).suffix.lower() != PurePosixPath(dst).suffix.lower():
            check.errors.append(f"확장자를 바꿀 수 없다: {src} → {dst}")

    moved = {src for src, _ in moves}
    final_rels = [r for r in current if r not in moved] + [dst for _, dst in moves]

    # 경로 충돌은 제외 폴더까지 **포함해서** 검사한다. 서로 다른 폴더의 동명 파일을 함께
    # 제외하면 둘 다 `_excluded/<같은 이름>` 으로 가서 한 장이 조용히 덮어써진다.
    counts: dict[str, int] = defaultdict(int)
    for r in final_rels:
        counts[r] += 1
    for r, c in sorted(counts.items()):
        if c > 1:
            check.errors.append(f"같은 경로로 여러 파일이 간다: {r} ({c}개)")

    # 반면 차분/강조 이름 규칙은 학습에 들어가는 것(=제외되지 않은 것)만 따진다.
    name_check = check_flat_names(
        [flat_name(r) for r in final_rels if not r.startswith(EXCLUDE_PREFIX)]
    )
    check.errors += name_check.errors
    check.warnings += name_check.warnings
    return moves, check


def remap_cache(raw_dir: Path, moves: list[tuple[str, str]]) -> None:
    """리네임을 임베딩 캐시 키에 반영한다.

    파일 내용은 그대로이므로(이름만 바뀜) 다시 임베딩할 이유가 없다. 이걸 안 하면
    적용할 때마다 데이터셋 전체가 재임베딩된다.
    """
    cache = cache_dir(raw_dir) / "embeddings.npz"
    if not cache.exists() or not moves:
        return
    try:
        blob = np.load(cache, allow_pickle=False)
        table = dict(moves)
        keys = [table.get(str(k), str(k)) for k in blob["keys"]]
        np.savez(cache, keys=np.array(keys), sigs=blob["sigs"], emb=blob["emb"])
    except Exception as exc:
        print(f"[warn] 임베딩 캐시 갱신 실패(다음 실행 때 재임베딩): {exc}")


def log_path(raw_dir: Path) -> Path:
    return cache_dir(raw_dir) / "rename_log.json"


def _read_log(raw_dir: Path) -> dict:
    p = log_path(raw_dir)
    if not p.exists():
        return {"version": 1, "dataset": raw_dir.name, "history": []}
    return json.loads(p.read_text(encoding="utf-8"))


def apply_moves(raw_dir: Path, moves: list[tuple[str, str]], *, record: bool = True) -> int:
    """2단계 리네임으로 적용한다 (a→b, b→a 같은 스왑/순환 안전).

    1) 모든 src 를 유니크한 임시 이름으로
    2) 임시 → dst
    """
    if not moves:
        return 0
    stamp = int(time.time())
    tmp: list[tuple[Path, Path]] = []
    for i, (src, dst) in enumerate(moves):
        s = resolve_rel(raw_dir, src)
        t = s.with_name(f".dedup_tmp_{stamp}_{i}{s.suffix}")
        s.rename(t)
        tmp.append((t, resolve_rel(raw_dir, dst)))
    for t, dst_path in tmp:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        t.rename(dst_path)
    remap_cache(raw_dir, moves)

    if record:
        data = _read_log(raw_dir)
        data["history"].append({"ts": stamp, "moves": [[s, d] for s, d in moves]})
        log_path(raw_dir).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return len(moves)


def undo_last(raw_dir: Path) -> tuple[int, str]:
    """가장 최근 적용 배치를 되돌린다. 반환: (되돌린 수, 메시지)."""
    data = _read_log(raw_dir)
    if not data["history"]:
        return 0, "되돌릴 기록이 없다."
    batch = data["history"][-1]
    reverse = [(d, s) for s, d in batch["moves"]]

    missing = [d for d, _s in reverse if not resolve_rel(raw_dir, d).exists()]
    if missing:
        return 0, f"되돌릴 수 없다 — 적용 후 파일이 바뀌었다: {', '.join(missing[:3])}"

    n = apply_moves(raw_dir, reverse, record=False)
    data["history"].pop()
    log_path(raw_dir).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(batch["ts"]))
    return n, f"{ts} 배치 {n}건을 되돌렸다."
