"""DINOv2 CLS 임베딩 + centroid 응집형(agglomerative) 그룹핑 상태 모델 (공용 로직).

웹 서버(src/webui/server.py)의 원시 dedup 탭과 런치 스크립트 1_dataset_server.py 가 쓴다.

설계 (2026-07 재설계):
  - dataset_raw/<원시> 의 원본 파일은 **완전 read-only**. 리네임/이동/삭제 안 한다.
  - 그룹핑 / 제외 / 강조(specialize) 현황은 `.dedup/<name>/state.json` 메타데이터로만 관리.
  - 기본적으로 모든 이미지는 개별 그룹(싱글턴). 그룹은 centroid 임베딩으로 대표되어
    다시 클러스터링 대상이 된다(응집형). threshold 를 낮춰가며 후보 쌍을 병합한다.
  - cannot_link: 두 엔티티가 "같은 그룹이 아니다"라고 명시하면(rel 쌍으로 저장) 자동 병합 후보에서
    영구히 제외된다.
  - 익스포트: state + repeat/rounding 으로 dataset/<name>/repeat_<N>/ 심볼릭 링크를 만든다
    (파일명 규칙이 아니라 그룹 메타데이터가 반복수를 구동한다).
      · 그룹 전체 반복수 합 = R (강조 그룹 = 2R). 그룹 크기 k 면 각 멤버 round(total/k).
      · 제외 이미지는 익스포트에서 빠진다.

이름 규약:
  rel   : raw_dir 기준 상대경로 (POSIX). state.json 의 안정적 키다(파일은 read-only 라 불변).
  flat  : 그 상대경로의 `/` 를 `__` 로 치환한 것. 익스포트 심볼릭 링크의 파일명.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

from .paths import DATASET_ROOT, DEDUP_CACHE_ROOT, IMAGE_EXTENSIONS, RAW_ROOT

CACHE_ROOT = DEDUP_CACHE_ROOT
MODEL_REPO = os.environ.get("DINOV2_MODEL", "facebook/dinov2-base")

# 임베딩 속도의 실제 병목은 GPU forward(수백 ms)가 아니라 PIL 디코드/리사이즈다. 그래서 이미지
# 로딩을 스레드풀로 병렬화한다(항상 켜짐 — 값이 안 바뀌는 결정론적 이득). WORKERS=0 이면 자동.
WORKERS = int(os.environ.get("DINOV2_WORKERS", "0") or "0")

# fast 모드: DINOv2 를 half precision(GPU 에서 bf16, 미지원이면 fp16)으로 돌리고 fast 이미지
# 프로세서를 쓴다. GPU forward 가 이미 짧아 이득은 작지만, 값이 fp32 와 미세하게 달라지므로
# 캐시를 모드별로 태깅(_cache_mode)해 모드 전환 시 자동 무효화한다 — 수동 삭제 불필요.
# 프로세스 전역 스위치(CLI --fast / 환경변수 DINOV2_FAST=1).
FAST = os.environ.get("DINOV2_FAST", "").lower() in ("1", "true", "yes", "on")


def _cache_mode() -> str:
    return "fast" if FAST else "full"


def _default_workers() -> int:
    return WORKERS if WORKERS > 0 else min(16, (os.cpu_count() or 8))


STATE_VERSION = 2


# ---------------------------------------------------------------------------
# 경로 / 이미지 수집
# ---------------------------------------------------------------------------
def flat_name(rel: PurePosixPath | str) -> str:
    """rel 경로를 `/` → `__` 로 플래트닝. 익스포트 심볼릭 링크 파일명."""
    return str(rel).replace("/", "__").replace(os.sep, "__")


def resolve_raw(arg: str) -> Path:
    """경로거나 dataset_raw/<이름>."""
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    candidate = (RAW_ROOT / arg).resolve()
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"[error] raw dataset not found: {arg} (dataset_raw/{arg} 확인)")


def _scan(base: Path) -> list[str]:
    if not base.is_dir():
        return []
    return [
        p.relative_to(base).as_posix()
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_images(raw_dir: Path) -> list[PurePosixPath]:
    """raw_dir 아래 모든 이미지의 상대경로(POSIX)."""
    return [PurePosixPath(r) for r in _scan(raw_dir)]


def resolve_rel(raw_dir: Path, rel: str) -> Path:
    """rel 문자열 → 실제 경로. raw 는 read-only 라 항상 raw_dir 아래다."""
    return raw_dir / rel


def cache_dir(raw_dir: Path) -> Path:
    """임베딩/썸네일/state 위치. raw_dir 바깥(.dedup/<name>/)에 둔다."""
    d = CACHE_ROOT / raw_dir.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_raw_datasets() -> list[str]:
    """dataset_raw/ 아래의 원시 데이터셋 이름들."""
    if not RAW_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in RAW_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# DINOv2 임베딩  (재설계 전과 동일 — 원본을 읽기만 한다)
# ---------------------------------------------------------------------------
def load_square_rgb(path: Path, size: int) -> Image.Image:
    """알파를 흰 배경에 합성 → 정사각 패딩 → size x size.

    tagging.py 의 _prepare_image 와 같은 규약(정사각 패딩). DINOv2 기본 전처리는 center-crop 이라
    세로로 긴 일러스트가 잘려 구도 비교가 어긋난다. (224 = 16*14, DINOv2 patch=14)
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

    def __init__(self, repo: str = MODEL_REPO, device: str | None = None, batch_size: int = 16,
                 fast: bool = False, workers: int | None = None):
        self.repo = repo
        self.device = device
        self.batch_size = batch_size
        self.fast = fast
        self.workers = workers if workers else _default_workers()
        self._model = None
        self._proc = None
        self._size = 224
        self._dtype = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fast 모드에서는 fast(텐서) 이미지 프로세서도 쓴다. 우리는 do_resize/center_crop 을
        # 끄고 미리 224 정사각으로 맞춰 넣으므로 정규화만 남아 값 차이는 무시할 수준이고,
        # 명시 지정하면 use_fast 경고도 사라진다. full 모드는 기존과 동일한 slow 프로세서.
        self._proc = AutoImageProcessor.from_pretrained(self.repo, use_fast=self.fast)
        model = AutoModel.from_pretrained(self.repo)
        # half precision 은 GPU 에서만 의미가 있다(CPU 는 fp32 로 폴백).
        if self.fast and self.device == "cuda":
            self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = model.to(self.device, dtype=self._dtype)
        else:
            self._dtype = torch.float32
            model = model.to(self.device)
        self._model = model.eval()
        tag = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[self._dtype]
        print(f"[dinov2] {self.repo} on {self.device} ({tag}, load×{self.workers}, "
              f"hidden={self._model.config.hidden_size})")

    def _forward(self, images: list) -> np.ndarray:
        import torch

        inputs = self._proc(
            images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
        ).to(self.device)
        if self._dtype != torch.float32:
            inputs = {k: (v.to(self._dtype) if torch.is_floating_point(v) else v)
                      for k, v in inputs.items()}
        hidden = self._model(**inputs).last_hidden_state
        cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
        return cls.float().cpu().numpy()

    def embed(self, paths: list[Path], progress=None, on_ready=None) -> np.ndarray:
        """on_ready: 모델 로딩이 끝나고 실제 임베딩이 시작되는 시점에 불린다.

        병목은 PIL 디코드/리사이즈(CPU)라 스레드풀로 병렬 로딩하면서 준비되는 대로 배치를 GPU 에
        흘려보낸다(로딩·forward 겹침). map 은 입력 순서를 보존하므로 결과 순서는 순차 로딩과 같다.
        """
        self._load()
        if on_ready:
            on_ready()
        import torch

        dim = self._model.config.hidden_size
        out = np.zeros((len(paths), dim), dtype=np.float32)
        if not len(paths):
            return out
        with torch.inference_mode(), ThreadPoolExecutor(max_workers=self.workers) as pool:
            loaded = pool.map(lambda p: load_square_rgb(p, self._size), paths)
            batch, done = [], 0
            for img in loaded:
                batch.append(img)
                if len(batch) >= self.batch_size:
                    out[done : done + len(batch)] = self._forward(batch)
                    done += len(batch)
                    batch = []
                    if progress:
                        progress(done, len(paths))
            if batch:
                out[done : done + len(batch)] = self._forward(batch)
                done += len(batch)
                if progress:
                    progress(done, len(paths))
        return out


_EMBEDDER: Dinov2Embedder | None = None


def get_embedder(batch_size: int = 16, fast: bool = False) -> Dinov2Embedder:
    """프로세스당 하나만 만든다(데이터셋 전환마다 모델 재로딩 방지). fast 가 바뀌면 재생성."""
    global _EMBEDDER
    if _EMBEDDER is None or _EMBEDDER.fast != fast:
        _EMBEDDER = Dinov2Embedder(batch_size=batch_size, fast=fast)
    _EMBEDDER.batch_size = batch_size
    return _EMBEDDER


def _sig(path: Path) -> list:
    st = path.stat()
    return [int(st.st_mtime), st.st_size]


def _read_cache(raw_dir: Path, sigs: dict[str, list]) -> dict[str, np.ndarray]:
    cache = cache_dir(raw_dir) / "embeddings.npz"
    if not cache.exists():
        return {}
    try:
        blob = np.load(cache, allow_pickle=False)
        # 다른 정밀도(fast/full)로 만든 임베딩은 섞으면 안 된다 → 전량 무효 취급.
        # (구형 캐시는 mode 키가 없다 = fp32 = "full")
        stored_mode = str(blob["mode"]) if "mode" in blob.files else "full"
        if stored_mode != _cache_mode():
            return {}
        out = {}
        for i, k in enumerate(blob["keys"]):
            k = str(k)
            if k in sigs and list(blob["sigs"][i]) == sigs[k]:
                out[k] = blob["emb"][i]
        return out
    except Exception as exc:
        print(f"[warn] 임베딩 캐시 무시: {exc}")
        return {}


def pending_count(raw_dir: Path, rels: list[PurePosixPath]) -> int:
    sigs = {str(r): _sig(raw_dir / r) for r in rels}
    return len(rels) - len(_read_cache(raw_dir, sigs))


def cached_embeddings(raw_dir: Path, rels: list[PurePosixPath]) -> np.ndarray | None:
    """전부 캐시돼 있으면 rels 순으로 스택, 하나라도 없으면 None."""
    sigs = {str(r): _sig(raw_dir / r) for r in rels}
    cached = _read_cache(raw_dir, sigs)
    if len(cached) != len(rels):
        return None
    return np.stack([cached[str(r)] for r in rels]).astype(np.float32)


def embedding_map(raw_dir: Path, rels: list[PurePosixPath]) -> dict[str, np.ndarray] | None:
    """{rel_str: 정규화 임베딩}. 하나라도 캐시 없으면 None."""
    emb = cached_embeddings(raw_dir, rels)
    if emb is None:
        return None
    return {str(r): emb[i] for i, r in enumerate(rels)}


def build_embeddings(raw_dir, rels, *, force=False, batch_size=16, progress=None, on_ready=None):
    """임베딩 캐시(.dedup/<name>/embeddings.npz)를 채우고 전체를 반환한다."""
    keys = [str(r) for r in rels]
    sigs = {k: _sig(raw_dir / k) for k in keys}
    cached = {} if force else _read_cache(raw_dir, sigs)

    todo = [k for k in keys if k not in cached]
    if todo:
        print(f"[dinov2] 임베딩 {len(todo)}장 ({_cache_mode()} 모드, 캐시 재사용 {len(cached)}장)")
        emb = get_embedder(batch_size, fast=FAST).embed(
            [raw_dir / k for k in todo], progress, on_ready)
        for i, k in enumerate(todo):
            cached[k] = emb[i]
    else:
        print(f"[dinov2] 캐시 재사용 {len(cached)}장 ({_cache_mode()} 모드)")

    stacked = np.stack([cached[k] for k in keys]).astype(np.float32)
    np.savez(
        cache_dir(raw_dir) / "embeddings.npz",
        keys=np.array(keys),
        sigs=np.array([sigs[k] for k in keys], dtype=np.int64),
        emb=stacked,
        mode=np.array(_cache_mode()),
    )
    return stacked


# ---------------------------------------------------------------------------
# 상태 (그룹핑 / 제외 / 강조) — .dedup/<name>/state.json
# ---------------------------------------------------------------------------
@dataclass
class State:
    groups: list[dict] = field(default_factory=list)   # [{"members":[rel,...], "special":bool}]
    excluded: list[str] = field(default_factory=list)
    cannot_link: set = field(default_factory=set)      # {frozenset({relA,relB})}


def state_path(raw_dir: Path) -> Path:
    return cache_dir(raw_dir) / "state.json"


def _sort_groups(state: State) -> None:
    for g in state.groups:
        g["members"].sort()
    state.groups.sort(key=lambda g: (-len(g["members"]), g["members"][0]))
    state.excluded.sort()


def load_state(raw_dir: Path, rels: list[PurePosixPath]) -> State:
    """state.json 을 읽어 현재 raw 이미지 목록과 정합화한다.

    - 저장돼 있지 않은(새로 추가된) 이미지는 싱글턴 그룹으로.
    - 사라진(삭제된) 이미지는 그룹/제외/cannot_link 에서 제거.
    저장은 비-기본 그룹(멤버>1 또는 강조)만 하므로, 나머지는 여기서 싱글턴으로 복원한다.
    """
    rawset = {str(r) for r in rels}
    p = state_path(raw_dir)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] state.json 무시(새로 시작): {exc}")

    excluded = [r for r in data.get("excluded", []) if r in rawset]
    exset = set(excluded)

    groups: list[dict] = []
    assigned = set(exset)
    for g in data.get("groups", []):
        mem = [r for r in g.get("members", []) if r in rawset and r not in assigned]
        if not mem:
            continue
        groups.append({"members": mem, "special": bool(g.get("special", False))})
        assigned.update(mem)

    for r in sorted(rawset):
        if r not in assigned:
            groups.append({"members": [r], "special": False})
            assigned.add(r)

    cannot = set()
    for pair in data.get("cannot_link", []):
        if len(pair) == 2 and pair[0] in rawset and pair[1] in rawset and pair[0] != pair[1] \
                and pair[0] not in exset and pair[1] not in exset:
            cannot.add(frozenset(pair))

    state = State(groups=groups, excluded=excluded, cannot_link=cannot)
    _sort_groups(state)
    return state


def save_state(raw_dir: Path, state: State) -> None:
    _sort_groups(state)
    # 싱글턴 기본 그룹은 저장하지 않는다(로드 때 복원). 멤버>1 또는 강조만.
    keep = [g for g in state.groups if len(g["members"]) > 1 or g["special"]]
    data = {
        "version": STATE_VERSION,
        "dataset": raw_dir.name,
        "groups": [{"members": g["members"], "special": g["special"]} for g in keep],
        "excluded": state.excluded,
        "cannot_link": [sorted(p) for p in state.cannot_link],
    }
    state_path(raw_dir).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 그룹 조회 / centroid / 후보
# ---------------------------------------------------------------------------
def _gidx_by_rel(state: State) -> dict[str, int]:
    out = {}
    for i, g in enumerate(state.groups):
        for m in g["members"]:
            out[m] = i
    return out


def group_index(state: State, rel: str) -> int:
    for i, g in enumerate(state.groups):
        if rel in g["members"]:
            return i
    raise KeyError(rel)


def centroids(state: State, emb: dict[str, np.ndarray]) -> np.ndarray:
    """그룹별 centroid(멤버 임베딩 평균을 L2 정규화). (G, D)."""
    vecs = []
    for g in state.groups:
        v = np.mean([emb[m] for m in g["members"]], axis=0)
        n = np.linalg.norm(v)
        vecs.append(v / n if n > 0 else v)
    return np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 1), np.float32)


def _representative(members: list[str], emb: dict[str, np.ndarray]) -> str:
    """centroid 에 가장 가까운 멤버 (그룹 대표 썸네일)."""
    if len(members) == 1:
        return members[0]
    c = np.mean([emb[m] for m in members], axis=0)
    return max(members, key=lambda m: float(emb[m] @ c))


EMPTY: frozenset = frozenset()


def _forbid_map(state: State) -> dict[str, set]:
    f: dict[str, set] = defaultdict(set)
    for pair in state.cannot_link:
        a, b = tuple(pair)
        f[a].add(b)
        f[b].add(a)
    return f


def _blocked(gi: dict, gj: dict, forbid: dict[str, set]) -> bool:
    """두 그룹 사이에 cannot_link 가 하나라도 있으면 병합 후보에서 제외."""
    mj = set(gj["members"])
    return any(forbid.get(a, EMPTY) & mj for a in gi["members"])


def candidate_pairs(state: State, emb: dict[str, np.ndarray], threshold: float,
                    limit: int = 300) -> list[dict]:
    """centroid 코사인 >= threshold 인 그룹 쌍 후보 (cannot_link 제외), 유사도 내림차순."""
    n = len(state.groups)
    if n < 2:
        return []
    C = centroids(state, emb)
    sim = C @ C.T
    forbid = _forbid_map(state)
    out = []
    for i in range(n):
        gi = state.groups[i]
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s < threshold:
                continue
            gj = state.groups[j]
            if _blocked(gi, gj, forbid):
                continue
            out.append((s, i, j))
    out.sort(key=lambda t: -t[0])
    out = out[:limit]

    cards = []
    for s, i, j in out:
        gi, gj = state.groups[i], state.groups[j]
        cards.append({
            "a": _representative(gi["members"], emb),
            "b": _representative(gj["members"], emb),
            "sim": s,
            "size_a": len(gi["members"]), "size_b": len(gj["members"]),
            "special_a": gi["special"], "special_b": gj["special"],
        })
    return cards


def group_view(state: State, emb: dict[str, np.ndarray] | None) -> list[dict]:
    """UI 표시용 그룹 목록. emb 있으면 대표 멤버를 centroid 기준으로 고른다."""
    view = []
    for i, g in enumerate(state.groups):
        rep = _representative(g["members"], emb) if emb else g["members"][0]
        view.append({
            "id": i,
            "rep": rep,
            "members": g["members"],
            "size": len(g["members"]),
            "special": g["special"],
        })
    return view


# ---------------------------------------------------------------------------
# 액션 (모두 rel 기준 — 그룹 id 는 매 조회마다 바뀔 수 있으므로)
# ---------------------------------------------------------------------------
def _drop_cannot_within(state: State, members: set) -> None:
    """같은 그룹 안으로 들어온 cannot_link 쌍은 무의미하므로 제거."""
    state.cannot_link = {
        p for p in state.cannot_link if not (set(p) <= members)
    }


def merge(state: State, rel_a: str, rel_b: str) -> None:
    """rel_a 의 그룹과 rel_b 의 그룹을 합친다(사용자 명시 병합)."""
    i, j = group_index(state, rel_a), group_index(state, rel_b)
    if i == j:
        return
    gi, gj = state.groups[i], state.groups[j]
    merged = {"members": gi["members"] + gj["members"],
              "special": gi["special"] or gj["special"]}
    state.groups = [g for k, g in enumerate(state.groups) if k not in (i, j)]
    state.groups.append(merged)
    _drop_cannot_within(state, set(merged["members"]))


def mark_different(state: State, rel_a: str, rel_b: str) -> None:
    """rel_a 그룹과 rel_b 그룹은 동일 그룹이 아니라고 명시(cannot_link). 다른 그룹일 때만."""
    i, j = group_index(state, rel_a), group_index(state, rel_b)
    if i == j:
        return
    for a in state.groups[i]["members"]:
        for b in state.groups[j]["members"]:
            state.cannot_link.add(frozenset((a, b)))


def ungroup(state: State, rel: str) -> None:
    """rel 이 속한 그룹을 싱글턴들로 해체(강조도 해제)."""
    i = group_index(state, rel)
    members = state.groups[i]["members"]
    state.groups.pop(i)
    for m in members:
        state.groups.append({"members": [m], "special": False})


def remove_member(state: State, rel: str) -> None:
    """rel 을 자기 그룹에서 빼내 싱글턴으로. (그룹이 1장뿐이면 no-op)"""
    i = group_index(state, rel)
    g = state.groups[i]
    if len(g["members"]) == 1:
        return
    g["members"] = [m for m in g["members"] if m != rel]
    state.groups.append({"members": [rel], "special": False})


def set_special(state: State, rel: str, on: bool) -> None:
    state.groups[group_index(state, rel)]["special"] = bool(on)


def exclude(state: State, rel: str) -> None:
    """rel 을 그룹에서 빼고 제외 목록으로. cannot_link 도 정리."""
    i = group_index(state, rel)
    g = state.groups[i]
    g["members"] = [m for m in g["members"] if m != rel]
    if not g["members"]:
        state.groups.pop(i)
    if rel not in state.excluded:
        state.excluded.append(rel)
    state.cannot_link = {p for p in state.cannot_link if rel not in p}


def include(state: State, rel: str) -> None:
    """제외를 취소하고 싱글턴 그룹으로 복귀."""
    if rel in state.excluded:
        state.excluded.remove(rel)
        state.groups.append({"members": [rel], "special": False})


# ---------------------------------------------------------------------------
# 익스포트 → dataset/<name>/repeat_<N>/ 심볼릭 링크
# ---------------------------------------------------------------------------
def round_repeat(total: int, k: int, rounding: str) -> int:
    """그룹 전체 반복수 total 을 k 로 나눈 개별 반복수(최소 1)."""
    val = total / k
    n = math.ceil(val) if rounding == "ceil" else (round(val) if rounding == "round" else math.floor(val))
    return max(1, n)


def plan_export(state: State, repeat: int, rounding: str) -> tuple[list[tuple[str, int]], dict]:
    """반환: ([(rel, N)], stats). 제외 이미지는 빠진다. flat 이름 충돌은 stats['collisions'].

    그룹 전체 반복수 합 = R(강조=2R). 그룹 크기 k 면 각 멤버 round(total/k).
    """
    assignments: list[tuple[str, int]] = []
    n_groups = n_singletons = n_multi = n_special = 0
    exset = set(state.excluded)
    for g in state.groups:
        members = [m for m in g["members"] if m not in exset]
        if not members:
            continue
        k = len(members)
        total = repeat * 2 if g["special"] else repeat
        n = round_repeat(total, k, rounding)
        for m in members:
            assignments.append((m, n))
        n_groups += 1
        if k > 1:
            n_multi += 1
        else:
            n_singletons += 1
        if g["special"]:
            n_special += 1

    flats: dict[str, list[str]] = defaultdict(list)
    for rel, _n in assignments:
        flats[flat_name(rel)].append(rel)
    collisions = {f: rels for f, rels in flats.items() if len(rels) > 1}

    dist: dict[int, int] = defaultdict(int)
    for _rel, n in assignments:
        dist[n] += 1
    stats = {
        "images": len(assignments),
        "groups": n_groups,
        "singletons": n_singletons,
        "multi": n_multi,
        "special": n_special,
        "excluded": len(exset),
        "dist": dict(sorted(dist.items())),
        "total_reps": sum(n for _rel, n in assignments),
        "collisions": collisions,
    }
    return assignments, stats


def _clear_repeat_dirs(out_dir: Path) -> None:
    for sub in out_dir.iterdir():
        if sub.is_dir() and re.fullmatch(r"repeat_\d+", sub.name):
            for f in sub.iterdir():
                if f.is_symlink() or f.is_file():
                    f.unlink()
            sub.rmdir()


def do_export(raw_dir: Path, state: State, name: str, repeat: int, rounding: str,
              *, force: bool = False, absolute: bool = False) -> dict:
    """dataset/<name>/repeat_<N>/ 에 상대 심볼릭 링크를 만든다. 반환: 결과 stats."""
    assignments, stats = plan_export(state, repeat, rounding)
    if not assignments:
        raise ValueError("익스포트할 이미지가 없다(전부 제외됨?).")
    if stats["collisions"]:
        first = next(iter(stats["collisions"]))
        raise ValueError(f"플래트닝 이름 충돌: {first} ({len(stats['collisions'])}건). 원시 파일명을 정리할 것.")

    out_dir = DATASET_ROOT / name
    if out_dir.exists():
        has_repeat = any(re.fullmatch(r"repeat_\d+", p.name)
                         for p in out_dir.iterdir() if p.is_dir())
        if has_repeat and not force:
            raise ValueError(f"이미 존재(force 필요): dataset/{name}")
        if has_repeat:
            _clear_repeat_dirs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for rel, n in assignments:
        sub = out_dir / f"repeat_{n}"
        sub.mkdir(exist_ok=True)
        link = sub / flat_name(rel)
        src = resolve_rel(raw_dir, rel)
        target = src if absolute else Path(os.path.relpath(src, sub))
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(target, link)
        created += 1

    stats["created"] = created
    stats["out_dir"] = str(out_dir)
    stats["name"] = name
    return stats
