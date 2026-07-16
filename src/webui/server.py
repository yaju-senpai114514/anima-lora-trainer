"""데이터셋 프로세싱 웹 서버 (공용 로직). 프론트엔드는 static/index.html.

구 0/1/2 단계(dedup 그룹핑/익스포트 · WD14 태깅 · toml 생성)를 하나의 서버로 병합했다.
루트 스크립트 0_dataset_server.py 가 얇은 CLI 로 감싼다.
원본(dataset_raw/)은 완전 read-only, dedup 현황은 .dedup/<name>/state.json 으로만 관리.
"""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import socketserver
import threading
import urllib.parse
import webbrowser
from pathlib import Path, PurePosixPath

from PIL import Image

from .. import configgen, dedup, tagging
from ..paths import BLACKLIST_PATH, DATASET_ROOT, DEDUP_CACHE_ROOT

THUMB_SIZE = 320
STATIC_DIR = Path(__file__).resolve().parent / "static"

# WD 추론 캐시(실경로 키 — 데이터셋을 넘나들며 공유)와 데이터셋별 마지막 태깅 설정.
WD_CACHE_PATH = DEDUP_CACHE_ROOT / "wd14_cache.json"
TAG_SETTINGS_PATH = DEDUP_CACHE_ROOT / "tag_settings.json"


# ---------------------------------------------------------------------------
# 상태 → UI 가 먹을 JSON
# ---------------------------------------------------------------------------
def _member(raw_dir: Path, rel: str) -> dict:
    """그룹/제외 타일용 멤버 정보 — 경로/파일크기 + 픽셀 해상도(_img_info 캐시)."""
    return {"rel": rel, "name": Path(rel).name, **_img_info(raw_dir, rel)}


# 픽셀 해상도는 PIL 헤더만 읽으면 되지만, 액션마다 state 를 재로딩하므로 캐시해 둔다.
_IMG_INFO_CACHE: dict[tuple, dict] = {}


def _img_info(raw_dir: Path, rel: str) -> dict:
    """후보 쌍 비교용 파일 정보: {bytes, w, h}. 실패 시 0."""
    p = dedup.resolve_rel(raw_dir, rel)
    try:
        st = p.stat()
    except OSError:
        return {"bytes": 0, "w": 0, "h": 0}
    key = (str(p), int(st.st_mtime), st.st_size)
    info = _IMG_INFO_CACHE.get(key)
    if info is None:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        info = {"bytes": st.st_size, "w": w, "h": h}
        _IMG_INFO_CACHE[key] = info
    return info


def state_payload(raw_dir: Path, threshold: float, *, build: bool = False,
                  refresh: bool = False, batch_size: int = 16) -> dict:
    """현재 상태(그룹/싱글턴/제외) + 후보 쌍. 임베딩이 없으면 need_embed 를 돌려준다."""
    rels = dedup.collect_images(raw_dir)
    if not rels:
        return {"dataset": raw_dir.name, "error": f"{raw_dir} 아래에 이미지가 없다."}

    if build:
        def cli_progress(done, total):
            print(f"  {done}/{total}", end="\n" if done == total else "\r", flush=True)
        dedup.build_embeddings(raw_dir, rels, force=refresh,
                               batch_size=batch_size, progress=cli_progress)

    emb = dedup.embedding_map(raw_dir, rels)
    state = dedup.load_state(raw_dir, rels)

    if emb is None:
        return {
            "dataset": raw_dir.name,
            "need_embed": True,
            "pending": dedup.pending_count(raw_dir, rels),
            "total_images": len(rels),
            "threshold": threshold,
        }

    view = dedup.group_view(state, emb)
    groups = []
    for g in view:
        groups.append({
            "id": g["id"],
            "rep": g["rep"],
            "size": g["size"],
            "special": g["special"],
            "members": [_member(raw_dir, m) for m in g["members"]],
        })

    cands = dedup.candidate_pairs(state, emb, threshold)
    for c in cands:  # 해상도/파일크기 비교 + 제외 버튼용
        c["a_info"] = _img_info(raw_dir, c["a"])
        c["b_info"] = _img_info(raw_dir, c["b"])

    multi = [g for g in groups if g["size"] > 1]
    singles = [g for g in groups if g["size"] == 1]
    n_special = sum(1 for g in groups if g["special"])

    return {
        "dataset": raw_dir.name,
        "raw_dir": str(raw_dir),
        "threshold": threshold,
        "total_images": len(rels),
        "n_groups": len(groups),
        "n_multi": len(multi),
        "n_singletons": len(singles),
        "n_special": n_special,
        "n_excluded": len(state.excluded),
        "n_cannot_link": len(state.cannot_link),
        "groups": groups,
        "candidates": cands,
        "excluded": [_member(raw_dir, r) for r in state.excluded],
    }


def print_report(payload: dict) -> None:
    if payload.get("error"):
        print(f"[dedup] {payload['error']}")
        return
    if payload.get("need_embed"):
        print(f"[dedup] {payload['dataset']}: 임베딩 필요 {payload['pending']}장")
        return
    print(f"\n[dedup] {payload['dataset']}  images={payload['total_images']}  "
          f"threshold={payload['threshold']}")
    print(f"[dedup] 그룹 {payload['n_groups']}개 (묶임 {payload['n_multi']} · 싱글턴 {payload['n_singletons']}) "
          f"· 강조 {payload['n_special']} · 제외 {payload['n_excluded']} · cannot_link {payload['n_cannot_link']}")
    print(f"[dedup] 현재 임계값 후보 쌍 {len(payload['candidates'])}개")
    for c in payload["candidates"][:30]:
        sa = f"[{c['size_a']}]" if c["size_a"] > 1 else ""
        sb = f"[{c['size_b']}]" if c["size_b"] > 1 else ""
        print(f"    {c['sim']*100:5.1f}%   {c['a']}{sa}  ~  {c['b']}{sb}")
    print()


# ---------------------------------------------------------------------------
# 썸네일 (원시/데이터셋 공용 — 캐시 키는 심링크를 푼 실경로 기준)
# ---------------------------------------------------------------------------
def _thumb(src: Path, size: int, cache_dir: Path) -> bytes:
    real = src.resolve()
    st = real.stat()
    key = hashlib.sha1(f"{real}|{int(st.st_mtime)}|{st.st_size}|{size}".encode()).hexdigest()
    cached = cache_dir / f"{key}.jpg"
    if cached.exists():
        return cached.read_bytes()
    im = Image.open(src)
    im = im.convert("RGB") if im.mode != "RGB" else im
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    data = buf.getvalue()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def thumb_bytes(raw_dir: Path, rel: str, size: int) -> bytes:
    return _thumb(dedup.resolve_rel(raw_dir, rel), size, dedup.cache_dir(raw_dir) / "thumbs")


def ds_thumb_bytes(ds_dir: Path, rel: str, size: int) -> bytes:
    return _thumb(ds_dir / rel, size, DEDUP_CACHE_ROOT / "_dataset_thumbs")


# ---------------------------------------------------------------------------
# 데이터셋(dataset/) 조회 — 태깅/컨피그 대상
# ---------------------------------------------------------------------------
def list_processed_datasets() -> list[str]:
    if not DATASET_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in DATASET_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def processed_payload() -> dict:
    """dataset/ 아래 폴더별 이미지/캡션/repeat 구조/toml 현황 + 마지막 태깅 설정."""
    settings = _load_json(TAG_SETTINGS_PATH)
    items = []
    for name in list_processed_datasets():
        d = DATASET_ROOT / name
        imgs = tagging.list_images(d, tagging.has_repeat_subsets(d))
        repeats = sorted(
            int(p.name.split("_")[1]) for p in d.iterdir()
            if p.is_dir() and p.name.startswith("repeat_") and p.name.split("_")[1].isdigit()
        )
        items.append({
            "name": name,
            "images": len(imgs),
            "captions": sum(1 for i in imgs if i.with_suffix(".txt").exists()),
            "repeats": repeats,
            "has_toml": (DATASET_ROOT / f"{name}.toml").exists(),
            "last_tag": settings.get(name),
        })
    return {"datasets": items}


def captions_payload(ds_dir: Path) -> dict:
    imgs = tagging.list_images(ds_dir, tagging.has_repeat_subsets(ds_dir))
    items = []
    for img in imgs:
        txt = img.with_suffix(".txt")
        items.append({
            "rel": img.relative_to(ds_dir).as_posix(),
            "name": img.name,
            "caption": txt.read_text(encoding="utf-8") if txt.exists() else None,
        })
    return {"dataset": ds_dir.name, "items": items}


# 임베딩/태깅은 수십 초~수 분이 걸린다. 한 번에 하나만(GPU 직렬화) 백그라운드 스레드에서
# 돌리고 UI 는 /api/job 을 폴링한다. kind: "embed" | "tag".
JOB = {"running": False, "kind": None, "dataset": None, "done": 0, "total": 0,
       "stage": "model", "error": None, "summary": None}
JOB_LOCK = threading.Lock()


def run_embed_job(raw_dir: Path, refresh: bool, batch_size: int) -> None:
    try:
        rels = dedup.collect_images(raw_dir)
        total = len(rels) if refresh else dedup.pending_count(raw_dir, rels)
        with JOB_LOCK:
            JOB.update(running=True, kind="embed", dataset=raw_dir.name, done=0, total=total,
                       stage="model", error=None, summary=None)

        def progress(done, _total):
            with JOB_LOCK:
                JOB["done"] = done

        def on_ready():
            with JOB_LOCK:
                JOB["stage"] = "embed"

        dedup.build_embeddings(raw_dir, rels, force=refresh, batch_size=batch_size,
                               progress=progress, on_ready=on_ready)
        print(f"[embed] {raw_dir.name} 완료")
    except Exception as exc:
        print(f"[embed] {raw_dir.name} 실패: {exc}")
        with JOB_LOCK:
            JOB["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with JOB_LOCK:
            JOB["running"] = False


def run_tag_job(ds_dir: Path, opts: dict) -> None:
    try:
        banned = tagging.load_blacklist(BLACKLIST_PATH)
        total = len(tagging.list_images(ds_dir, tagging.has_repeat_subsets(ds_dir)))
        with JOB_LOCK:
            JOB.update(running=True, kind="tag", dataset=ds_dir.name, done=0, total=total,
                       stage="model", error=None, summary=None)

        def progress(done, _total):
            with JOB_LOCK:
                JOB["done"] = done

        def on_ready():
            with JOB_LOCK:
                JOB["stage"] = "tag"

        stats = tagging.tag_folder(
            ds_dir, opts["trigger"], banned,
            general_threshold=float(opts.get("general_threshold", 0.35)),
            character_threshold=float(opts.get("character_threshold", 0.85)),
            include_rating=bool(opts.get("include_rating")),
            overwrite=bool(opts.get("overwrite")),
            cache_path=WD_CACHE_PATH,
            progress=progress, on_ready=on_ready,
        )
        summary = (f"태깅 {stats['processed']}장 완료 (WD 캐시 {stats['cached']} · "
                   f"스킵 {stats['skipped']} · 실패 {stats['failed']})")
        with JOB_LOCK:
            JOB["summary"] = summary
        print(f"[tag] {ds_dir.name}: {summary}")
        top = sorted(stats["removed"].items(), key=lambda kv: kv[1], reverse=True)[:10]
        if top:
            print("[tag] blacklist 제거 상위: " + ", ".join(f"{t}×{n}" for t, n in top))
    except Exception as exc:
        print(f"[tag] {ds_dir.name} 실패: {exc}")
        with JOB_LOCK:
            JOB["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with JOB_LOCK:
            JOB["running"] = False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def make_handler(default_ds: str | None, threshold: float, batch_size: int):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

        def _ds(self, name: str | None) -> Path:
            """데이터셋 이름 → raw_dir. dataset_raw/ 의 실제 목록에 있는 이름만 받으므로
            경로 조작(../ 등)은 통과하지 못한다."""
            if not name:
                raise PermissionError("데이터셋이 지정되지 않았다.")
            if name not in dedup.list_raw_datasets():
                raise PermissionError(f"알 수 없는 데이터셋: {name}")
            return (dedup.RAW_ROOT / name).resolve()

        def _safe(self, raw_dir: Path, rel: str) -> Path:
            """rel 을 실제 경로로 풀되 raw_dir 밖으로 나가면 거부한다(원본은 read-only)."""
            p = dedup.resolve_rel(raw_dir, rel).resolve()
            root = raw_dir.resolve()
            if not (p == root or root in p.parents):
                raise PermissionError(f"경로 거부: {rel}")
            return p

        def _proc(self, name: str | None) -> Path:
            """데이터셋 이름 → dataset/<name>. 실제 목록에 있는 이름만 받는다."""
            if not name:
                raise PermissionError("데이터셋이 지정되지 않았다.")
            if name not in list_processed_datasets():
                raise PermissionError(f"알 수 없는 데이터셋: {name}")
            return DATASET_ROOT / name

        @staticmethod
        def _safe_ds_rel(ds_dir: Path, rel: str) -> Path:
            """dataset/<name>/ 안의 rel. 이미지는 raw 로의 심링크라 resolve() 가 밖을
            가리키는 게 정상이므로, 검사는 **어휘적으로만** 한다(.. / 절대경로 거부)."""
            pp = PurePosixPath(rel)
            if pp.is_absolute() or any(part in ("..", "") for part in pp.parts) or not pp.parts:
                raise PermissionError(f"경로 거부: {rel}")
            return ds_dir / rel

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        # -- 액션 공통: state 로드 → 변형 → 저장 -----------------------------
        def _mutate(self, raw_dir: Path, fn) -> None:
            rels = dedup.collect_images(raw_dir)
            state = dedup.load_state(raw_dir, rels)
            fn(state)
            dedup.save_state(raw_dir, state)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            try:
                if url.path == "/":
                    # 매 요청마다 파일에서 읽는다 — 로컬 툴이라 비용이 없고, HTML 수정이
                    # 서버 재시작 없이 반영된다.
                    html = (STATIC_DIR / "index.html").read_bytes()
                    return self._send(200, html, "text/html; charset=utf-8")

                if url.path == "/api/datasets":
                    items = []
                    for name in dedup.list_raw_datasets():
                        d = (dedup.RAW_ROOT / name).resolve()
                        rels = dedup.collect_images(d)
                        items.append({
                            "name": name,
                            "images": len(rels),
                            "pending": dedup.pending_count(d, rels) if rels else 0,
                        })
                    return self._json({"datasets": items, "default": default_ds,
                                       "threshold": threshold})

                if url.path == "/api/job":
                    with JOB_LOCK:
                        return self._json(dict(JOB))

                if url.path == "/api/state":
                    raw_dir = self._ds(qs.get("ds", [None])[0])
                    thr = float(qs.get("threshold", [threshold])[0])
                    return self._json(state_payload(raw_dir, thr, batch_size=batch_size))

                if url.path in ("/thumb", "/full"):
                    raw_dir = self._ds(qs.get("ds", [None])[0])
                    rel = qs.get("rel", [""])[0]
                    self._safe(raw_dir, rel)
                    size = int(qs.get("s", [THUMB_SIZE])[0]) if url.path == "/thumb" else 1400
                    return self._send(200, thumb_bytes(raw_dir, rel, size), "image/jpeg")

                # -- 데이터셋(dataset/) 쪽: 태깅/컨피그 대상 -------------------
                if url.path == "/api/processed":
                    return self._json(processed_payload())

                if url.path == "/api/captions":
                    ds_dir = self._proc(qs.get("name", [None])[0])
                    return self._json(captions_payload(ds_dir))

                if url.path == "/api/toml":
                    ds_dir = self._proc(qs.get("name", [None])[0])
                    p = DATASET_ROOT / f"{ds_dir.name}.toml"
                    return self._json({
                        "name": ds_dir.name,
                        "exists": p.exists(),
                        "text": p.read_text(encoding="utf-8") if p.exists() else "",
                    })

                if url.path in ("/dsthumb", "/dsfull"):
                    ds_dir = self._proc(qs.get("name", [None])[0])
                    rel = qs.get("rel", [""])[0]
                    self._safe_ds_rel(ds_dir, rel)
                    size = int(qs.get("s", [THUMB_SIZE])[0]) if url.path == "/dsthumb" else 1400
                    return self._send(200, ds_thumb_bytes(ds_dir, rel, size), "image/jpeg")

                return self._send(404, b"not found", "text/plain")
            except PermissionError as exc:
                return self._json({"error": str(exc)}, 403)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            try:
                body = self._body()

                # -- 데이터셋(dataset/) 쪽: 태깅 잡 / 캡션 수정 / toml 생성 -----
                if url.path == "/api/tag":
                    ds_dir = self._proc(body.get("name"))
                    trigger = str(body.get("trigger") or "").strip()
                    if not trigger:
                        return self._json({"error": "trigger 가 필요하다."}, 400)
                    with JOB_LOCK:
                        if JOB["running"]:
                            return self._json({"error": f"이미 작업 중: {JOB['kind']} {JOB['dataset']}"}, 409)
                        JOB.update(running=True, kind="tag", dataset=ds_dir.name,
                                   done=0, total=0, error=None, summary=None)
                    # 마지막 태깅 설정 저장 → 다음 태깅/컨피그 다이얼로그 프리필 (rating→keep_tokens=2)
                    settings = _load_json(TAG_SETTINGS_PATH)
                    settings[ds_dir.name] = {
                        "trigger": trigger,
                        "general_threshold": float(body.get("general_threshold", 0.35)),
                        "character_threshold": float(body.get("character_threshold", 0.85)),
                        "include_rating": bool(body.get("include_rating")),
                    }
                    TAG_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    TAG_SETTINGS_PATH.write_text(
                        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
                    opts = dict(settings[ds_dir.name], overwrite=bool(body.get("overwrite")))
                    threading.Thread(target=run_tag_job, args=(ds_dir, opts), daemon=True).start()
                    return self._json({"started": True, "dataset": ds_dir.name})

                if url.path == "/api/caption":
                    ds_dir = self._proc(body.get("name"))
                    img = self._safe_ds_rel(ds_dir, str(body["rel"]))
                    txt = img.with_suffix(".txt")
                    text = str(body.get("text", "")).strip()
                    if text:
                        txt.write_text(text, encoding="utf-8")
                    elif txt.exists():
                        txt.unlink()  # 빈 캡션 저장 = 캡션 삭제
                    return self._json({"ok": True, "caption": text or None})

                if url.path == "/api/save-toml":
                    ds_dir = self._proc(body.get("name"))
                    text = str(body.get("text", ""))
                    if not text.strip():
                        return self._json({"error": "toml 내용이 비어 있다."}, 400)
                    import toml as toml_lib
                    try:
                        parsed = toml_lib.loads(text)
                    except Exception as exc:
                        return self._json({"error": f"toml 파싱 실패: {exc}"}, 400)
                    # sd-scripts 는 toml 의 batch_size 가 --train_batch_size 를 이긴다 — 경고만.
                    warning = None
                    levels = [parsed.get("general", {})] + list(parsed.get("datasets", []))
                    if any(isinstance(v, dict) and "batch_size" in v for v in levels):
                        warning = ("batch_size 가 toml 에 있다 — --train_batch_size 를 이기므로 "
                                   "학습 스크립트의 GPU 수 검증에 걸릴 수 있다 (빼는 것을 권장).")
                    out = DATASET_ROOT / f"{ds_dir.name}.toml"
                    out.write_text(text, encoding="utf-8")
                    print(f"[config] wrote {out} (편집기 저장)")
                    return self._json({"ok": True, "path": str(out), "warning": warning})

                if url.path == "/api/make-config":
                    ds_dir = self._proc(body.get("name"))
                    try:
                        resolution = configgen.parse_resolution(str(body.get("resolution", "1024")))
                        info = configgen.make_config(
                            ds_dir.name,
                            resolution=resolution,
                            num_repeats=int(body.get("num_repeats", 2)),
                            keep_tokens=int(body.get("keep_tokens", 1)),
                            shuffle_caption=bool(body.get("shuffle_caption", True)),
                            enable_bucket=bool(body.get("enable_bucket", True)),
                            bucket_no_upscale=bool(body.get("bucket_no_upscale")),
                            min_bucket_reso=int(body.get("min_bucket_reso", 512)),
                            max_bucket_reso=int(body.get("max_bucket_reso", 1536)),
                        )
                    except ValueError as exc:
                        return self._json({"error": str(exc)}, 400)
                    if body.get("dry_run"):
                        return self._json({"dry_run": True, "info": info})
                    out = Path(info["out_path"])
                    if out.exists() and not body.get("force"):
                        return self._json({"error": f"이미 존재(덮어쓰기 체크 필요): dataset/{out.name}"}, 400)
                    out.write_text(info["toml"], encoding="utf-8")
                    print(f"[config] wrote {out}")
                    return self._json({"ok": True, "info": info})

                raw_dir = self._ds(body.get("dataset"))

                if url.path == "/api/embed":
                    with JOB_LOCK:
                        if JOB["running"]:
                            return self._json({"error": f"이미 작업 중: {JOB['kind']} {JOB['dataset']}"}, 409)
                        JOB.update(running=True, kind="embed", dataset=raw_dir.name,
                                   done=0, total=0, error=None, summary=None)
                    threading.Thread(
                        target=run_embed_job,
                        args=(raw_dir, bool(body.get("refresh")), batch_size),
                        daemon=True,
                    ).start()
                    return self._json({"started": True, "dataset": raw_dir.name})

                # -- 그룹핑 액션 (전부 rel 기준) --------------------------------
                if url.path == "/api/merge":
                    a, b = str(body["a"]), str(body["b"])
                    self._safe(raw_dir, a); self._safe(raw_dir, b)
                    self._mutate(raw_dir, lambda s: dedup.merge(s, a, b))
                    return self._json({"ok": True})

                if url.path == "/api/different":
                    a, b = str(body["a"]), str(body["b"])
                    self._safe(raw_dir, a); self._safe(raw_dir, b)
                    self._mutate(raw_dir, lambda s: dedup.mark_different(s, a, b))
                    return self._json({"ok": True})

                if url.path == "/api/ungroup":
                    rel = str(body["rel"]); self._safe(raw_dir, rel)
                    self._mutate(raw_dir, lambda s: dedup.ungroup(s, rel))
                    return self._json({"ok": True})

                if url.path == "/api/remove-member":
                    rel = str(body["rel"]); self._safe(raw_dir, rel)
                    self._mutate(raw_dir, lambda s: dedup.remove_member(s, rel))
                    return self._json({"ok": True})

                if url.path == "/api/exclude":
                    rel = str(body["rel"]); self._safe(raw_dir, rel)
                    self._mutate(raw_dir, lambda s: dedup.exclude(s, rel))
                    return self._json({"ok": True})

                if url.path == "/api/include":
                    rel = str(body["rel"]); self._safe(raw_dir, rel)
                    self._mutate(raw_dir, lambda s: dedup.include(s, rel))
                    return self._json({"ok": True})

                if url.path == "/api/special":
                    rel = str(body["rel"]); on = bool(body.get("on"))
                    self._safe(raw_dir, rel)
                    self._mutate(raw_dir, lambda s: dedup.set_special(s, rel, on))
                    return self._json({"ok": True})

                if url.path == "/api/export":
                    rels = dedup.collect_images(raw_dir)
                    state = dedup.load_state(raw_dir, rels)
                    name = str(body.get("name") or raw_dir.name)
                    repeat = int(body.get("repeat", 10))
                    rounding = str(body.get("rounding", "ceil"))
                    if body.get("dry_run"):
                        _assign, stats = dedup.plan_export(state, repeat, rounding)
                        stats["collisions"] = {k: v for k, v in stats["collisions"].items()}
                        return self._json({"dry_run": True, "stats": stats})
                    try:
                        stats = dedup.do_export(
                            raw_dir, state, name, repeat, rounding,
                            force=bool(body.get("force")), absolute=bool(body.get("absolute")))
                    except ValueError as exc:
                        return self._json({"error": str(exc)}, 400)
                    print(f"[export] dataset/{name}: {stats['created']}개 링크")
                    return self._json({"ok": True, "stats": stats})

                return self._send(404, b"not found", "text/plain")
            except KeyError as exc:
                return self._json({"error": f"필드 누락: {exc}"}, 400)
            except PermissionError as exc:
                return self._json({"error": str(exc)}, 403)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host: str, port: int, *, default_ds: str | None, threshold: float,
          batch_size: int, open_browser: bool = False) -> None:
    """웹 UI 서버를 띄우고 Ctrl-C 까지 블록한다."""
    handler = make_handler(default_ds, threshold, batch_size)
    with _Server((host, port), handler) as httpd:
        url = f"http://{host}:{port}"
        print(f"[serve] {url}  (Ctrl-C 로 종료)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] 종료")
