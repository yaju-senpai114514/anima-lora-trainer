#!/usr/bin/env python3
"""[0] dataset_raw/<원시> 를 DINOv2 로 임베딩해 웹 UI 에서 그룹핑/제외/강조하고,
가공된 심볼릭 링크 데이터셋(dataset/<name>/repeat_<N>/)까지 한 번에 익스포트한다.

파이프라인의 0단계 — 원시(중첩) 데이터셋을 학습용 dataset/ 으로 만드는 진입점이다.
원본 파일은 **완전 read-only** 다 — 리네임/이동/삭제하지 않는다. 그룹핑/제외/강조 현황은
`.dedup/<name>/state.json` 메타데이터로만 관리한다(dedup.py). 다음 단계는 1_tag_dataset.py.

워크플로우:
  1. 데이터셋 선택 → DINOv2 임베딩(캐시).
  2. 기본적으로 모든 이미지는 개별 그룹(싱글턴).
  3. 임계값(threshold)을 천천히 낮춰가며 뜨는 **후보 쌍**을 확인해서
       · [병합] 같은 그림(차분) → 한 그룹으로. 그룹은 centroid 로 다시 클러스터링된다.
       · [다르다] 별개다 → cannot_link 로 영구히 후보에서 제외.
     (자동 병합은 하지 않는다. 연쇄 병합 방지.)
  4. 그룹/싱글턴 단위로 강조(2R) 마킹, 중복/불량 이미지는 제외.
  5. 익스포트: repeat/rounding 지정 → dataset/<name>/repeat_<N>/ 심볼릭 링크 생성.
       · 그룹 전체 반복수 합 = R (강조 = 2R). 그룹 크기 k 면 각 멤버 round(total/k).

사용 예:
    uv run python 0_dedup_raw.py                 # 그냥 띄운다. 전부 웹 UI 에서.
    uv run python 0_dedup_raw.py mychar          # 미리 임베딩 + 요약 출력 후 UI
    uv run python 0_dedup_raw.py mychar --print   # 서버 없이 현황만 출력
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import json
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

from PIL import Image

import dedup

THUMB_SIZE = 320


# ---------------------------------------------------------------------------
# 상태 → UI 가 먹을 JSON
# ---------------------------------------------------------------------------
def _member(raw_dir: Path, rel: str) -> dict:
    p = dedup.resolve_rel(raw_dir, rel)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    return {"rel": rel, "name": Path(rel).name, "bytes": size}


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
# 썸네일
# ---------------------------------------------------------------------------
def thumb_bytes(raw_dir: Path, rel: str, size: int) -> bytes:
    src = dedup.resolve_rel(raw_dir, rel)
    st = src.stat()
    key = hashlib.sha1(f"{rel}|{int(st.st_mtime)}|{st.st_size}|{size}".encode()).hexdigest()
    cached = dedup.cache_dir(raw_dir) / "thumbs" / f"{key}.jpg"
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


# 임베딩은 수십 초가 걸린다. 백그라운드 스레드에서 돌리고 UI 는 /api/job 을 폴링한다.
JOB = {"running": False, "dataset": None, "done": 0, "total": 0, "stage": "model", "error": None}
JOB_LOCK = threading.Lock()


def run_embed_job(raw_dir: Path, refresh: bool, batch_size: int) -> None:
    try:
        rels = dedup.collect_images(raw_dir)
        total = len(rels) if refresh else dedup.pending_count(raw_dir, rels)
        with JOB_LOCK:
            JOB.update(running=True, dataset=raw_dir.name, done=0, total=total,
                       stage="model", error=None)

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
                    return self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")

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

                return self._send(404, b"not found", "text/plain")
            except PermissionError as exc:
                return self._json({"error": str(exc)}, 403)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            try:
                body = self._body()
                raw_dir = self._ds(body.get("dataset"))

                if url.path == "/api/embed":
                    with JOB_LOCK:
                        if JOB["running"]:
                            return self._json({"error": f"이미 임베딩 중: {JOB['dataset']}"}, 409)
                        JOB.update(running=True, dataset=raw_dir.name, done=0, total=0, error=None)
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="[0a] DINOv2 그룹핑/제외/강조 + 심볼릭 링크 익스포트 웹 UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("raw", nargs="?", default=None,
                    help="dataset_raw/<이름> 또는 경로. 생략하면 웹 UI 에서 고른다")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="후보 쌍 centroid 코사인 임계값. 낮출수록 후보가 많아진다 (기본 0.85)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="DINOv2 배치 크기 (기본 16, --fast 면 32)")
    ap.add_argument("--fast", action="store_true",
                    help="DINOv2 를 half precision(GPU 에서 bf16/fp16) + fast 프로세서로. GPU forward 가 "
                         "이미 짧아 이득은 작다(진짜 병목은 이미지 로딩 → 항상 병렬). 값이 fp32 와 달라 "
                         "캐시는 모드별로 자동 무효화된다(수동 삭제 불필요). 환경변수 DINOV2_FAST=1 도 가능")
    ap.add_argument("--workers", type=int, default=None,
                    help="이미지 로딩 스레드 수 (기본 auto = min(16, CPU수)). 임베딩 속도를 지배한다")
    ap.add_argument("--refresh", action="store_true", help="임베딩 캐시 무시하고 다시 계산")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="서버를 띄우지 않고 현황만 출력")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="브라우저 자동 실행")
    args = ap.parse_args()

    if args.print_only and not args.raw:
        raise SystemExit("[error] --print 는 데이터셋 이름이 필요하다 (예: 0_dedup_raw.py mychar --print)")

    # fast/workers 는 프로세스 전역 스위치다. --fast/--workers 또는 DINOV2_FAST/DINOV2_WORKERS.
    if args.fast:
        dedup.FAST = True
    if args.workers:
        dedup.WORKERS = args.workers
    batch_size = args.batch_size if args.batch_size else (32 if dedup.FAST else 16)
    print(f"[dinov2] 임베딩 모드: {dedup._cache_mode()} · 로딩 스레드 {dedup._default_workers()} · batch {batch_size}")

    default_ds = None
    if args.raw:
        raw_dir = dedup.resolve_raw(args.raw)
        default_ds = raw_dir.name
        payload = state_payload(raw_dir, args.threshold, build=True,
                                refresh=args.refresh, batch_size=batch_size)
        if payload.get("error"):
            raise SystemExit(f"[error] {payload['error']}")
        print_report(payload)
        if args.print_only:
            return 0

    if not dedup.list_raw_datasets():
        raise SystemExit(f"[error] dataset_raw/ 아래에 원시 데이터셋이 없다: {dedup.RAW_ROOT}")

    handler = make_handler(default_ds, args.threshold, batch_size)
    with _Server((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}"
        print(f"[serve] {url}  (Ctrl-C 로 종료)")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] 종료")
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dedup — 그룹핑 · 익스포트</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2b313d;
    --fg:#e6e9ef; --dim:#98a2b3; --accent:#5b9dff; --ok:#3ecf8e; --warn:#f5a524; --err:#f45b69;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,"Pretendard","Apple SD Gothic Neo",sans-serif; }
  header { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--line);
           padding:12px 20px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .meta { color:var(--dim); font-size:12px; }
  .spacer { flex:1; }
  button { background:var(--panel2); color:var(--fg); border:1px solid var(--line); border-radius:6px;
           padding:6px 12px; cursor:pointer; font-size:13px; }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#08101f; font-weight:600; }
  button.ok { border-color:var(--ok); color:var(--ok); }
  button.danger:hover { border-color:var(--err); color:var(--err); }
  main { padding:20px; max-width:1600px; margin:0 auto; }
  section { margin-bottom:26px; }
  .sechead { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
  .sechead h2 { font-size:14px; margin:0; font-weight:600; }
  .sechead .tiny { color:var(--dim); }

  /* 후보 쌍 */
  .cand { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          display:flex; gap:12px; align-items:center; padding:12px; margin-bottom:10px; }
  .cand.hot { border-color:var(--accent); }
  .side { display:flex; gap:8px; align-items:center; }
  .simbig { font-family:ui-monospace,monospace; font-size:18px; font-weight:600; min-width:64px; text-align:center; }
  .vs { color:var(--dim); font-size:12px; }
  .cand .acts { display:flex; gap:6px; margin-left:auto; }

  /* 그룹 / 그리드 */
  .group { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           margin-bottom:14px; overflow:hidden; }
  .group.special { border-color:var(--warn); }
  .ghead { display:flex; gap:10px; align-items:center; padding:10px 14px;
           background:var(--panel2); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; padding:12px; }
  .tile { background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:7px;
          display:flex; flex-direction:column; gap:5px; }
  .tile.excl { opacity:.5; border-color:var(--err); }
  .thumb { width:100%; aspect-ratio:1; object-fit:contain; background:#0a0c10; border-radius:4px; cursor:zoom-in; }
  .mini { width:64px; height:64px; object-fit:cover; border-radius:6px; background:#0a0c10; cursor:zoom-in; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:10px; }
  .selectable { position:relative; }
  .selectable.sel .thumb { outline:3px solid var(--accent); outline-offset:-3px; }
  select, input[type=number], input[type=text] { background:var(--panel2); color:var(--fg);
           border:1px solid var(--line); border-radius:6px; padding:5px 8px; font-size:13px; }
  input[type=number] { width:70px; font-family:ui-monospace,monospace; }
  .bar { height:6px; background:var(--panel2); border-radius:99px; overflow:hidden; margin:10px 0; }
  .bar > div { height:100%; background:var(--accent); transition:width .2s; }
  .center { text-align:center; padding:48px 20px; }
  .badge { font-size:10px; padding:1px 6px; border-radius:99px; border:1px solid var(--line); color:var(--dim); }
  .badge.warn { color:var(--warn); border-color:var(--warn); }
  .badge.ok { color:var(--ok); border-color:var(--ok); }
  .tiny { font-size:11px; color:var(--dim); }
  dialog { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:10px;
           padding:0; max-width:720px; width:90vw; }
  dialog::backdrop { background:rgba(0,0,0,.7); }
  .dlg-body { padding:18px; max-height:66vh; overflow:auto; }
  .dlg-foot { padding:12px 16px; border-top:1px solid var(--line); display:flex; gap:8px; justify-content:flex-end; }
  .field { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
  .field label { min-width:130px; color:var(--dim); font-size:13px; }
  .msg { padding:8px 12px; border-radius:6px; margin-bottom:8px; font-size:12px; }
  .msg.err { background:rgba(244,91,105,.12); color:var(--err); }
  .msg.ok { background:rgba(62,207,142,.12); color:var(--ok); }
  #lightbox img { max-width:100%; max-height:80vh; display:block; margin:auto; }
  #lightbox .dlg-body { padding:8px; }
  #toast { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:var(--panel2);
           border:1px solid var(--line); border-radius:8px; padding:10px 16px; display:none; z-index:50; }
  kbd { font-family:ui-monospace,monospace; font-size:11px; background:var(--panel2);
        border:1px solid var(--line); border-radius:4px; padding:0 5px; }
</style>
</head>
<body>
<header>
  <h1>dedup</h1>
  <select id="ds" title="dataset_raw/ 의 원시 데이터셋"></select>
  <label class="tiny">임계값
    <input type="number" id="thr" step="0.01" min="0.40" max="0.99" title="centroid 코사인 유사도">
  </label>
  <button id="embed" title="DINOv2 임베딩을 다시 계산한다">임베딩 재계산</button>
  <span class="meta" id="meta">로딩…</span>
  <span class="spacer"></span>
  <button id="export" class="primary">익스포트…</button>
</header>
<main id="root"></main>

<dialog id="export-dlg">
  <div class="dlg-body">
    <h2 style="margin:0 0 14px">심볼릭 링크 데이터셋 익스포트</h2>
    <div class="field"><label>이름 (dataset/)</label>
      <input type="text" id="ex-name" style="flex:1"></div>
    <div class="field"><label>repeat (R)</label>
      <input type="number" id="ex-repeat" value="10" min="1">
      <span class="tiny">그룹 합 R · 강조 그룹 2R · 그룹 크기 k 면 각 round(합/k)</span></div>
    <div class="field"><label>rounding</label>
      <select id="ex-round">
        <option value="ceil">ceil (올림)</option>
        <option value="round">round (반올림)</option>
        <option value="floor">floor (내림)</option>
      </select></div>
    <div class="field"><label>덮어쓰기</label>
      <input type="checkbox" id="ex-force"><span class="tiny">기존 dataset/&lt;이름&gt;/repeat_* 을 지우고 새로 쓴다</span></div>
    <div id="ex-preview"></div>
  </div>
  <div class="dlg-foot">
    <button onclick="document.getElementById('export-dlg').close()">닫기</button>
    <button id="ex-dry">미리보기</button>
    <button class="primary" id="ex-run">익스포트 실행</button>
  </div>
</dialog>
<dialog id="lightbox"><div class="dlg-body"><img id="lightbox-img"></div></dialog>
<div id="toast"></div>

<script>
let DATA = null;
let DS = null;
let THR = 0.85;
let PICK = null;   // {rel} 그룹/싱글턴에서 "다르다/병합" 대상으로 찍어둔 것

const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const toast = m => { const t = document.getElementById('toast');
  t.textContent = m; t.style.display='block'; clearTimeout(t._h);
  t._h = setTimeout(()=>t.style.display='none', 3000); };
const setMeta = t => document.getElementById('meta').textContent = t;
const setRoot = h => document.getElementById('root').innerHTML = h;
const kb = b => b >= 1048576 ? (b/1048576).toFixed(1)+'MB' : (b/1024).toFixed(0)+'KB';
const thumb = (rel, s) => `/thumb?ds=${encodeURIComponent(DS)}&rel=${encodeURIComponent(rel)}${s?`&s=${s}`:''}`;

async function boot() {
  const d = await (await fetch('/api/datasets')).json();
  if (d.error) return setRoot(`<div class="msg err">${esc(d.error)}</div>`);
  const sel = document.getElementById('ds');
  sel.innerHTML = d.datasets.map(x =>
    `<option value="${esc(x.name)}">${esc(x.name)} · ${x.images}장${
      x.pending ? ` · 임베딩 ${x.pending}장 필요` : ''}</option>`).join('');
  DS = d.datasets.some(x => x.name === d.default) ? d.default : (d.datasets[0] || {}).name;
  sel.value = DS;
  THR = d.threshold;
  document.getElementById('thr').value = THR.toFixed(2);
  const j = await (await fetch('/api/job')).json();
  if (j.running) { DS = j.dataset; sel.value = DS; return poll(); }
  await load();
}

async function load() {
  if (!DS) return setRoot('<p class="tiny">dataset_raw/ 에 원시 데이터셋이 없다.</p>');
  setMeta('불러오는 중…');
  PICK = null;
  const r = await fetch(`/api/state?ds=${encodeURIComponent(DS)}&threshold=${THR}`);
  DATA = await r.json();
  if (DATA.error) { setRoot(`<div class="msg err">${esc(DATA.error)}</div>`); setMeta(DS); return; }
  if (DATA.need_embed) return renderNeedEmbed();
  setMeta(`이미지 ${DATA.total_images}장 · 그룹 ${DATA.n_groups}`
    + ` (묶임 ${DATA.n_multi} · 싱글턴 ${DATA.n_singletons})`
    + ` · 강조 ${DATA.n_special} · 제외 ${DATA.n_excluded}`
    + (DATA.n_cannot_link ? ` · 다름 ${DATA.n_cannot_link}` : ''));
  render();
}

function renderNeedEmbed() {
  setMeta(`이미지 ${DATA.total_images}장`);
  setRoot(`<div class="center">
    <p>이 데이터셋은 아직 DINOv2 임베딩이 없다 — <b>${DATA.pending}장</b> 계산이 필요하다.</p>
    <p class="tiny">GPU 로 수십 초 걸린다. 결과는 캐시된다.</p>
    <button class="primary" onclick="startEmbed(false)">DINOv2 임베딩 시작</button>
  </div>`);
}

async function startEmbed(refresh) {
  const res = await (await fetch('/api/embed', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ dataset: DS, refresh }) })).json();
  if (res.error) return toast(res.error);
  poll();
}

async function poll() {
  const j = await (await fetch('/api/job')).json();
  if (j.error) return setRoot(`<div class="msg err">임베딩 실패: ${esc(j.error)}</div>`);
  if (j.running) {
    const loading = j.stage === 'model';
    const pct = j.total ? Math.round(j.done / j.total * 100) : 0;
    setMeta(loading ? 'DINOv2 모델 로딩 중…' : `임베딩 중… ${j.done}/${j.total}`);
    setRoot(`<div class="center">
      <p>${loading ? 'DINOv2 모델을 올리는 중' : 'DINOv2 임베딩 중'} — ${esc(j.dataset)}</p>
      ${loading ? '<p class="tiny">첫 실행은 모델 로딩에 수 초 걸린다.</p>'
        : `<div class="bar" style="max-width:420px;margin:12px auto"><div style="width:${pct}%"></div></div>
           <p class="tiny">${j.done} / ${j.total} (${pct}%)</p>`}
    </div>`);
    setTimeout(poll, 500);
    return;
  }
  await load();
}

// -- 액션 (전부 서버 왕복 후 재로딩) ------------------------------------------
async function act(path, payload) {
  const res = await (await fetch(path, { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ dataset: DS, ...payload }) })).json();
  if (res.error) { toast(res.error); return null; }
  return res;
}
const gid = id => DATA.groups.find(g => g.id === id);
async function merge(a, b)      { if (await act('/api/merge', {a, b}))      await load(); }
async function different(a, b)  { if (await act('/api/different', {a, b}))  await load(); }
async function ungroup(rel)     { if (await act('/api/ungroup', {rel}))     await load(); }
async function removeMember(rel){ if (await act('/api/remove-member',{rel}))await load(); }
async function exclude(rel)     { if (await act('/api/exclude', {rel}))     await load(); }
async function include(rel)     { if (await act('/api/include', {rel}))     await load(); }
async function special(rel, on) { if (await act('/api/special', {rel, on})) await load(); }

/* 그룹/싱글턴에서 대표 썸네일을 찍으면 PICK 에 담고, 두 번째를 찍으면 액션 바를 띄운다.
   같은 걸 다시 찍으면 해제. */
function pick(rep) {
  if (PICK && PICK.rel === rep) { PICK = null; render(); return; }
  if (!PICK) { PICK = { rel: rep }; render(); return; }
  // 두 번째 선택 → 확인 바가 뜬 상태로 render (아래 pickbar)
  PICK.other = rep; render();
}
function clearPick() { PICK = null; render(); }

function render() {
  const root = document.getElementById('root');
  const cands = DATA.candidates || [];
  const multi = DATA.groups.filter(g => g.size > 1);
  const singles = DATA.groups.filter(g => g.size === 1);
  let html = '';

  // 수동 병합/다름 확인 바 (그룹 대표 두 개를 찍었을 때)
  if (PICK && PICK.other) {
    html += `<div class="cand hot" style="position:sticky;top:64px;z-index:5">
      <div class="side"><img class="mini" src="${thumb(PICK.rel,120)}"></div>
      <span class="vs">선택한 둘을</span>
      <div class="side"><img class="mini" src="${thumb(PICK.other,120)}"></div>
      <div class="acts">
        <button class="ok" onclick="merge('${esc(PICK.rel)}','${esc(PICK.other)}')">병합</button>
        <button class="danger" onclick="different('${esc(PICK.rel)}','${esc(PICK.other)}')">다르다</button>
        <button onclick="clearPick()">취소</button>
      </div></div>`;
  } else if (PICK) {
    html += `<div class="msg ok">기준 이미지 선택됨 — 병합/구분할 다른 그룹의 대표를 하나 더 찍으세요.
      <button onclick="clearPick()" style="margin-left:8px">취소</button></div>`;
  }

  // 1) 후보 쌍
  html += `<section><div class="sechead"><h2>후보 쌍</h2>
    <span class="tiny">임계값 ${THR.toFixed(2)} 이상 · ${cands.length}쌍 ·
    <kbd>병합</kbd> 같은 그림 / <kbd>다르다</kbd> 별개(다시 안 뜸)</span></div>`;
  if (!cands.length) {
    html += `<p class="tiny">이 임계값에서는 후보가 없다. 위 임계값을 낮추면 더 뜬다.</p>`;
  }
  for (const c of cands) {
    html += `<div class="cand">
      <span class="simbig">${(c.sim*100).toFixed(1)}%</span>
      <div class="side">
        <img class="mini" src="${thumb(c.a,120)}" onclick="zoom('${esc(c.a)}')">
        ${c.size_a>1?`<span class="badge">그룹 ${c.size_a}</span>`:''}
        ${c.special_a?'<span class="badge warn">강조</span>':''}
      </div>
      <span class="vs">~</span>
      <div class="side">
        <img class="mini" src="${thumb(c.b,120)}" onclick="zoom('${esc(c.b)}')">
        ${c.size_b>1?`<span class="badge">그룹 ${c.size_b}</span>`:''}
        ${c.special_b?'<span class="badge warn">강조</span>':''}
      </div>
      <div class="acts">
        <button class="ok" onclick="merge('${esc(c.a)}','${esc(c.b)}')">병합</button>
        <button class="danger" onclick="different('${esc(c.a)}','${esc(c.b)}')">다르다</button>
      </div></div>`;
  }
  html += `</section>`;

  // 2) 묶인 그룹
  html += `<section><div class="sechead"><h2>묶인 그룹</h2>
    <span class="tiny">${multi.length}개 · 그룹 전체 반복수 합 R (강조 2R)</span></div>`;
  if (!multi.length) html += `<p class="tiny">아직 병합된 그룹이 없다.</p>`;
  for (const g of multi) html += groupCard(g);
  html += `</section>`;

  // 3) 개별 이미지(싱글턴)
  html += `<section><div class="sechead"><h2>개별 이미지</h2>
    <span class="tiny">${singles.length}장 · 대표를 찍어 위에서 병합/구분하거나, 강조/제외</span></div>
    <div class="grid">`;
  for (const g of singles) {
    const m = g.members[0];
    const picked = PICK && (PICK.rel === g.rep || PICK.other === g.rep);
    html += `<div class="tile selectable ${picked?'sel':''} ${g.special?'':''}">
      <img class="thumb" loading="lazy" src="${thumb(m.rel)}" onclick="pick('${esc(g.rep)}')"
           ondblclick="zoom('${esc(m.rel)}')" title="클릭=선택, 더블클릭=확대">
      <div class="tiny" style="display:flex;gap:6px;align-items:center">
        ${g.special?'<span class="badge warn">강조</span>':''}
        <span class="spacer" style="flex:1"></span>${kb(m.bytes)}</div>
      <div class="row" style="display:flex;gap:5px">
        <button style="flex:1" onclick="special('${esc(g.rep)}',${!g.special})">${g.special?'강조 해제':'강조'}</button>
        <button class="danger" onclick="exclude('${esc(g.rep)}')">제외</button>
      </div></div>`;
  }
  html += `</div></section>`;

  // 4) 제외됨
  if (DATA.excluded.length) {
    html += `<section><div class="sechead"><h2>제외됨</h2>
      <span class="tiny">${DATA.excluded.length}장 · 익스포트에서 빠진다</span></div><div class="grid">`;
    for (const e of DATA.excluded) {
      html += `<div class="tile excl">
        <img class="thumb" loading="lazy" src="${thumb(e.rel)}" onclick="zoom('${esc(e.rel)}')">
        <div class="tiny">${esc(e.name)}</div>
        <button onclick="include('${esc(e.rel)}')">복구</button></div>`;
    }
    html += `</div></section>`;
  }

  root.innerHTML = html;
}

function groupCard(g) {
  const picked = PICK && (PICK.rel === g.rep || PICK.other === g.rep);
  let h = `<div class="group ${g.special?'special':''}">
    <div class="ghead">
      <img class="mini" src="${thumb(g.rep,120)}" onclick="pick('${esc(g.rep)}')"
           style="${picked?'outline:3px solid var(--accent);outline-offset:-3px':''}"
           title="대표 — 클릭해 다른 그룹과 병합/구분">
      <span class="tiny">그룹 ${g.size}장</span>
      ${g.special?'<span class="badge warn">강조 (2R)</span>':''}
      <span class="spacer"></span>
      <button onclick="special('${esc(g.rep)}',${!g.special})">${g.special?'강조 해제':'강조(2R)'}</button>
      <button onclick="ungroup('${esc(g.rep)}')">그룹 해체</button>
    </div><div class="tiles">`;
  for (const m of g.members) {
    h += `<div class="tile">
      <img class="thumb" loading="lazy" src="${thumb(m.rel)}" onclick="zoom('${esc(m.rel)}')">
      <div class="tiny">${kb(m.bytes)}</div>
      <div class="row" style="display:flex;gap:5px">
        <button style="flex:1" onclick="removeMember('${esc(m.rel)}')" title="이 이미지를 그룹에서 빼 싱글턴으로">분리</button>
        <button class="danger" onclick="exclude('${esc(m.rel)}')">제외</button>
      </div></div>`;
  }
  return h + `</div></div>`;
}

function zoom(rel) {
  document.getElementById('lightbox-img').src = `/full?ds=${encodeURIComponent(DS)}&rel=${encodeURIComponent(rel)}`;
  document.getElementById('lightbox').showModal();
}
document.getElementById('lightbox').onclick = e => e.target.closest('img') || e.currentTarget.close();

// -- 익스포트 --------------------------------------------------------------
const exDlg = document.getElementById('export-dlg');
document.getElementById('export').onclick = () => {
  if (!DATA || DATA.need_embed) return toast('먼저 임베딩이 필요하다.');
  document.getElementById('ex-name').value = DS;
  document.getElementById('ex-preview').innerHTML = '';
  exDlg.showModal();
};
function exPayload(extra) {
  return {
    name: document.getElementById('ex-name').value.trim() || DS,
    repeat: parseInt(document.getElementById('ex-repeat').value) || 10,
    rounding: document.getElementById('ex-round').value,
    force: document.getElementById('ex-force').checked,
    ...extra,
  };
}
function renderStats(s) {
  const dist = Object.entries(s.dist || {}).map(([n,c]) => `repeat_${n}×${c}`).join(' · ');
  let h = `<div class="msg ok">이미지 ${s.images}장 · 그룹 ${s.groups}
    (싱글턴 ${s.singletons} · 묶임 ${s.multi} · 강조 ${s.special}) · 제외 ${s.excluded}<br>
    반복수 분포: ${dist || '-'} · 총 링크 ${s.total_reps}개</div>`;
  const col = s.collisions || {};
  if (Object.keys(col).length)
    h += `<div class="msg err">플래트닝 이름 충돌 ${Object.keys(col).length}건 —
      원시 파일명을 정리해야 익스포트된다. 예: ${esc(Object.keys(col)[0])}</div>`;
  return h;
}
document.getElementById('ex-dry').onclick = async () => {
  const res = await act('/api/export', exPayload({ dry_run: true }));
  if (res) document.getElementById('ex-preview').innerHTML = renderStats(res.stats);
};
document.getElementById('ex-run').onclick = async () => {
  const res = await act('/api/export', exPayload({}));
  if (!res) return;
  document.getElementById('ex-preview').innerHTML = renderStats(res.stats)
    + `<div class="msg ok">완료 — ${esc(res.stats.out_dir)} 에 링크 ${res.stats.created}개 생성.</div>`;
  toast(`익스포트 완료: dataset/${res.stats.name}`);
};

// -- 컨트롤 ----------------------------------------------------------------
document.getElementById('ds').onchange = e => { DS = e.target.value; load(); };
document.getElementById('thr').onchange = e => {
  const v = parseFloat(e.target.value);
  if (!isNaN(v) && v > 0 && v < 1) { THR = v; load(); } else { e.target.value = THR.toFixed(2); }
};
document.getElementById('embed').onclick = () => {
  if (confirm(`${DS} 의 임베딩을 다시 계산한다. 캐시를 버리고 전부 다시 돌린다.`)) startEmbed(true);
};

boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
