#!/usr/bin/env python3
"""[0a] dataset_raw/<원시> 의 차분/중복 후보를 DINOv2 로 찾아 웹 UI 에서 리네임한다.

0_process_raw.py 의 차분(`_sabun`)/강조(`_special_`) 규칙은 순전히 파일명에만 의존한다.
같은 그림의 변형(차분)을 이름으로 묶어주지 않으면 그 그림만 K배 과대표집된다. 이 스텝은
DINOv2-base 의 CLS 임베딩으로 변형 후보를 찾아내고, 웹 UI 에서 원본을 고르면 규칙에 맞는
이름으로 일괄 리네임한다. 즉 0_process_raw.py **앞에** 오는 선택적 정리 단계다.

사용 예:
    # 그냥 띄운다. 데이터셋 선택 / 임베딩 / 임계값 조절 모두 웹 UI 에서 한다.
    uv run python 0a_dedup_raw.py

    # 데이터셋을 지정하면 미리 임베딩해서 결과를 찍고, UI 에도 그 데이터셋을 띄워 둔다.
    uv run python 0a_dedup_raw.py mychar

    # 서버 없이 탐지 결과만 출력 / 마지막 적용 되돌리기 (이름 필수)
    uv run python 0a_dedup_raw.py mychar --print
    uv run python 0a_dedup_raw.py mychar --undo

임계값은 UI 에서 바꾸면 즉시 반영된다(임베딩과 무관하게 그룹핑만 다시 한다).

임계값 0.85 는 mychar 의 기존 `_sabun` 표기를 정답으로 놓고 실측한 값이다
(차분 쌍 중앙값 0.95 / 무관한 쌍 상위5% 0.72 → 0.85 에서 재현율 92.9%).

다음 단계: 0_process_raw.py <원시> --repeat 10
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

import numpy as np
from PIL import Image

import dedup

THUMB_SIZE = 320


# ---------------------------------------------------------------------------
# 탐지 → UI 가 먹을 JSON
# ---------------------------------------------------------------------------
def build_payload(raw_dir: Path, threshold: float, *, build: bool = False,
                  refresh: bool = False, batch_size: int = 16) -> dict:
    """그룹 탐지 결과. build=False 면 임베딩을 새로 돌리지 않고, 캐시가 없으면
    need_embed 를 돌려준다 (웹 UI 가 임베딩을 따로 트리거하게 하기 위해)."""
    rels = dedup.collect_images(raw_dir)
    if not rels:
        return {"dataset": raw_dir.name, "error": f"{raw_dir} 아래에 이미지가 없다."}

    if build:
        def cli_progress(done, total):
            print(f"  {done}/{total}", end="\n" if done == total else "\r", flush=True)
        emb = dedup.build_embeddings(raw_dir, rels, force=refresh,
                                     batch_size=batch_size, progress=cli_progress)
    else:
        emb = dedup.cached_embeddings(raw_dir, rels)
    if emb is None:
        return {
            "dataset": raw_dir.name,
            "need_embed": True,
            "pending": dedup.pending_count(raw_dir, rels),
            "total_images": len(rels),
            "threshold": threshold,
        }

    groups = dedup.find_groups(rels, emb, threshold)

    def member(i: int, sim: float | None) -> dict:
        rel = rels[i]
        path = raw_dir / rel
        stem = Path(rel.name).stem
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        return {
            "rel": str(rel),
            "dir": dedup.dir_prefix(rel),
            "name": rel.name,
            "stem": stem,
            "ext": Path(rel.name).suffix,
            "flat": dedup.flat_name(rel),
            "w": w,
            "h": h,
            "bytes": path.stat().st_size,
            "is_sabun": dedup.is_sabun(dedup.flat_stem(rel)),
            "is_special": dedup.is_special(dedup.flat_stem(rel)),
            "malformed": dedup.malformed_sabun(stem),
            "sim": sim,
        }

    out_groups = []
    for gid, idxs in enumerate(groups):
        # 그룹 대표: 나머지와의 평균 유사도가 가장 높은 것(medoid).
        sub = emb[idxs]
        medoid = idxs[int(np.argmax((sub @ sub.T).sum(axis=1)))]
        # 단, 원본 후보는 차분 접미사가 **전혀** 없는 것으로 한정한다. is_sabun() 만 보면
        # `_sabun1` 같은 비규격 이름이 원본으로 뽑힌다(0_process_raw 가 인식 못 하므로).
        clean = [i for i in idxs if dedup.loose_base(dedup.flat_stem(rels[i])) == dedup.flat_stem(rels[i])]
        anchor = clean[int(np.argmax((emb[clean] @ sub.T).sum(axis=1)))] if clean else medoid

        anchor_vec = emb[anchor]
        members = [member(i, float(emb[i] @ anchor_vec)) for i in idxs]
        members.sort(key=lambda m: (m["rel"] != str(rels[anchor]), -(m["sim"] or 0)))
        dirs = {m["dir"] for m in members}
        out_groups.append({
            "id": gid,
            "base": dedup.loose_base(Path(rels[anchor].name).stem),
            "anchor": str(rels[anchor]),
            "inherited_special": dedup.inherits_special(rels[anchor]),
            "multi_dir": len(dirs) > 1,
            "members": members,
        })
    out_groups.sort(key=lambda g: (-len(g["members"]), g["base"]))
    # id 는 정렬 **후** 의 인덱스여야 한다. UI 가 DATA.groups[id] 로 되짚는다.
    for i, g in enumerate(out_groups):
        g["id"] = i

    excluded = []
    for rel in dedup.collect_excluded(raw_dir):
        p = dedup.resolve_rel(raw_dir, rel)
        excluded.append({"rel": rel, "name": Path(rel).name, "bytes": p.stat().st_size})

    return {
        "dataset": raw_dir.name,
        "raw_dir": str(raw_dir),
        "threshold": threshold,
        "total_images": len(rels),
        "grouped_images": sum(len(g["members"]) for g in out_groups),
        "sabun_suffix": dedup.SABUN_SUFFIX,
        "special_marker": dedup.SPECIAL_MARKER,
        "exclude_prefix": dedup.EXCLUDE_PREFIX,
        "groups": out_groups,
        "excluded": excluded,
        "has_undo": bool(dedup._read_log(raw_dir)["history"]),
    }


def print_report(payload: dict) -> None:
    print(f"\n[dedup] {payload['dataset']}  images={payload['total_images']}  "
          f"threshold={payload['threshold']}")
    print(f"[dedup] 후보 그룹 {len(payload['groups'])}개 / 이미지 {payload['grouped_images']}장\n")
    for g in payload["groups"]:
        flags = []
        if g["inherited_special"]:
            flags.append("폴더강조")
        if g["multi_dir"]:
            flags.append("폴더걸침")
        tag = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  # {g['base']}  ({len(g['members'])}장){tag}")
        for m in g["members"]:
            role = "원본" if m["rel"] == g["anchor"] else ("차분" if m["is_sabun"] else "  ? ")
            warn = " ← _sabun_N 아님" if m["malformed"] else ""
            print(f"      {role}  sim={m['sim']:.4f}  {m['rel']}{warn}")
        print()


# ---------------------------------------------------------------------------
# 웹 UI
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


# 임베딩은 수십 초가 걸린다. 요청 안에서 돌리면 브라우저가 통째로 멈추므로 백그라운드
# 스레드에서 돌리고 UI 는 /api/job 을 폴링해 진행률을 본다. 동시에 두 개는 돌리지 않는다.
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

        def on_ready():  # 모델 로딩 끝 → 여기서부터 진행률이 실제로 움직인다
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


def make_handler(default_ds: str | None, threshold: float, batch_size: int):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 요청마다 stderr 로 시끄럽게 찍지 않는다
            pass

        # -- helpers ---------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _ds(self, name: str | None) -> Path:
            """데이터셋 이름 → raw_dir. dataset_raw/ 의 실제 목록에 있는 이름만 받으므로
            경로 조작(../ 등)은 애초에 통과하지 못한다."""
            if not name:
                raise PermissionError("데이터셋이 지정되지 않았다.")
            if name not in dedup.list_raw_datasets():
                raise PermissionError(f"알 수 없는 데이터셋: {name}")
            return (dedup.RAW_ROOT / name).resolve()

        def _safe(self, raw_dir: Path, rel: str) -> Path:
            """rel 을 실제 경로로 풀되, raw_dir/제외폴더 밖으로 나가면 거부한다."""
            p = dedup.resolve_rel(raw_dir, rel).resolve()
            roots = [raw_dir.resolve(), dedup.excluded_dir(raw_dir).resolve()]
            if not any(p == r or r in p.parents for r in roots):
                raise PermissionError(f"경로 거부: {rel}")
            return p

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        # -- routes ----------------------------------------------------------
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
                    return self._json({
                        "datasets": items,
                        "default": default_ds,
                        "threshold": threshold,
                    })

                if url.path == "/api/job":
                    with JOB_LOCK:
                        return self._json(dict(JOB))

                if url.path == "/api/groups":
                    raw_dir = self._ds(qs.get("ds", [None])[0])
                    thr = float(qs.get("threshold", [threshold])[0])
                    return self._json(build_payload(raw_dir, thr, batch_size=batch_size))

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

                if url.path in ("/api/check", "/api/apply"):
                    targets = {str(k): str(v) for k, v in body.get("targets", {}).items()}
                    for k, v in targets.items():
                        self._safe(raw_dir, k)
                        self._safe(raw_dir, v)
                    moves, check = dedup.plan_moves(raw_dir, targets)
                    if url.path == "/api/check" or check.errors:
                        return self._json({
                            "applied": 0, "moves": moves,
                            "errors": check.errors, "warnings": check.warnings,
                        })
                    n = dedup.apply_moves(raw_dir, moves)
                    print(f"[apply] {raw_dir.name}: {n}건 리네임")
                    return self._json({
                        "applied": n, "moves": moves, "errors": [], "warnings": check.warnings,
                    })

                if url.path == "/api/undo":
                    n, msg = dedup.undo_last(raw_dir)
                    print(f"[undo] {raw_dir.name}: {msg}")
                    return self._json({"undone": n, "message": msg})

                return self._send(404, b"not found", "text/plain")
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
        description="[0a] DINOv2(CLS) 차분/중복 탐지 + 웹 UI 리네임",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("raw", nargs="?", default=None,
                    help="dataset_raw/<이름> 의 원시 데이터셋 이름 또는 경로. "
                         "생략하면 웹 UI 에서 고른다 (--print/--undo 는 필수)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="CLS 코사인 유사도 임계값. 낮을수록 더 많이 묶인다. UI 에서도 바꿀 수 있다 (기본 0.85)")
    ap.add_argument("--batch-size", type=int, default=16, help="DINOv2 배치 크기 (기본 16)")
    ap.add_argument("--refresh", action="store_true", help="임베딩 캐시 무시하고 다시 계산")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="서버를 띄우지 않고 탐지 결과만 출력")
    ap.add_argument("--undo", action="store_true", help="가장 최근 적용 배치를 되돌리고 종료")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="브라우저 자동 실행")
    args = ap.parse_args()

    if (args.print_only or args.undo) and not args.raw:
        raise SystemExit("[error] --print / --undo 는 데이터셋 이름이 필요하다 "
                         "(예: 0a_dedup_raw.py mychar --print)")

    if args.undo:
        n, msg = dedup.undo_last(dedup.resolve_raw(args.raw))
        print(f"[undo] {msg}")
        return 0 if n else 1

    default_ds = None
    if args.raw:
        # 이름을 줬으면 미리 임베딩해서 결과를 찍어주고, UI 에도 그 데이터셋을 띄워 둔다.
        raw_dir = dedup.resolve_raw(args.raw)
        default_ds = raw_dir.name
        payload = build_payload(raw_dir, args.threshold, build=True,
                                refresh=args.refresh, batch_size=args.batch_size)
        if payload.get("error"):
            raise SystemExit(f"[error] {payload['error']}")
        print_report(payload)
        if args.print_only:
            return 0

    if not dedup.list_raw_datasets():
        raise SystemExit(f"[error] dataset_raw/ 아래에 원시 데이터셋이 없다: {dedup.RAW_ROOT}")

    handler = make_handler(default_ds, args.threshold, args.batch_size)
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
<title>차분 정리 — dedup</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2b313d;
    --fg:#e6e9ef; --dim:#98a2b3; --accent:#5b9dff; --ok:#3ecf8e; --warn:#f5a524; --err:#f45b69;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,"Pretendard","Apple SD Gothic Neo",sans-serif; }
  header { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--line);
           padding:12px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .meta { color:var(--dim); font-size:12px; }
  .spacer { flex:1; }
  button { background:var(--panel2); color:var(--fg); border:1px solid var(--line); border-radius:6px;
           padding:6px 12px; cursor:pointer; font-size:13px; }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#08101f; font-weight:600; }
  button.danger:hover { border-color:var(--err); color:var(--err); }
  main { padding:20px; max-width:1600px; margin:0 auto; }
  .group { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           margin-bottom:16px; overflow:hidden; }
  .ghead { display:flex; gap:10px; align-items:center; padding:10px 14px;
           background:var(--panel2); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  .gbase { font-family:ui-monospace,monospace; font-size:13px; color:var(--accent); }
  .tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; padding:14px; }
  .tile { background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:8px;
          display:flex; flex-direction:column; gap:6px; }
  .tile.orig { border-color:var(--ok); }
  .tile.excl { opacity:.45; border-color:var(--err); }
  /* 이미 규칙대로 묶여 있는 그룹은 손댈 게 없다 → 죽여 두고 접는다. */
  .group.done { opacity:.65; }
  .group.done:hover { opacity:1; }
  .group.orphan { border-color:var(--err); }
  .caret { padding:2px 8px; line-height:1; font-family:ui-monospace,monospace; }
  .mini { width:34px; height:34px; object-fit:cover; border-radius:4px; background:#0a0c10;
          cursor:zoom-in; }
  .foldbar { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
  .thumb { width:100%; aspect-ratio:1; object-fit:contain; background:#0a0c10; border-radius:4px; cursor:zoom-in; }
  .row { display:flex; gap:6px; align-items:center; }
  input[type=text] { width:100%; background:#0a0c10; color:var(--fg); border:1px solid var(--line);
                     border-radius:4px; padding:4px 6px; font-family:ui-monospace,monospace; font-size:11px; }
  input[type=text].changed { border-color:var(--warn); color:var(--warn); }
  select, input[type=number] { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
                     border-radius:6px; padding:5px 8px; font-size:13px; }
  input[type=number] { width:72px; font-family:ui-monospace,monospace; }
  .bar { height:6px; background:var(--panel2); border-radius:99px; overflow:hidden; margin:10px 0; }
  .bar > div { height:100%; background:var(--accent); transition:width .2s; }
  .center { text-align:center; padding:48px 20px; }
  .badge { font-size:10px; padding:1px 6px; border-radius:99px; border:1px solid var(--line); color:var(--dim); }
  .badge.ok { color:var(--ok); border-color:var(--ok); }
  .badge.warn { color:var(--warn); border-color:var(--warn); }
  .badge.err { color:var(--err); border-color:var(--err); }
  .sim { font-family:ui-monospace,monospace; font-size:11px; color:var(--dim); }
  .tiny { font-size:11px; color:var(--dim); }
  .note { padding:6px 14px; font-size:12px; background:rgba(245,165,36,.1); color:var(--warn);
          border-bottom:1px solid var(--line); }
  dialog { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:10px;
           padding:0; max-width:900px; width:90vw; }
  dialog::backdrop { background:rgba(0,0,0,.7); }
  .dlg-body { padding:16px; max-height:60vh; overflow:auto; }
  .dlg-foot { padding:12px 16px; border-top:1px solid var(--line); display:flex; gap:8px; justify-content:flex-end; }
  .diff { font-family:ui-monospace,monospace; font-size:12px; margin-bottom:4px; }
  .diff .a { color:var(--err); } .diff .b { color:var(--ok); }
  .msg { padding:8px 12px; border-radius:6px; margin-bottom:8px; font-size:12px; }
  .msg.err { background:rgba(244,91,105,.12); color:var(--err); }
  .msg.warn { background:rgba(245,165,36,.12); color:var(--warn); }
  #lightbox img { max-width:100%; max-height:80vh; display:block; margin:auto; }
  #toast { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:var(--panel2);
           border:1px solid var(--line); border-radius:8px; padding:10px 16px; display:none; z-index:50; }
</style>
</head>
<body>
<header>
  <h1>차분 정리</h1>
  <select id="ds" title="dataset_raw/ 의 원시 데이터셋"></select>
  <label class="tiny">임계값
    <input type="number" id="thr" step="0.01" min="0.50" max="0.99" title="CLS 코사인 유사도">
  </label>
  <button id="embed" title="DINOv2 임베딩을 다시 계산한다">임베딩</button>
  <span class="meta" id="meta">로딩…</span>
  <span class="spacer"></span>
  <span class="meta" id="pending">변경 0건</span>
  <button id="reset">되돌리기(편집)</button>
  <button id="undo">마지막 적용 취소</button>
  <button id="apply" class="primary">변경사항 적용</button>
</header>
<main id="root"></main>

<dialog id="preview">
  <div class="dlg-body" id="preview-body"></div>
  <div class="dlg-foot">
    <button onclick="document.getElementById('preview').close()">닫기</button>
    <button class="primary" id="confirm">적용</button>
  </div>
</dialog>
<dialog id="lightbox"><div class="dlg-body"><img id="lightbox-img"></div></dialog>
<div id="toast"></div>

<script>
let DATA = null;
let DS = null;       // 현재 데이터셋 이름
let THR = 0.85;      // 현재 임계값
/* 상태는 "이름"이 아니라 "역할"로 들고 있는다. 연결요소가 서로 다른 원본을 엮어버리는
   일이 흔하므로(예: _p0_l 과 _p0_r 이 한 그룹), 기본 역할은 'keep' = 아무것도 안 함 이다.
   최종 파일명은 targets() 가 역할로부터 계산한다. */
/* 'keep' = 아직 안 본 것(기본), 'kept' = 사용자가 "별개다"라고 명시한 것. 파일에 미치는
   효과는 같고(둘 다 이름 유지) 그룹이 끝났는지 판단할 때만 갈린다. */
let ROLE = {};       // rel -> 'keep' | 'kept' | 'orig' | 'sabun' | 'excl' | 'manual'
let MANUAL = {};     // rel -> 직접 입력한 stem ('manual' 일 때만)
let BASE = {};       // groupId -> 원본 base stem (강조 마커 없는 상태)
let SPECIAL_ON = {}; // groupId -> 강조 여부
/* 접힘은 **화면 상태**라 load() 에서만 정한다. 매 render 마다 상태에서 다시 계산하면
   멤버를 제외하는 순간 작업하던 그룹이 눈앞에서 접혀버린다. */
let COLLAPSED = {};  // groupId -> 접힘 여부

const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const toast = m => { const t = document.getElementById('toast');
  t.textContent = m; t.style.display='block'; setTimeout(()=>t.style.display='none', 3000); };

// 규칙: dedup.py 의 SABUN_RE / SABUN_SUFFIX / SPECIAL_MARKER 와 같아야 한다.
// (여기서 틀려도 서버가 0_process_raw 와 동일한 정규식으로 검증해 적용을 막는다 — 여기 것은 표시용)
const sabunName = (base, n) => `${base}_sabun_${n}`;
const SPECIAL = () => DATA.special_marker;
const SABUN = /_sabun(_.*)?$/;        // dedup.SABUN_RE — 0_process_raw 가 인식하는 것
const LOOSE = /_sabun(\d+|_.*)?$/;    // dedup.LOOSE_SABUN_RE — `_sabun1` 류까지

const stemOf = rel => { const n = rel.split('/').pop(); const i = n.lastIndexOf('.');
  return i < 0 ? n : n.slice(0, i); };
const dirOf  = rel => { const i = rel.lastIndexOf('/'); return i < 0 ? '' : rel.slice(0, i+1); };
const stripSpecial = s => s.split(SPECIAL()).join('');
const cleanBase = stem => stripSpecial(stem.replace(LOOSE, ''));

/* 0_process_raw 는 **플래트닝된** 이름으로 판정한다 — 폴더명이 이름 앞에 붙으므로
   `앨범_special/x.png` 는 basename 을 안 건드려도 강조가 된다. 그래서 판정은 flat 기준. */
const flatStem = rel => stemOf(rel.split('/').join('__'));
const isSabun = fs => SABUN.test(fs);
const sabunBase = fs => fs.replace(SABUN, '');
const isMalformed = fs => LOOSE.test(fs) && !SABUN.test(fs);

const setMeta = t => document.getElementById('meta').textContent = t;
const setRoot = h => document.getElementById('root').innerHTML = h;

/* 데이터셋 목록을 채우고 첫 화면을 띄운다. --print 없이 인자 없이 띄운 경우
   default 가 null 이므로 첫 데이터셋을 고른다. */
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
  // 서버가 이미 임베딩 중일 수 있다(인자로 띄운 뒤 새로고침 등) → 먼저 확인.
  const j = await (await fetch('/api/job')).json();
  if (j.running) { DS = j.dataset; sel.value = DS; return poll(); }
  await load();
}

async function load() {
  if (!DS) return setRoot('<p class="tiny">dataset_raw/ 에 원시 데이터셋이 없다.</p>');
  setMeta('불러오는 중…');
  const r = await fetch(`/api/groups?ds=${encodeURIComponent(DS)}&threshold=${THR}`);
  DATA = await r.json();
  if (DATA.error) { setRoot(`<div class="msg err">${esc(DATA.error)}</div>`); setMeta(DS); return; }
  if (DATA.need_embed) return renderNeedEmbed();
  ROLE = {}; MANUAL = {}; BASE = {}; SPECIAL_ON = {}; COLLAPSED = {};
  for (const g of DATA.groups) {
    for (const m of g.members) ROLE[m.rel] = 'keep';
    BASE[g.id] = cleanBase(g.base);
    // 이미 basename 에 마커가 있는 그룹이면 켜 둔 상태로 시작(정리해도 강조가 유지되게).
    SPECIAL_ON[g.id] = !g.inherited_special && g.members.some(m => m.is_special);
  }
  for (const e of DATA.excluded) ROLE[e.rel] = 'excl';
  // 역할이 전부 'keep' 인 지금의 targets() = 디스크 현재 상태. 이미 정리된 그룹은 접고 연다.
  const now = targets();
  for (const g of DATA.groups) COLLAPSED[g.id] = classify(g, now).state === 'done';
  setMeta(`이미지 ${DATA.total_images}장 · 후보 그룹 ${DATA.groups.length}개 (${DATA.grouped_images}장)`
    + (DATA.excluded.length ? ` · 제외 ${DATA.excluded.length}장` : ''));
  document.getElementById('undo').disabled = !DATA.has_undo;
  render();
}

/* 아직 임베딩이 없는 데이터셋 → 여기서 바로 트리거한다. */
function renderNeedEmbed() {
  setMeta(`이미지 ${DATA.total_images}장`);
  setRoot(`<div class="center">
    <p>이 데이터셋은 아직 DINOv2 임베딩이 없다 — <b>${DATA.pending}장</b> 계산이 필요하다.</p>
    <p class="tiny">GPU 로 수십 초 걸린다. 결과는 캐시되므로 다음부터는 바로 열린다.</p>
    <button class="primary" onclick="startEmbed(false)">DINOv2 임베딩 시작</button>
  </div>`);
}

async function startEmbed(refresh) {
  const r = await fetch('/api/embed', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ dataset: DS, refresh }) });
  const res = await r.json();
  if (res.error) return toast(res.error);
  poll();
}

/* 임베딩은 백그라운드 스레드에서 돈다 → 진행률을 폴링해 보여준다. */
async function poll() {
  const j = await (await fetch('/api/job')).json();
  if (j.error) { setRoot(`<div class="msg err">임베딩 실패: ${esc(j.error)}</div>`); return; }
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

/* 역할 → 최종 rel. `_special_` 은 차분 접미사보다 **앞**에 와야 한다: 0_process_raw 의
   is_special 은 `_sabun...` 을 잘라낸 base 에서 마커를 찾기 때문이다.
   base 를 항상 마커 없는 상태로 두고 여기서만 붙이므로 자연히 그렇게 된다. */
function targets() {
  const T = {};
  for (const g of DATA.groups) {
    const orig = g.members.find(m => ROLE[m.rel] === 'orig');
    const base = BASE[g.id] + (SPECIAL_ON[g.id] ? SPECIAL() : '');
    // 그룹이 여러 폴더에 걸쳐 있으면 원본 폴더로 모은다 — 그래야 플래트닝 후 같은 base 가 된다.
    const dir = orig ? dirOf(orig.rel) : '';
    let n = 0;
    for (const m of g.members) {
      const r = ROLE[m.rel];
      // 제외는 상대경로째로 옮긴다. basename 만 쓰면 (1) 다른 폴더의 동명 파일끼리 충돌하고
      // (2) 복구할 때 앨범 폴더를 잃어 폴더 상속 강조까지 날아간다.
      if      (r === 'excl')   T[m.rel] = DATA.exclude_prefix + m.rel;
      else if (r === 'orig')   T[m.rel] = dir + base + m.ext;
      else if (r === 'sabun')  T[m.rel] = dir + sabunName(base, ++n) + m.ext;
      else if (r === 'manual') T[m.rel] = dirOf(m.rel) + MANUAL[m.rel] + m.ext;
      else                     T[m.rel] = m.rel;   // 'keep' / 'kept' — 손대지 않는다
    }
  }
  // 복구는 제외 시 보존해 둔 상대경로 그대로 되돌린다 (무손실 왕복).
  for (const e of DATA.excluded)
    T[e.rel] = ROLE[e.rel] === 'excl' ? e.rel : e.rel.slice(DATA.exclude_prefix.length);
  return T;
}
const changed = () => Object.entries(targets()).filter(([a, b]) => a !== b);

/* 최종 이름이 0_process_raw 에게 어떻게 보이는가. 판정은 파일 하나가 아니라 **차분 base
   단위**다: base + `_sabun_*` 가 다 합쳐 R 회, 표준 이미지는 각자 R 회. 즉 여기서 '표준'
   으로 남는 중복이 곧 과대표집이다.

   targets() 는 그룹 멤버만 담지만 그래도 맞다 — find_groups 가 loose_base 로 시드하므로
   같은 base 를 쓰는 파일은 반드시 같은 그룹에 들어와 있다.

   접힘 초기값 말고는 전부 이걸로 그린다. 사용자가 원본을 고르는 순간 뱃지가 '이미 차분
   그룹'으로 바뀌는 게 곧 "이 그룹 끝났다"는 신호가 된다. */
function classify(g, T) {
  const live = g.members.filter(m => ROLE[m.rel] !== 'excl');
  const stems = new Map(live.map(m => [m.rel, flatStem(T[m.rel])]));
  const perBase = new Map();        // 차분 base -> {n: 그 base 를 쓰는 파일 수, sabun: 그중 차분 수}
  for (const fs of stems.values()) {
    const b = sabunBase(fs);
    const e = perBase.get(b) || { n: 0, sabun: 0 };
    e.n++;
    if (isSabun(fs)) e.sabun++;
    perBase.set(b, e);
  }
  const kind = new Map();           // rel -> 'orig' | 'sabun' | 'std'
  for (const [rel, fs] of stems)
    kind.set(rel, perBase.get(sabunBase(fs)).sabun ? (isSabun(fs) ? 'sabun' : 'orig') : 'std');

  const nSabun = [...perBase.values()].reduce((a, e) => a + e.sabun, 0);
  // 차분만 있고 원본이 없는 base → 0_process_raw.py:126 이 SystemExit 한다. 적용 시 서버도
  // 막지만, 원본을 제외하는 순간 바로 보이는 게 낫다.
  const orphan = [...perBase.values()].some(e => e.sabun && e.sabun === e.n);
  const malformed = [...stems.values()].some(isMalformed);
  /* 연결요소는 서로 다른 원본을 엮어놓기도 한다(_p0_l + _p0_r 등). 그런 그룹에서 무관한
     멤버를 '유지'로 빼면 판단이 끝난 것이므로 더 채근하지 않는다. 아직 손대지 않은
     ('keep') 표준만 '확인 필요'로 센다 — 안 그러면 카운터가 0 이 될 수 없다. */
  const open = live.some(m => ROLE[m.rel] === 'keep' && kind.get(m.rel) === 'std');

  let state, label, cls;
  if (orphan)               [state, label, cls] = ['orphan', '원본 없음 — 이대로면 0_process_raw 가 멈춘다', 'err'];
  else if (malformed)       [state, label, cls] = ['partial', '비규격 차분 이름 — 0_process_raw 가 못 알아본다', 'warn'];
  else if (live.length <= 1) [state, label, cls] = ['done', '정리됨', 'ok'];
  else if (!open)           [state, label, cls] = ['done', nSabun ? '이미 차분 그룹' : '정리됨', 'ok'];
  else if (nSabun)          [state, label, cls] = ['partial', '일부만 표기됨', 'warn'];
  else                      [state, label, cls] = ['none', '미표기 — 전부 표준 취급', ''];
  return { kind, state, label, cls };
}

/* 원본 지정 = 그룹 일괄 정리. 나머지는 차분으로 돌린다(제외/수동은 건드리지 않는다).
   잘못 엮인 멤버는 사용자가 개별로 '유지' 를 눌러 빼면 된다. */
function setOrig(g, rel) {
  BASE[g.id] = cleanBase(stemOf(rel));
  for (const m of g.members) {
    if (ROLE[m.rel] === 'excl' || ROLE[m.rel] === 'manual') continue;
    ROLE[m.rel] = m.rel === rel ? 'orig' : 'sabun';
  }
  render();
}
function setRole(rel, role) { ROLE[rel] = role; render(); }
function toggleExclude(rel) { ROLE[rel] = ROLE[rel] === 'excl' ? 'keep' : 'excl'; render(); }
function toggleSpecial(g) { SPECIAL_ON[g.id] = !SPECIAL_ON[g.id]; render(); }
function toggleCollapse(id) { COLLAPSED[id] = !COLLAPSED[id]; render(); }
function foldDone(v) {
  const T = targets();
  for (const g of DATA.groups) if (classify(g, T).state === 'done') COLLAPSED[g.id] = v;
  render();
}

function render() {
  const root = document.getElementById('root');
  const T = targets();
  const C = {};
  for (const g of DATA.groups) C[g.id] = classify(g, T);
  const done = DATA.groups.filter(g => C[g.id].state === 'done');

  let html = '';
  if (done.length) {
    const folded = done.every(g => COLLAPSED[g.id]);
    html += `<div class="foldbar">
      <span class="tiny">이미 차분 그룹 <b>${done.length}</b>개 ·
        확인 필요 <b>${DATA.groups.length - done.length}</b>개</span>
      <span class="spacer"></span>
      <button onclick="foldDone(${!folded})">
        ${folded ? `정리된 그룹 ${done.length}개 펼치기` : '정리된 그룹 접기'}</button>
    </div>`;
  }
  for (const g of DATA.groups) {
    const st = C[g.id], shut = COLLAPSED[g.id];
    html += `<div class="group ${st.state}"><div class="ghead">
      <button class="caret" onclick="toggleCollapse(${g.id})">${shut ? '&#9656;' : '&#9662;'}</button>
      ${shut ? `<img class="mini" loading="lazy" src="/thumb?ds=${encodeURIComponent(DS)}&rel=${encodeURIComponent(g.anchor)}&s=80"
                     onclick="zoom('${encodeURIComponent(g.anchor)}')">` : ''}
      <span class="gbase">${esc(BASE[g.id])}</span>
      <span class="tiny">${g.members.length}장</span>
      <span class="badge ${st.cls}">${esc(st.label)}</span>
      ${g.inherited_special ? '<span class="badge ok">강조(폴더 상속)</span>' : ''}
      <span class="spacer"></span>
      ${shut || g.inherited_special ? ''
        : `<button onclick="toggleSpecial(DATA.groups[${g.id}])">
             ${SPECIAL_ON[g.id] ? '강조 해제' : '강조(2R)'}</button>`}
    </div>`;
    if (shut) { html += '</div>'; continue; }
    if (g.multi_dir) html += `<div class="note">이 그룹은 여러 폴더에 걸쳐 있다. 원본을 지정하면
      플래트닝 후 같은 차분 그룹이 되도록 <b>원본 폴더로 모은다</b>.</div>`;
    html += '<div class="tiles">';
    for (const m of g.members) {
      const role = ROLE[m.rel], t = T[m.rel], stem = stemOf(t);
      const kind = st.kind.get(m.rel);
      // 뱃지는 "무슨 역할을 줬나"가 아니라 "적용 후 0_process_raw 가 뭘로 볼까"를 보여준다.
      // 그래야 이미 표기된 그룹이 손 안 대도 차분으로 보이고, 안 묶인 중복이 '표준'으로
      // 드러난다(= 각자 R 회 = 과대표집).
      let badge = role === 'excl' ? '<span class="badge err">제외</span>'
        : { orig:'<span class="badge ok">원본</span>',
            sabun:'<span class="badge ok">차분</span>',
            std:'<span class="badge">표준</span>' }[kind];
      if (role !== 'excl' && isMalformed(flatStem(t))) badge += '<span class="badge err">규칙위반</span>';
      if (role === 'manual') badge += '<span class="badge warn">수동</span>';
      html += `<div class="tile ${kind === 'orig' && role !== 'excl' ? 'orig' : ''} ${role === 'excl' ? 'excl' : ''}">
        <img class="thumb" loading="lazy" src="/thumb?ds=${encodeURIComponent(DS)}&rel=${encodeURIComponent(m.rel)}"
             onclick="zoom('${encodeURIComponent(m.rel)}')">
        <div class="row">
          <span class="sim">${m.sim !== null ? (m.sim * 100).toFixed(1) + '%' : ''}</span>
          <span class="spacer"></span>
          ${badge}
        </div>
        <input type="text" class="${t !== m.rel ? 'changed' : ''}" value="${esc(stem)}"
               onchange="editName('${esc(m.rel)}', this.value)">
        <div class="tiny">${m.w}×${m.h} · ${(m.bytes/1024).toFixed(0)}KB</div>
        <div class="row">
          <button style="flex:1" ${role === 'orig' ? 'disabled' : ''}
                  onclick="setOrig(DATA.groups[${g.id}], '${esc(m.rel)}')">원본으로</button>
          <button ${role === 'kept' ? 'disabled' : ''} title="이 그룹과는 별개다 — 이름 그대로 두고 확인 완료로 친다"
                  onclick="setRole('${esc(m.rel)}', 'kept')">유지</button>
          <button class="danger" onclick="toggleExclude('${esc(m.rel)}')">
            ${role === 'excl' ? '복구' : '제외'}</button>
        </div>
      </div>`;
    }
    html += '</div></div>';
  }
  if (DATA.excluded.length) {
    html += `<div class="group"><div class="ghead"><span class="gbase">제외됨</span>
      <span class="tiny">${DATA.excluded.length}장</span></div><div class="tiles">`;
    for (const e of DATA.excluded) {
      html += `<div class="tile ${ROLE[e.rel] === 'excl' ? 'excl' : ''}">
        <img class="thumb" loading="lazy" src="/thumb?ds=${encodeURIComponent(DS)}&rel=${encodeURIComponent(e.rel)}"
             onclick="zoom('${encodeURIComponent(e.rel)}')">
        <div class="tiny">${esc(e.name)}</div>
        <button onclick="toggleExclude('${esc(e.rel)}')">
          ${ROLE[e.rel] === 'excl' ? '복구' : '제외 유지'}</button></div>`;
    }
    html += '</div></div>';
  }
  root.innerHTML = html || '<p class="tiny">후보 그룹이 없다. 위 임계값을 낮추면 더 묶인다.</p>';
  document.getElementById('pending').textContent = `변경 ${changed().length}건`;
}

function editName(rel, value) {
  const v = value.trim();
  if (!v || v === stemOf(rel)) { ROLE[rel] = 'keep'; }
  else { ROLE[rel] = 'manual'; MANUAL[rel] = v; }
  render();
}
function zoom(relEnc) {
  document.getElementById('lightbox-img').src =
    `/full?ds=${encodeURIComponent(DS)}&rel=${relEnc}`;
  document.getElementById('lightbox').showModal();
}
document.getElementById('lightbox').onclick = e => e.target.id === 'lightbox' && e.target.close();

async function post(path) {
  const r = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ dataset: DS, targets: Object.fromEntries(changed()) }) });
  return r.json();
}

document.getElementById('apply').onclick = async () => {
  if (!changed().length) return toast('변경사항이 없다.');
  const res = await post('/api/check');
  let html = '';
  if ((res.errors || []).length || (res.warnings || []).length)
    html += `<p class="tiny">검증은 <b>적용 후 데이터셋 전체</b> 기준이다 —
             이번에 손대지 않은 파일의 기존 문제도 함께 나온다.</p>`;
  for (const e of res.errors || []) html += `<div class="msg err">오류: ${esc(e)}</div>`;
  for (const w of res.warnings || []) html += `<div class="msg warn">경고: ${esc(w)}</div>`;
  html += `<p class="tiny">${res.moves.length}건 리네임:</p>`;
  for (const [a, b] of res.moves)
    html += `<div class="diff"><span class="a">${esc(a)}</span> → <span class="b">${esc(b)}</span></div>`;
  document.getElementById('preview-body').innerHTML = html;
  document.getElementById('confirm').disabled = (res.errors || []).length > 0;
  document.getElementById('preview').showModal();
};

document.getElementById('confirm').onclick = async () => {
  const res = await post('/api/apply');
  document.getElementById('preview').close();
  if (res.errors && res.errors.length) return toast('오류로 적용 취소됨');
  toast(`${res.applied}건 적용됨`);
  await load();
};

document.getElementById('reset').onclick = () => load();
document.getElementById('ds').onchange = e => { DS = e.target.value; load(); };
// 임계값은 임베딩과 무관하다 — 그룹핑만 다시 하므로 즉시 반영된다.
document.getElementById('thr').onchange = e => {
  const v = parseFloat(e.target.value);
  if (!isNaN(v) && v > 0 && v < 1) { THR = v; load(); } else { e.target.value = THR.toFixed(2); }
};
document.getElementById('embed').onclick = () => {
  if (confirm(`${DS} 의 임베딩을 다시 계산한다. 캐시를 버리고 전부 다시 돌린다.`)) startEmbed(true);
};
document.getElementById('undo').onclick = async () => {
  const r = await fetch('/api/undo', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ dataset: DS }) });
  const res = await r.json();
  toast(res.message || res.error);
  await load();
};

boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
