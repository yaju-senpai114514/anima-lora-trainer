#!/usr/bin/env python3
"""[0] 데이터셋 프로세싱 웹 서버 — dedup 그룹핑/익스포트 + WD14 태깅 + toml 생성 통합.

구 0_dedup_raw.py / 1_tag_dataset.py / 2_make_config.py 를 하나의 서버로 병합했다.
얇은 CLI 래퍼다 — 로직은 src/dedup.py · src/tagging.py · src/configgen.py, 서버/프론트엔드는
src/webui/. 원본(dataset_raw/)은 완전 read-only, dedup 현황은 .dedup/<name>/state.json 으로만
관리한다. 웹 UI 에서 toml 까지 만들면 다음은 ./3_run_training.sh <name>.

워크플로우 (웹 UI 탭 2개):
  원시 dedup 탭 : dataset_raw/<원시> → DINOv2 임베딩 → 그룹핑/제외/강조 →
                  dataset/<name>/repeat_<N>/ 심볼릭 링크 익스포트
  데이터셋 탭   : dataset/<name>/ → WD14 태깅(blacklist/trigger, WD 추론 캐시) →
                  캡션 검수/수정 → dataset/<name>.toml 생성 (batch_size 미기재)

사용 예:
    uv run python 0_dataset_server.py                 # → http://127.0.0.1:8765
    uv run python 0_dataset_server.py mychar          # 미리 임베딩 + 요약 출력 후 UI
    uv run python 0_dataset_server.py mychar --print  # 서버 없이 dedup 현황만 출력
"""
from __future__ import annotations

import argparse
import sys

from src import dedup
from src.webui.server import print_report, serve, state_payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="[0] 데이터셋 프로세싱 웹 서버 (dedup 그룹핑/익스포트 · WD14 태깅 · toml 생성)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("raw", nargs="?", default=None,
                    help="dataset_raw/<이름> 또는 경로. 주면 미리 임베딩한다. 생략하면 웹 UI 에서 고른다")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="dedup 후보 쌍 centroid 코사인 임계값. 낮출수록 후보가 많아진다 (기본 0.85)")
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
                    help="서버를 띄우지 않고 dedup 현황만 출력")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="브라우저 자동 실행")
    args = ap.parse_args()

    if args.print_only and not args.raw:
        raise SystemExit("[error] --print 는 데이터셋 이름이 필요하다 (예: 0_dataset_server.py mychar --print)")

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
        print(f"[warn] dataset_raw/ 아래에 원시 데이터셋이 없다: {dedup.RAW_ROOT} — 데이터셋 탭만 쓸 수 있다.")

    serve(args.host, args.port, default_ds=default_ds, threshold=args.threshold,
          batch_size=batch_size, open_browser=args.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
