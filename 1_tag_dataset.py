#!/usr/bin/env python3
"""WD 태깅 -> blacklist 제거 -> trigger prepend 까지 한 번에 수행하는 데이터셋 준비 스크립트.

태깅 로직은 app.py 의 WD14Tagger 를 그대로 재사용하므로 웹 에디터와 100% 동일한
모델/전처리/threshold 동작을 보장한다.

사용 예:
    # dataset/mychar 의 이미지 태깅, blacklist.txt 적용, 트리거 @mychar 를 맨 앞에
    .venv/bin/python 1_tag_dataset.py mychar --trigger @mychar

    # 임의 경로 + 기존 캡션 덮어쓰기 + 먼저 결과만 확인(dry-run)
    .venv/bin/python 1_tag_dataset.py /path/to/imgs --trigger @mychar --overwrite --dry-run

결과: 각 이미지 옆에 <이미지이름>.txt (쉼표 구분, trigger 가 항상 첫 토큰).
keep_tokens=1 과 맞물려 trigger 가 셔플에서 고정된다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# app.py 의 태깅 구현을 그대로 사용 (모델/전처리 일치 보장)
from app import IMAGE_EXTENSIONS, IMAGE_ROOT, WD14Tagger


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


def resolve_dataset(arg: str) -> Path:
    """경로면 그대로, 아니면 dataset/<이름> 으로 해석."""
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    candidate = (IMAGE_ROOT / arg).resolve()
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"[error] dataset/dir not found: {arg}")


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def has_repeat_subsets(folder: Path) -> bool:
    """0_process_raw.py 가 만든 repeat_<N> 서브셋 구조인지."""
    return any(p.is_dir() and p.name.startswith("repeat_") for p in folder.iterdir())


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


def main() -> int:
    ap = argparse.ArgumentParser(description="WD tag -> blacklist 제거 -> trigger prepend")
    ap.add_argument("dataset", help="dataset/<이름> 의 이름 또는 이미지 폴더 경로")
    ap.add_argument("--trigger", required=True, help="맨 앞에 넣을 트리거 (예: @mychar)")
    ap.add_argument("--blacklist", default=str(Path(__file__).resolve().parent / "blacklist.txt"),
                    help="blacklist 파일 (기본: 레포 blacklist.txt)")
    ap.add_argument("--general-threshold", type=float, default=0.35)
    ap.add_argument("--character-threshold", type=float, default=0.85)
    ap.add_argument("--include-rating", action="store_true",
                    help="예측된 rating 태그(general/sensitive/...)를 trigger 다음에 삽입")
    ap.add_argument("--overwrite", action="store_true", help="기존 .txt 덮어쓰기")
    ap.add_argument("--recursive", action="store_true",
                    help="하위 폴더까지 재귀 태깅 (0_process_raw.py 의 repeat_<N> 구조용)")
    ap.add_argument("--dry-run", action="store_true", help="파일 안 쓰고 결과만 출력")
    args = ap.parse_args()

    folder = resolve_dataset(args.dataset)
    banned = load_blacklist(Path(args.blacklist))
    # repeat_<N> 서브셋 구조면 재귀 태깅을 자동 활성화
    recursive = args.recursive or has_repeat_subsets(folder)
    images = list_images(folder, recursive)
    if not images:
        raise SystemExit(f"[error] no images in {folder}")

    print(f"[info] dir={folder}  images={len(images)}  blacklist={len(banned)} tags  "
          f"trigger='{args.trigger}'  dry_run={args.dry_run}")

    tagger = WD14Tagger()
    removed_counter: dict[str, int] = {}
    processed = skipped = failed = 0

    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            result = tagger.tag_image(img, args.general_threshold, args.character_threshold)
        except Exception as exc:  # 이미지 손상 등은 건너뛰고 계속
            print(f"[warn] {img.name}: {exc}")
            failed += 1
            continue

        raw_tags = [t for t in result["tags"].split(",")]
        rating = None
        if args.include_rating and result.get("rating"):
            rating = result["rating"]["name"]

        caption = build_caption(raw_tags, args.trigger, banned, rating, removed_counter)

        if args.dry_run:
            print(f"\n# {img.name}\n{caption}")
        else:
            txt.write_text(caption, encoding="utf-8")
        processed += 1

    print(f"\n[done] processed={processed} skipped={skipped} failed={failed}")
    if removed_counter:
        print("[removed by blacklist] (tag x count)")
        for tag, n in sorted(removed_counter.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {n:5d}  {tag}")
    else:
        print("[removed by blacklist] none matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
