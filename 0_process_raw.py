#!/usr/bin/env python3
"""[0] dataset_raw/<원시> 를 dataset/<name> 으로 플래트닝하는 전처리 스크립트.

dataset_raw/PROCESS_RAW.md 규칙을 구현한다. 원시 데이터셋의 하위 디렉토리를 재귀적으로 훑어
모든 이미지를 "경로구분자 / → __" 로 플래트닝한 이름의 심볼릭 링크로 dataset/<name>/repeat_<N>/
아래에 배치한다. N(반복수)은 아래 규칙으로 이미지마다 결정된다.

이미지 분류(플래트닝된 basename 기준):
  - 차분 이미지 그룹: 접미사가 `_sabun` 또는 `_sabun_<>` 인 변형들 + 그 원본(접미사 없는 것).
      · 그룹 전체 반복수의 합이 표준 이미지 1장과 비슷해지도록, 각 이미지는 R/K 로 나눈다(K=그룹 크기).
      · 원본 없이 `_sabun_<>` 만 있으면 오류 → 중단.
  - 강조 이미지: basename 에 `_special_` 문자열이 있으면(=원시에서 special/ 폴더나 *_special 앨범
    아래에 있던 것이 플래트닝되며 __special__ 로 바뀐 것). 반복수 2배.
      · `special` 문자열은 있는데 `_special_` 마커가 아니면 경고만 하고 표준으로 취급.
  - 표준 이미지: 위 어디에도 안 걸리면 반복수 R.
  - 차분 + 강조 동시 가능: 그룹 전체 합이 2R 이 되도록 각 이미지 2R/K.

반복수(R/K, 2R/K)가 정수로 안 떨어지면 --rounding(ceil|floor)로 반올림한다(최소 1).

사용 예:
    # dataset_raw/mystyle_raw → dataset/mystyle, 표준 반복 10, 올림
    uv run python 0_process_raw.py mystyle_raw --repeat 10

    # 출력 이름/내림 지정 + 덮어쓰기, 먼저 결과만 확인
    uv run python 0_process_raw.py mystyle_raw --name mystyle --rounding floor --force --dry-run

다음 단계: 1_tag_dataset.py <name> --trigger @<name> --recursive → 2_make_config.py <name>
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "dataset_raw"
DATASET_ROOT = ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# 접미사 `_sabun` / `_sabun_<무엇이든>` 을 스템 끝에서 잡아낸다.
SABUN_RE = re.compile(r"_sabun(?:_.*)?$")


def flatten_name(raw_dir: Path, img: Path) -> str:
    """raw_dir 기준 상대경로의 / 를 __ 로 치환한 플래트닝 파일명."""
    rel = img.relative_to(raw_dir)
    return str(rel).replace(os.sep, "__")


def sabun_base(stem: str) -> str:
    """차분 접미사를 제거한 그룹 base. `foo_sabun_unreal` → `foo`."""
    return SABUN_RE.sub("", stem)


def resolve_raw(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    candidate = (RAW_ROOT / arg).resolve()
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"[error] raw dataset not found: {arg} (dataset_raw/{arg} 확인)")


def collect_images(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class Member:
    __slots__ = ("src", "flat", "stem", "is_sabun")

    def __init__(self, src: Path, flat: str):
        self.src = src
        self.flat = flat
        self.stem = flat[: -len(Path(flat).suffix)] if Path(flat).suffix else flat
        self.is_sabun = SABUN_RE.search(self.stem) is not None


def round_repeat(total: int, k: int, rounding: str) -> int:
    """그룹 전체 반복수 total 을 k 로 나눈 개별 반복수(최소 1)."""
    val = total / k
    n = math.ceil(val) if rounding == "ceil" else math.floor(val)
    return max(1, n)


def plan(raw_dir: Path, repeat: int, rounding: str):
    """이미지별 (Member, 반복수 N) 목록과 경고/오류를 계산한다."""
    images = collect_images(raw_dir)
    if not images:
        raise SystemExit(f"[error] no images under {raw_dir}")

    # 플래트닝 → 그룹핑 (base 별)
    groups: dict[str, list[Member]] = defaultdict(list)
    collisions: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        m = Member(img, flatten_name(raw_dir, img))
        collisions[m.flat].append(img)
        groups[sabun_base(m.stem)].append(m)

    dups = {name: paths for name, paths in collisions.items() if len(paths) > 1}
    if dups:
        for name, paths in dups.items():
            print(f"[error] 플래트닝 파일명 충돌: {name}")
            for p in paths:
                print(f"          {p}")
        raise SystemExit("[error] 이름 충돌로 중단. 원시 폴더/파일명을 정리하라.")

    assignments: list[tuple[Member, int]] = []
    warnings: list[str] = []
    n_std = n_special = n_sabun_img = 0

    for base, members in sorted(groups.items()):
        has_sabun = any(m.is_sabun for m in members)
        has_original = any(not m.is_sabun for m in members)
        is_special = "_special_" in base
        if (not is_special) and ("special" in base.lower()):
            warnings.append(f"'special' 문자열이 있으나 강조 마커(_special_)가 아님 → 표준 취급: {base}")

        if has_sabun and not has_original:
            for m in members:
                print(f"[error]   차분 변형: {m.flat}")
            raise SystemExit(f"[error] 차분 그룹에 원본이 없다(원본 {base}.* 누락): {base}")

        if has_sabun:
            k = len(members)
            total = repeat * 2 if is_special else repeat
            n = round_repeat(total, k, rounding)
            for m in members:
                assignments.append((m, n))
            n_sabun_img += k
        else:
            # 차분 아님 → 각 이미지 독립 (보통 K=1)
            for m in members:
                n = repeat * 2 if is_special else repeat
                assignments.append((m, n))
                if is_special:
                    n_special += 1
                else:
                    n_std += 1

    stats = {
        "images": len(images), "groups": len(groups),
        "standard": n_std, "special": n_special, "sabun_imgs": n_sabun_img,
    }
    return assignments, warnings, stats


def clear_output(out_dir: Path) -> None:
    """기존 repeat_* 서브셋(우리가 만든 심볼릭 구조)만 정리한다."""
    for sub in out_dir.iterdir():
        if sub.is_dir() and re.fullmatch(r"repeat_\d+", sub.name):
            for f in sub.iterdir():
                f.unlink()
            sub.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description="[0] dataset_raw/<원시> → dataset/<name> 플래트닝(심볼릭 링크)")
    ap.add_argument("raw", help="dataset_raw/<이름> 의 원시 데이터셋 이름 또는 경로 (예: mystyle_raw)")
    ap.add_argument("--name", default=None,
                    help="출력 데이터셋 이름 (기본: raw 이름에서 끝의 '_raw' 제거)")
    ap.add_argument("--repeat", type=int, default=10,
                    help="표준 이미지의 반복수 R (강조=2R, 차분그룹=합이 R). 기본 10")
    ap.add_argument("--rounding", choices=["ceil", "floor"], default="ceil",
                    help="차분 그룹 개별 반복수 반올림 정책 (기본 ceil)")
    ap.add_argument("--absolute", action="store_true",
                    help="심볼릭 링크 타깃을 절대경로로 (기본: 레포 이동에 강한 상대경로)")
    ap.add_argument("--force", action="store_true", help="기존 dataset/<name> 의 repeat_* 서브셋 덮어쓰기")
    ap.add_argument("--dry-run", action="store_true", help="링크를 만들지 않고 계획만 출력")
    args = ap.parse_args()

    raw_dir = resolve_raw(args.raw)
    name = args.name or re.sub(r"_raw$", "", raw_dir.name)
    out_dir = DATASET_ROOT / name

    assignments, warnings, stats = plan(raw_dir, args.repeat, args.rounding)

    # 반복수별 분포
    dist: dict[int, int] = defaultdict(int)
    for _m, n in assignments:
        dist[n] += 1

    print(f"[plan] raw={raw_dir}")
    print(f"[plan] out=dataset/{name}  repeat(R)={args.repeat}  rounding={args.rounding}")
    print(f"[plan] images={stats['images']}  groups={stats['groups']}  "
          f"(표준 {stats['standard']}, 강조 {stats['special']}, 차분 {stats['sabun_imgs']})")
    print(f"[plan] 반복수 분포(repeat_N: 이미지수): "
          + ", ".join(f"repeat_{n}:{c}" for n, c in sorted(dist.items())))
    total_reps = sum(n * c for n, c in dist.items())
    print(f"[plan] 에폭당 총 학습 이미지(반복 반영) ≈ {total_reps}")
    # 배치별 에폭당 스텝수 = ceil(반복반영 이미지수 / batch). 총 스텝 = 이 값 × epochs.
    # (enable_bucket 시 해상도 버킷별 반올림으로 실제값이 조금 더 클 수 있음)
    steps_line = "  ".join(f"b{b}:{math.ceil(total_reps / b)}" for b in (1, 2, 4, 8, 16))
    print(f"[plan] 에폭당 스텝수(batch별, ceil): {steps_line}")
    for w in warnings:
        print(f"[warn] {w}")

    if args.dry_run:
        print("[dry-run] 링크 생성 생략.")
        return 0

    if out_dir.exists():
        has_repeat = any(re.fullmatch(r"repeat_\d+", p.name) for p in out_dir.iterdir() if p.is_dir())
        if has_repeat and not args.force:
            raise SystemExit(f"[error] 이미 존재(--force 로 덮어쓰기): {out_dir}")
        if has_repeat:
            clear_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for m, n in assignments:
        sub = out_dir / f"repeat_{n}"
        sub.mkdir(exist_ok=True)
        link = sub / m.flat
        if args.absolute:
            target = m.src
        else:
            target = Path(os.path.relpath(m.src, sub))
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(target, link)
        created += 1

    print(f"[done] {created} 개 심볼릭 링크 생성 → {out_dir}")
    print(f"  next: uv run python 1_tag_dataset.py {name} --trigger @{name} --recursive")
    print(f"        uv run python 2_make_config.py {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
