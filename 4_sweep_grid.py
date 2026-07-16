#!/usr/bin/env python3
"""[4] 학습 완료된 LoRA 들을 에폭별로 적용해 그리드 스윕을 돌리는 파이프라인 스크립트.

3_run_training.sh 가 output/<name>/ 에 찍어둔 에폭 체크포인트들
(anima_<name>_r<dim>-NNNNNN.safetensors + 최종 anima_<name>_r<dim>.safetensors)을 모아,
같은 프롬프트 묶음을 여러 에폭에 적용해 생성한 뒤 "행=에폭 × 열=프롬프트" 그리드 PNG 로 묶는다.

흐름:
    1) config.env 에서 모델 경로 / NETWORK_DIM / MAX_TRAIN_EPOCHS 를 읽는다.
    2) output/<name>/ 의 에폭 체크포인트를 스캔하고, 최종 에폭을 포함해 N개(기본 4)를 고른다.
    3) 프롬프트 묶음을 만든다. 두 가지 모드:
         --mode samples : samples.txt(또는 --samples-file)에 적어둔 프롬프트를 사용.
         --mode tags    : dataset/<name>/ 의 캡션(.txt) 태그에서 적당히 샘플링해 프롬프트를 만듦.
       각 프롬프트(=그리드 한 컬럼)에는 그리드 매칭용 시드(--d)가 1개씩 배정된다.
    4) 에폭마다 sd-scripts/anima_minimal_inference.py 를 --from_file 배치 모드로 돌려
       output/<name>/<out>/ep<NN>/ 에 이미지를 생성한다.
    5) 시드로 이미지를 매칭해 그리드 PNG(grid_<name>.png)를 합성한다.

사용 예:
    # samples.txt 의 프롬프트로, 최종 포함 4개 에폭 스윕
    uv run python 4_sweep_grid.py mychar

    # 데이터셋 태그에서 5개 프롬프트를 샘플링, 에폭 6개
    uv run python 4_sweep_grid.py mychar --mode tags --num-prompts 5 --num-epochs 6

    # 에폭 직접 지정 + 생성은 건너뛰고 기존 결과로 그리드만 다시 합성
    uv run python 4_sweep_grid.py mychar --epochs 12,16,20,24 --grid-only
"""
from __future__ import annotations

import argparse
import glob
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset"
OUTPUT_ROOT = ROOT / "output"
INFERENCE = ROOT / "sd-scripts" / "anima_minimal_inference.py"

# 프롬프트 줄 끝 오버라이드 기본값 (anima_minimal_inference.parse_prompt_line 규약).
DEFAULT_NEGATIVE = "lowres, worst quality, low quality, bad anatomy, jpeg artifacts"
OVERRIDE_FLAGS = ("d", "s", "g", "l", "w", "h", "n", "fs")  # 줄에 이미 존재하는지 검사할 플래그
# samples.txt 안에서 trigger(@<name>) 자리로 쓰는 토큰. 스윕 시 --trigger 값으로 치환된다.
TRIGGER_PLACEHOLDER = "<trigger_placeholder>"


# ----------------------------------------------------------------------------- config
def load_config() -> dict[str, str]:
    """config.env 를 bash 로 source 해서 필요한 값만 뽑아온다($HOME/변수 확장 보존)."""
    keys = ["PATH_MODEL", "PATH_QWEN", "PATH_VAE", "NETWORK_DIM", "MAX_TRAIN_EPOCHS"]
    script = f'source "{ROOT / "config.env"}" && ' + " && ".join(f'echo "${k}"' for k in keys)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    values = out.splitlines()
    if len(values) < len(keys):
        raise SystemExit(f"[error] config.env 파싱 실패: {out!r}")
    return dict(zip(keys, values))


# ------------------------------------------------------------------- checkpoint 스캔/선택
def discover_checkpoints(name: str, dim: str, max_epochs: int) -> dict[int, Path]:
    """output/<name>/ 에서 {에폭: 체크포인트경로} 를 모은다. 최종 체크포인트 = MAX_TRAIN_EPOCHS."""
    out_dir = OUTPUT_ROOT / name
    if not out_dir.is_dir():
        raise SystemExit(f"[error] output dir not found: {out_dir} (먼저 3_run_training.sh {name} 실행)")

    ckpts: dict[int, Path] = {}
    pat = re.compile(rf"^anima_{re.escape(name)}_r{dim}-(\d+)\.safetensors$")
    for p in out_dir.glob(f"anima_{name}_r{dim}-*.safetensors"):
        m = pat.match(p.name)
        if m:
            ckpts[int(m.group(1))] = p

    final = out_dir / f"anima_{name}_r{dim}.safetensors"
    if final.exists():
        ckpts[max_epochs] = final  # 최종 LoRA = MAX_TRAIN_EPOCHS 에폭

    if not ckpts:
        raise SystemExit(f"[error] {out_dir} 에 anima_{name}_r{dim}*.safetensors 체크포인트가 없다.")
    return ckpts


def select_epochs(available: list[int], num: int, explicit: list[int] | None) -> list[int]:
    """최종 에폭을 포함해 num 개의 에폭을 균등하게 고른다."""
    available = sorted(available)
    if explicit:
        chosen = [e for e in explicit if e in available]
        missing = [e for e in explicit if e not in available]
        if missing:
            print(f"[warn] 존재하지 않는 에폭 무시: {missing} (가용: {available})")
        if not chosen:
            raise SystemExit(f"[error] 지정한 에폭 중 가용한 것이 없다. 가용: {available}")
        return sorted(chosen)

    # 최종 에폭부터 뒤에서 num 개 (가장 잘 학습된 후반 에폭들)
    num = max(1, min(num, len(available)))
    return available[-num:]


# ------------------------------------------------------------------------ 프롬프트 빌드
def compose_line(prompt: str, present: dict[str, str], orig: str,
                 seed: int, defaults: dict) -> tuple[str, int]:
    """줄에 없는 오버라이드 플래그만 기본값/시드로 채워 최종 prompts.txt 한 줄을 만든다.

    returns: (최종 줄, 그리드 매칭에 쓸 시드)
    """
    line = orig
    extra: list[str] = []
    if "d" not in present:
        extra.append(f"--d {seed}")
    if "s" not in present:
        extra.append(f"--s {defaults['steps']}")
    if "g" not in present and "l" not in present:
        extra.append(f"--g {defaults['guidance']}")
    if "w" not in present:
        extra.append(f"--w {defaults['width']}")
    if "h" not in present:
        extra.append(f"--h {defaults['height']}")
    if "n" not in present:
        extra.append(f"--n {defaults['negative']}")
    if extra:
        line = f"{line} {' '.join(extra)}"
    final_seed = int(present["d"]) if "d" in present else seed
    return line, final_seed


def parse_present_flags(line: str) -> tuple[str, dict[str, str]]:
    """' --' 로 분리해 프롬프트 본문과 이미 존재하는 플래그를 추출."""
    parts = line.split(" --")
    prompt = parts[0].strip()
    present: dict[str, str] = {}
    for part in parts[1:]:
        opt = part.split(" ", 1)[0].strip()
        if opt in OVERRIDE_FLAGS:
            present[opt] = part.split(" ", 1)[1].strip() if " " in part else ""
    return prompt, present


def build_from_samples(samples_file: Path, defaults: dict, seed_base: int, trigger: str):
    """samples.txt 의 각 줄을 프롬프트 컬럼으로. 줄에 없는 플래그는 기본값으로 채운다.

    <trigger_placeholder> 토큰은 trigger(예: @mychar)로 치환된다.
    """
    if not samples_file.is_file():
        raise SystemExit(f"[error] samples 파일 없음: {samples_file}")
    columns = []  # (line, seed, label)
    for raw in samples_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        raw = raw.replace(TRIGGER_PLACEHOLDER, trigger)
        prompt, present = parse_present_flags(raw)
        line, seed = compose_line(prompt, present, raw, seed_base + len(columns), defaults)
        columns.append((line, seed, prompt))
    if not columns:
        raise SystemExit(f"[error] {samples_file} 에 사용할 프롬프트가 없다.")
    return columns


def build_from_tags(name: str, num_prompts: int, num_tags: int,
                    defaults: dict, seed_base: int, rng_seed: int):
    """dataset/<name>/ 캡션 태그에서 샘플링해 프롬프트 컬럼을 만든다.

    캡션 한 개를 고른 뒤 trigger(첫 토큰)는 고정, 나머지 태그에서 num_tags 개를 뽑아 한 컬럼으로.
    """
    data_dir = IMAGE_ROOT / name
    # repeat_<N> 서브셋 구조면 캡션이 하위에 있으므로 재귀 글롭. 단일 폴더면 top-level 만 잡힌다.
    captions = sorted(data_dir.rglob("*.txt"))
    if not captions:
        raise SystemExit(f"[error] {data_dir} 에 캡션(.txt)이 없다. 먼저 1_tag_dataset.py {name} 실행.")

    rng = random.Random(rng_seed)
    picks = captions if len(captions) <= num_prompts else rng.sample(captions, num_prompts)

    columns = []
    for i, cap in enumerate(picks):
        toks = [t.strip() for t in cap.read_text(encoding="utf-8").strip().split(",") if t.strip()]
        if not toks:
            continue
        trigger, rest = toks[0], toks[1:]
        if len(rest) > num_tags:
            rest = rng.sample(rest, num_tags)
        prompt = ", ".join([trigger, *rest])
        line, seed = compose_line(prompt, {}, prompt, seed_base + i, defaults)
        columns.append((line, seed, prompt))
    if not columns:
        raise SystemExit("[error] 태그 샘플링 결과가 비었다.")
    return columns


# ------------------------------------------------------------------------------- 생성
def write_prompts_file(path: Path, columns) -> None:
    header = "# 4_sweep_grid.py 가 생성한 prompts.txt (--from_file 배치 입력)\n"
    body = "\n".join(line for line, _seed, _label in columns)
    path.write_text(header + body + "\n", encoding="utf-8")


def run_inference(cfg: dict, lora: Path, prompts_file: Path, outdir: Path,
                  multiplier: float, device: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(INFERENCE),
        "--dit", cfg["PATH_MODEL"],
        "--text_encoder", cfg["PATH_QWEN"],
        "--vae", cfg["PATH_VAE"],
        "--qwen_image_vae_2d",
        "--lora_weight", str(lora),
        "--lora_multiplier", str(multiplier),
        "--from_file", str(prompts_file),
        "--save_path", str(outdir),
        "--device", device,
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


# ------------------------------------------------------------------------------- 그리드
def load_font(size: int):
    from PIL import ImageFont
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap_label(text: str, width: int = 22, max_lines: int = 4) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " …"
    return "\n".join(lines)


def find_image(outdir: Path, seed: int) -> Path | None:
    hits = sorted(glob.glob(str(outdir / f"*_{seed}_*.png")))
    if not hits:
        hits = sorted(glob.glob(str(outdir / f"*_{seed}.png")))
    return Path(hits[-1]) if hits else None


def build_grid(sweep_dir: Path, name: str, epochs: list[int], columns,
               cell: int = 384, pad: int = 8) -> Path:
    from PIL import Image, ImageDraw

    row_label_w, col_label_h = 100, 70
    ncols, nrows = len(columns), len(epochs)
    grid_w = row_label_w + ncols * (cell + pad) + pad
    grid_h = col_label_h + nrows * (cell + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (18, 18, 18))
    draw = ImageDraw.Draw(grid)
    f_lbl, f_small = load_font(20), load_font(15)

    for c, (_line, _seed, label) in enumerate(columns):
        x = row_label_w + c * (cell + pad) + pad
        draw.multiline_text((x + 6, 6), wrap_label(label), fill=(230, 230, 230), font=f_small, spacing=2)

    for r, ep in enumerate(epochs):
        y = col_label_h + r * (cell + pad) + pad
        draw.text((10, y + cell // 2 - 12), f"ep {ep}", fill=(255, 210, 120), font=f_lbl)
        for c, (_line, seed, _label) in enumerate(columns):
            x = row_label_w + c * (cell + pad) + pad
            path = find_image(sweep_dir / f"ep{ep}", seed)
            if path is None:
                draw.rectangle([x, y, x + cell, y + cell], outline=(80, 80, 80))
                draw.text((x + 10, y + 10), f"missing\nseed {seed}", fill=(200, 80, 80), font=f_small)
                continue
            img = Image.open(path).convert("RGB")
            canvas = Image.new("RGB", (cell, cell), (24, 24, 24))
            w, h = img.size
            scale = min(cell / w, cell / h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            canvas.paste(img, ((cell - img.width) // 2, (cell - img.height) // 2))
            grid.paste(canvas, (x, y))

    out = sweep_dir / f"grid_{name}.png"
    grid.save(out)
    print(f"[done] grid → {out}  ({grid_w}x{grid_h})")
    return out


# -------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="[4] LoRA 에폭 그리드 스윕")
    ap.add_argument("name", help="dataset/output 이름 (예: mychar)")
    ap.add_argument("--mode", choices=["samples", "tags"], default="samples",
                    help="프롬프트 소스: samples=samples.txt, tags=데이터셋 태그 샘플링 (기본 samples)")
    # 프롬프트 소스 옵션
    ap.add_argument("--samples-file", default=str(ROOT / "samples.txt"),
                    help="samples 모드에서 읽을 프롬프트 파일 (기본 ./samples.txt)")
    ap.add_argument("--trigger", default=None,
                    help="samples.txt 의 <trigger_placeholder> 를 치환할 trigger (기본 @<name>)")
    ap.add_argument("--num-prompts", type=int, default=4,
                    help="tags 모드에서 샘플링할 프롬프트(=컬럼) 수 (기본 4)")
    ap.add_argument("--num-tags", type=int, default=10,
                    help="tags 모드에서 캡션당 뽑을 태그 수 (trigger 제외, 기본 10)")
    ap.add_argument("--rng-seed", type=int, default=0, help="태그 샘플링 재현용 RNG 시드")
    # 에폭 선택
    ap.add_argument("--num-epochs", type=int, default=4, help="고를 에폭 수, 최종 포함 (기본 4)")
    ap.add_argument("--epochs", default=None,
                    help="에폭 직접 지정 (쉼표구분, 예: 12,16,20,24). 지정 시 --num-epochs 무시")
    # 생성 기본값
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="컬럼별 시드 시작값 (i번째 컬럼 = seed_base+i). 그리드 매칭 키")
    ap.add_argument("--multiplier", type=float, default=1.0, help="LoRA multiplier")
    ap.add_argument("--device", default="cuda")
    # 출력/실행 제어
    ap.add_argument("--out", default=None, help="스윕 출력 폴더 (기본 output/<name>/sweep4)")
    ap.add_argument("--cell", type=int, default=384, help="그리드 셀 한 변 픽셀 (기본 384)")
    ap.add_argument("--grid-only", action="store_true", help="생성은 건너뛰고 기존 이미지로 그리드만 합성")
    ap.add_argument("--dry-run", action="store_true", help="prompts.txt/계획만 출력, 생성 안 함")
    args = ap.parse_args()

    cfg = load_config()
    dim = cfg["NETWORK_DIM"]
    max_epochs = int(cfg["MAX_TRAIN_EPOCHS"])

    ckpts = discover_checkpoints(args.name, dim, max_epochs)
    explicit = [int(x) for x in args.epochs.split(",") if x.strip()] if args.epochs else None
    epochs = select_epochs(list(ckpts), args.num_epochs, explicit)

    defaults = {
        "steps": args.steps, "guidance": args.guidance,
        "width": args.width, "height": args.height, "negative": args.negative,
    }
    trigger = args.trigger if args.trigger else f"@{args.name}"
    if args.mode == "samples":
        columns = build_from_samples(Path(args.samples_file).resolve(), defaults, args.seed_base, trigger)
    else:
        columns = build_from_tags(args.name, args.num_prompts, args.num_tags,
                                  defaults, args.seed_base, args.rng_seed)

    seeds = [s for _l, s, _lab in columns]
    if len(set(seeds)) != len(seeds):
        print(f"[warn] 컬럼 시드 중복 발견 {seeds} → 그리드 매칭이 어긋날 수 있다.")

    sweep_dir = Path(args.out).resolve() if args.out else (OUTPUT_ROOT / args.name / "sweep4")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = sweep_dir / "prompts.txt"
    write_prompts_file(prompts_file, columns)

    print(f"[plan] name={args.name}  mode={args.mode}  dim={dim}")
    print(f"[plan] epochs({len(epochs)}/{len(ckpts)}, 최종 {max_epochs} 포함): {epochs}")
    print(f"[plan] prompts({len(columns)}): {prompts_file}")
    for line, seed, _ in columns:
        print(f"        seed={seed}: {line[:90]}")
    print(f"[plan] out: {sweep_dir}")

    if args.dry_run:
        print("[dry-run] 생성/그리드 생략.")
        return 0

    if not args.grid_only:
        for ep in epochs:
            print(f"\n=========== EPOCH {ep} ===========")
            run_inference(cfg, ckpts[ep], prompts_file, sweep_dir / f"ep{ep}",
                          args.multiplier, args.device)

    build_grid(sweep_dir, args.name, epochs, columns, cell=args.cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
