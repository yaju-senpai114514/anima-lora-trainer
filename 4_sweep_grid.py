#!/usr/bin/env python3
"""[4] 학습 완료된 LoRA 들을 에폭별로 적용해 그리드 스윕을 돌리는 파이프라인 스크립트.

3_run_training*.sh 가 output/<run>/ 에 찍어둔 에폭 체크포인트들
(anima_<trigger>_r<dim>-NNNNNN.safetensors + 최종 anima_<trigger>_r<dim>.safetensors)을
모아, 같은 프롬프트 묶음을 여러 에폭에 적용해 생성한 뒤 "행=에폭 × 열=프롬프트" 그리드로 묶는다.

흐름:
    1) config_model.env 에서 모델 경로(PATH_MODEL/PATH_QWEN/PATH_VAE)만 읽는다.
    2) 출력 폴더 자체에서 trigger / dim / 에폭을 **자동 추출**한다(심링크·config.env shim 불필요).
       최종 에폭은 폴더명의 `_ep<N>` 에서, 없으면 최대 numbered+1 로 추정.
    3) 프롬프트 묶음을 만든다:
         --mode samples : samples.txt(또는 --samples-file)의 프롬프트.
         --mode tags    : dataset/<trigger>/ 캡션에서 샘플링 (= in-distribution 프롬프트).
    4) 에폭마다 sd-scripts/anima_minimal_inference.py 를 배치 모드로 돌린다.
       **여러 GPU 에 에폭을 분산**해 병렬로 생성한다(--gpus, 기본: 시스템의 모든 GPU).
    5) 시드로 이미지를 매칭해 그리드 PNG(grid_<trigger>.png)를 합성한다.

사용 예:
    # 출력 폴더를 직접 지정. samples.txt 로 최종 포함 4개 에폭, 모든 GPU 병렬.
    uv run python 4_sweep_grid.py mychar_gb4_lr4e-5_ep18

    # 트리거만 줘도 output/ 에서 유일한 폴더면 자동으로 찾는다.
    uv run python 4_sweep_grid.py mychar --mode tags --num-prompts 6 --num-epochs 6

    # 특정 GPU 만 / 에폭 직접 지정 / 생성 건너뛰고 그리드만 재합성
    uv run python 4_sweep_grid.py mychar --gpus 0,1 --epochs 12,16,20,24 --grid-only
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import random
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset"
OUTPUT_ROOT = ROOT / "output"
INFERENCE = ROOT / "sd-scripts" / "anima_minimal_inference.py"
MODEL_ENV = ROOT / "config_model.env"

# 프롬프트 줄 끝 오버라이드 기본값 (anima_minimal_inference.parse_prompt_line 규약).
DEFAULT_NEGATIVE = "lowres, worst quality, low quality, bad anatomy, jpeg artifacts"
OVERRIDE_FLAGS = ("d", "s", "g", "l", "w", "h", "n", "fs")  # 줄에 이미 존재하는지 검사할 플래그
# samples.txt 안에서 trigger(@<name>) 자리로 쓰는 토큰. 스윕 시 --trigger 값으로 치환된다.
TRIGGER_PLACEHOLDER = "<trigger_placeholder>"

# anima_<trigger>_r<dim>[-<epoch>].safetensors  →  trigger / dim / (numbered) epoch
CKPT_RE = re.compile(r"^anima_(?P<trig>.+)_r(?P<dim>\d+)(?:-(?P<ep>\d+))?\.safetensors$")


# ----------------------------------------------------------------------------- config
def load_model_paths() -> dict[str, str]:
    """config_model.env 를 bash 로 source 해서 모델 경로 3종만 뽑는다($HOME/변수 확장 보존).

    예전엔 4_sweep 전용 config.env shim 을 따로 만들어야 했지만, 이제 학습과 같은
    config_model.env 를 직접 읽는다(dim/에폭은 출력 폴더에서 자동 추출하므로 불필요).
    """
    if not MODEL_ENV.is_file():
        raise SystemExit(
            f"[error] {MODEL_ENV} 없음.\n"
            f"        cp config_model.env.example config_model.env 후 경로를 채울 것."
        )
    keys = ["PATH_MODEL", "PATH_QWEN", "PATH_VAE"]
    script = f'source "{MODEL_ENV}" && ' + " && ".join(f'echo "${k}"' for k in keys)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    values = out.splitlines()
    if len(values) < len(keys) or not all(values[: len(keys)]):
        raise SystemExit(f"[error] config_model.env 파싱 실패: {out!r}")
    paths = dict(zip(keys, values))
    for k in keys:
        if not Path(paths[k]).is_file():
            print(f"[warn] {k} 경로에 파일이 없다: {paths[k]}")
    return paths


# ------------------------------------------------------------------- 출력 폴더 / 체크포인트
def resolve_run_dir(arg: str) -> Path:
    """arg → 실제 출력 폴더. 경로거나, output/<arg>, 또는 output/<arg>* 유일 매칭."""
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = OUTPUT_ROOT / arg
    if cand.is_dir():
        return cand.resolve()
    matches = sorted(
        d for d in OUTPUT_ROOT.glob(f"{arg}*")
        if d.is_dir() and any(d.glob("anima_*_r*.safetensors"))
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise SystemExit(
            f"[error] '{arg}' 로 시작하는 출력 폴더가 여러 개다:\n  "
            + "\n  ".join(m.name for m in matches)
            + "\n  정확한 폴더명을 지정할 것."
        )
    raise SystemExit(f"[error] 출력 폴더를 찾을 수 없다: {arg} (output/ 아래 확인)")


def discover(run_dir: Path) -> tuple[str, str, dict[int, Path]]:
    """출력 폴더에서 (trigger, dim, {에폭: 경로}) 를 자동 추출한다.

    최종(번호 없는) 체크포인트의 에폭은 폴더명 `_ep<N>` 에서, 없으면 최대 numbered+1 로 추정.
    """
    ckpts_num: dict[int, Path] = {}
    triggers, dims = set(), set()
    final_path: Path | None = None
    for p in run_dir.glob("anima_*_r*.safetensors"):
        m = CKPT_RE.match(p.name)
        if not m:
            continue
        triggers.add(m["trig"])
        dims.add(m["dim"])
        if m["ep"] is not None:
            ckpts_num[int(m["ep"])] = p
        else:
            final_path = p
    if not triggers:
        raise SystemExit(f"[error] {run_dir} 에 anima_*_r*.safetensors 체크포인트가 없다.")
    if len(triggers) > 1 or len(dims) > 1:
        raise SystemExit(
            f"[error] {run_dir} 에 여러 trigger/dim 이 섞여 있다: trig={sorted(triggers)} dim={sorted(dims)}"
        )
    trigger, dim = triggers.pop(), dims.pop()

    ckpts = dict(ckpts_num)
    if final_path is not None:
        m = re.search(r"_ep(\d+)", run_dir.name)
        if m:
            final_ep = int(m.group(1))
        elif ckpts_num:
            final_ep = max(ckpts_num) + 1
            print(f"[warn] 폴더명에 _ep<N> 이 없어 최종 에폭을 {final_ep}(최대 numbered+1)로 추정")
        else:
            final_ep = 1
        ckpts[final_ep] = final_path
    if not ckpts:
        raise SystemExit(f"[error] {run_dir} 에 사용할 체크포인트가 없다.")
    return trigger, dim, ckpts


def select_epochs(available: list[int], num: int, explicit: list[int] | None) -> list[int]:
    """최종 에폭을 포함해 num 개의 후반 에폭을 고른다."""
    available = sorted(available)
    if explicit:
        chosen = [e for e in explicit if e in available]
        missing = [e for e in explicit if e not in available]
        if missing:
            print(f"[warn] 존재하지 않는 에폭 무시: {missing} (가용: {available})")
        if not chosen:
            raise SystemExit(f"[error] 지정한 에폭 중 가용한 것이 없다. 가용: {available}")
        return sorted(chosen)
    num = max(1, min(num, len(available)))
    return available[-num:]


# ------------------------------------------------------------------------ 프롬프트 빌드
def compose_line(prompt: str, present: dict[str, str], orig: str,
                 seed: int, defaults: dict) -> tuple[str, int]:
    """줄에 없는 오버라이드 플래그만 기본값/시드로 채워 prompts.txt 한 줄을 만든다."""
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
    """samples.txt 의 각 줄을 프롬프트 컬럼으로. <trigger_placeholder> 는 trigger 로 치환."""
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


def build_from_tags(trigger_name: str, num_prompts: int, num_tags: int,
                    defaults: dict, seed_base: int, rng_seed: int):
    """dataset/<trigger>/ 캡션 태그에서 샘플링해 프롬프트 컬럼을 만든다 (in-distribution).

    캡션 한 개를 고른 뒤 trigger(첫 토큰)는 고정, 나머지 태그에서 num_tags 개를 뽑아 한 컬럼으로.
    """
    data_dir = IMAGE_ROOT / trigger_name
    captions = sorted(data_dir.rglob("*.txt"))
    if not captions:
        raise SystemExit(
            f"[error] {data_dir} 에 캡션(.txt)이 없다. tags 모드는 dataset/{trigger_name}/ 가 필요하다."
        )
    rng = random.Random(rng_seed)
    picks = captions if len(captions) <= num_prompts else rng.sample(captions, num_prompts)

    columns = []
    for i, cap in enumerate(picks):
        toks = [t.strip() for t in cap.read_text(encoding="utf-8").strip().split(",") if t.strip()]
        if not toks:
            continue
        trig, rest = toks[0], toks[1:]
        if len(rest) > num_tags:
            rest = rng.sample(rest, num_tags)
        prompt = ", ".join([trig, *rest])
        line, seed = compose_line(prompt, {}, prompt, seed_base + i, defaults)
        columns.append((line, seed, prompt))
    if not columns:
        raise SystemExit("[error] 태그 샘플링 결과가 비었다.")
    return columns


def write_prompts_file(path: Path, columns) -> None:
    header = "# 4_sweep_grid.py 가 생성한 prompts.txt (--from_file 배치 입력)\n"
    body = "\n".join(line for line, _seed, _label in columns)
    path.write_text(header + body + "\n", encoding="utf-8")


# ------------------------------------------------------------------------- 생성 (병렬)
def detect_gpus() -> list[int]:
    """nvidia-smi 로 GPU 인덱스 목록을 얻는다. 실패하면 빈 목록."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return [i for i, ln in enumerate(out.splitlines()) if ln.strip().startswith("GPU")]
    except Exception:
        return []


def run_inference_one(paths: dict, lora: Path, prompts_file: Path, outdir: Path,
                      multiplier: float, gpu: str, log_path: Path) -> None:
    """한 에폭 = 한 서브프로세스. gpu 로 CUDA_VISIBLE_DEVICES 를 고정해 GPU 를 핀한다."""
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cpu" if str(gpu) == "cpu" else "cuda"
    cmd = [
        sys.executable, str(INFERENCE),
        "--dit", paths["PATH_MODEL"],
        "--text_encoder", paths["PATH_QWEN"],
        "--vae", paths["PATH_VAE"],
        "--qwen_image_vae_2d",
        "--lora_weight", str(lora),
        "--lora_multiplier", str(multiplier),
        "--from_file", str(prompts_file),
        "--save_path", str(outdir),
        "--device", device,
    ]
    env = dict(os.environ)
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # 이 서브프로세스는 이 GPU 하나만 본다 → cuda:0
    with open(log_path, "wb") as lf:
        subprocess.run(cmd, cwd=str(ROOT), check=True, env=env, stdout=lf, stderr=subprocess.STDOUT)


def sweep_parallel(paths: dict, ckpts: dict[int, Path], epochs: list[int],
                   sweep_dir: Path, multiplier: float, gpus: list) -> dict[int, bool]:
    """에폭들을 GPU 풀에 분산해 병렬 생성한다. 반환: {에폭: 성공여부}.

    한 에폭이 실패해도 나머지는 계속 돈다(그리드에서 그 칸만 missing 으로 나온다).
    """
    gpu_q: queue.Queue = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    prompts_file = sweep_dir / "prompts.txt"
    lock = threading.Lock()

    def work(ep: int) -> tuple[int, bool]:
        gpu = gpu_q.get()
        log_path = sweep_dir / f"ep{ep}.log"
        try:
            with lock:
                print(f"  [ep{ep}] ▶ gpu {gpu} 에서 시작", flush=True)
            run_inference_one(paths, ckpts[ep], prompts_file, sweep_dir / f"ep{ep}",
                              multiplier, gpu, log_path)
            with lock:
                print(f"  [ep{ep}] ✓ 완료 (gpu {gpu})", flush=True)
            return ep, True
        except subprocess.CalledProcessError:
            tail = []
            if log_path.exists():
                tail = log_path.read_text(errors="replace").splitlines()[-10:]
            with lock:
                print(f"  [ep{ep}] ✗ 실패 (gpu {gpu}) — 로그 마지막 10줄 ({log_path}):", flush=True)
                for ln in tail:
                    print(f"      {ln}", flush=True)
            return ep, False
        finally:
            gpu_q.put(gpu)

    results: dict[int, bool] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        for ep, ok in ex.map(work, epochs):
            results[ep] = ok
    return results


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


def build_grid(sweep_dir: Path, trigger: str, epochs: list[int], columns,
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

    out = sweep_dir / f"grid_{trigger}.png"
    grid.save(out)
    print(f"[done] grid → {out}  ({grid_w}x{grid_h})")
    return out


# -------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="[4] LoRA 에폭 그리드 스윕 (다중 GPU 병렬)")
    ap.add_argument("run", help="출력 폴더 이름/경로 또는 트리거 (예: mychar_gb4_lr4e-5_ep18, mychar)")
    ap.add_argument("--mode", choices=["samples", "tags"], default="samples",
                    help="프롬프트 소스: samples=samples.txt, tags=데이터셋 태그(in-distribution) (기본 samples)")
    # 프롬프트 소스 옵션
    ap.add_argument("--samples-file", default=str(ROOT / "samples.txt"),
                    help="samples 모드에서 읽을 프롬프트 파일 (기본 ./samples.txt)")
    ap.add_argument("--trigger", default=None,
                    help="프롬프트 trigger (기본 @<자동추출 트리거>)")
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
    # 병렬 실행
    ap.add_argument("--gpus", default=None,
                    help="사용할 GPU 인덱스 (쉼표, 예: 0,1,2). 기본: 시스템의 모든 GPU. 'cpu' 도 가능")
    # 출력/실행 제어
    ap.add_argument("--out", default=None, help="스윕 출력 폴더 (기본 <run>/sweep_<mode>)")
    ap.add_argument("--cell", type=int, default=384, help="그리드 셀 한 변 픽셀 (기본 384)")
    ap.add_argument("--grid-only", action="store_true", help="생성은 건너뛰고 기존 이미지로 그리드만 합성")
    ap.add_argument("--dry-run", action="store_true", help="prompts.txt/계획만 출력, 생성 안 함")
    args = ap.parse_args()

    paths = load_model_paths()
    run_dir = resolve_run_dir(args.run)
    trigger, dim, ckpts = discover(run_dir)

    explicit = [int(x) for x in args.epochs.split(",") if x.strip()] if args.epochs else None
    epochs = select_epochs(list(ckpts), args.num_epochs, explicit)

    defaults = {
        "steps": args.steps, "guidance": args.guidance,
        "width": args.width, "height": args.height, "negative": args.negative,
    }
    trig_token = args.trigger if args.trigger else f"@{trigger}"
    if args.mode == "samples":
        columns = build_from_samples(Path(args.samples_file).resolve(), defaults, args.seed_base, trig_token)
    else:
        columns = build_from_tags(trigger, args.num_prompts, args.num_tags,
                                  defaults, args.seed_base, args.rng_seed)

    seeds = [s for _l, s, _lab in columns]
    if len(set(seeds)) != len(seeds):
        print(f"[warn] 컬럼 시드 중복 발견 {seeds} → 그리드 매칭이 어긋날 수 있다.")

    sweep_dir = Path(args.out).resolve() if args.out else (run_dir / f"sweep_{args.mode}")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    write_prompts_file(sweep_dir / "prompts.txt", columns)

    # GPU 목록 결정
    if args.gpus is not None:
        gpus: list = ["cpu"] if args.gpus.strip().lower() == "cpu" else \
            [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpus = detect_gpus()
    if not gpus:
        raise SystemExit("[error] 사용할 GPU 를 찾지 못했다. --gpus 로 지정하거나 --gpus cpu 를 쓸 것.")

    print(f"[plan] run={run_dir.name}  trigger={trigger}  dim=r{dim}  mode={args.mode}")
    print(f"[plan] epochs({len(epochs)}/{len(ckpts)}): {epochs}  (가용 {sorted(ckpts)})")
    print(f"[plan] prompts({len(columns)}) → {sweep_dir / 'prompts.txt'}")
    for line, seed, _ in columns:
        print(f"        seed={seed}: {line[:90]}")
    print(f"[plan] gpus={gpus}  out={sweep_dir}")

    if args.dry_run:
        print("[dry-run] 생성/그리드 생략.")
        return 0

    if not args.grid_only:
        results = sweep_parallel(paths, ckpts, epochs, sweep_dir, args.multiplier, gpus)
        failed = [ep for ep, ok in results.items() if not ok]
        if failed:
            print(f"[warn] 실패한 에폭: {sorted(failed)} (그리드에서 해당 행은 missing 으로 표시된다)")

    build_grid(sweep_dir, trigger, epochs, columns, cell=args.cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
