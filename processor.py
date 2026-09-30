#!/usr/bin/env python3
"""Batch gemstone catalogue editor using the OpenAI API. See README.md.

Per batch (one sub-directory of INPUT_DIR holding exactly three photos):
  1. one vision call sees all three photos -> Front/Back/Side mapping + background
  2. three independent image edits, each fed ONLY its own source photo
Python owns validation, naming, sizing, retries, resumability and logging.
"""
import argparse
import base64
import glob
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
RULES_FILE = ROOT / "rules" / "gemstone_editor.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
ROLES = {"front": "Front", "back": "Back", "side": "Side_Angle"}  # role -> filename part
ROLE_LABEL = {"front": "FRONT", "back": "BACK", "side": "SIDE / ANGLE"}
BACKGROUNDS = {"black": ("Black", "deep black #080808"), "white": ("White", "neutral white #FFFFFF")}
STONE_TYPES = ["white_or_warm_white", "white_or_warm_white_opal", "other_opal", "other"]
OUT_PX = 1080
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
CLASSIFY_PREVIEW_PX = 1536  # classification sees downscaled copies; edits get full resolution
BACKOFF_BASE = 2.0  # seconds; wait after attempt n is BACKOFF_BASE * 2**(n-1)

log = logging.getLogger("gemstone")


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    failed_dir: Path
    log_dir: Path
    image_model: str = "gpt-image-2.5-sunburst"
    vision_model: str = "gpt-6-astra"
    image_size: str = "1088x1088"
    image_quality: str = "high"
    max_retries: int = 3
    timeout: float = 300.0


def load_config():
    load_dotenv(ROOT / ".env")
    d = lambda key, default: (ROOT / os.getenv(key, default)).resolve()
    return Config(
        input_dir=d("INPUT_DIR", "input"),
        output_dir=d("OUTPUT_DIR", "output"),
        failed_dir=d("FAILED_DIR", "failed"),
        log_dir=d("LOG_DIR", "logs"),
        image_model=os.getenv("OPENAI_IMAGE_MODEL") or Config.image_model,
        vision_model=os.getenv("OPENAI_VISION_MODEL") or Config.vision_model,
        image_size=os.getenv("OPENAI_IMAGE_SIZE") or Config.image_size,
        image_quality=os.getenv("OPENAI_IMAGE_QUALITY") or Config.image_quality,
        max_retries=max(1, int(os.getenv("MAX_RETRIES", "3"))),
        timeout=float(os.getenv("OPENAI_TIMEOUT", "300")),
    )


class BadOutput(Exception):
    """The model answered, but the answer is unusable (malformed JSON, bad image). Retryable."""


class StageError(Exception):
    """A stage failed permanently."""

    def __init__(self, stage, error_type, message, attempts):
        super().__init__(message)
        self.stage, self.error_type, self.message, self.attempts = stage, error_type, message, attempts


# ---------------------------------------------------------------- helpers

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize(name):
    """Filesystem-safe batch name used for output files and folders."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "batch"
    return "_" + name if name.split(".")[0].upper() in WINDOWS_RESERVED else name


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def output_filename(out_name, role, background):
    return f"{out_name}_{ROLES[role]}_{BACKGROUNDS[background][0]}_BG.png"


def is_valid_output(path):
    """Exists, non-empty, decodes fully, PNG, exactly 1080x1080."""
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            if im.format != "PNG" or im.size != (OUT_PX, OUT_PX):
                return False
            im.load()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- batch validation

def find_sources(batch_dir):
    return sorted(p for p in batch_dir.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS)


def validate_batch(batch_dir):
    """Return (sources, reason); reason is None when the batch is valid."""
    sources = find_sources(batch_dir)
    if len(sources) != 3:
        listed = f": {', '.join(s.name for s in sources)}" if sources else ""
        return sources, f"expected exactly 3 images, found {len(sources)}{listed}"
    for s in sources:
        try:
            with Image.open(s) as im:
                im.verify()
        except Exception as e:  # Pillow raises assorted types for corrupt files
            return sources, f"unreadable image {s.name}: {e}"
    return sources, None


def batch_identity(sources):
    h = hashlib.sha256()
    for s in sources:
        h.update(s.name.encode() + b"\0" + s.read_bytes())
    return h.hexdigest()


def discover(cfg):
    return sorted(p for p in cfg.input_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def batch_state(cfg, batch_dir):
    """(state, detail) without any API call. state: invalid|failed|complete|incomplete|pending."""
    out_name = sanitize(batch_dir.name)
    sources, reason = validate_batch(batch_dir)
    if reason:
        return "invalid", reason
    err = read_json(cfg.failed_dir / out_name / "error.json")
    if err is not None:
        return "failed", f"{err.get('stage')}: {err.get('message')}"
    out_dir = cfg.output_dir / out_name
    m = read_json(out_dir / "manifest.json")
    if not m or m.get("identity") != batch_identity(sources):
        return "pending", "classify + 3 edits"
    outs = m.get("outputs") or {}
    missing = [r for r in ROLES if not (outs.get(r) and is_valid_output(out_dir / outs[r]))]
    if not missing:
        return "complete", "all 3 outputs valid"
    return "incomplete", f"reuse classification, edit: {', '.join(missing)}"


# ---------------------------------------------------------------- OpenAI calls

def upload_bytes(path):
    """(filename, bytes, mime) for upload. Applies EXIF rotation in memory; the source is never touched."""
    with Image.open(path) as im:
        orientation = im.getexif().get(0x0112, 1)
        if orientation == 1:
            return path.name, path.read_bytes(), MIME[path.suffix.lower()]
        buf = io.BytesIO()
        ImageOps.exif_transpose(im).save(buf, "PNG")
    return path.stem + ".png", buf.getvalue(), "image/png"


def preview_data_url(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((CLASSIFY_PREVIEW_PX, CLASSIFY_PREVIEW_PX))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CLASSIFY_PROMPT = """You are preparing ONE gemstone batch for editing under the gemstone editing rules below.
The three photographs of the same gemstone follow, each preceded by its file name: {names}.

Decide:
- which photograph is the genuine Front, Back and Side / Angle view (use each file exactly once);
- the gemstone's predominant natural body colour category (for opals judge body colour, not fire);
- the background the rules require (black for white/warm-white stones, white otherwise).

Judge from visible geometry, cut, curvature, facets, silhouette, orientation and surface characteristics.
Do NOT infer roles from file names. Confidence values are your honest 0-1 estimates.

--- GEMSTONE EDITING RULES ---
{rules}"""


def classification_schema(names):
    conf_keys = ["front", "back", "side", "background"]
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "front_source": {"type": "string", "enum": names},
            "back_source": {"type": "string", "enum": names},
            "side_source": {"type": "string", "enum": names},
            "background": {"type": "string", "enum": list(BACKGROUNDS)},
            "stone_type": {"type": "string", "enum": STONE_TYPES},
            "confidence": {
                "type": "object",
                "properties": {k: {"type": "number"} for k in conf_keys},
                "required": conf_keys,
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "front_source", "back_source", "side_source",
                     "background", "stone_type", "confidence"],
        "additionalProperties": False,
    }


def validate_classification(data, names):
    """Raise BadOutput unless the mapping uses each source exactly once and the background is consistent."""
    if not isinstance(data, dict):
        raise BadOutput("classification is not a JSON object")
    mapping = {r: data.get(f"{r}_source") for r in ROLES}
    if sorted(map(str, mapping.values())) != sorted(names):
        raise BadOutput(f"each source must map to exactly one role, got {mapping}")
    background, stone = data.get("background"), data.get("stone_type")
    if background not in BACKGROUNDS:
        raise BadOutput(f"background must be black or white, got {background!r}")
    if stone not in STONE_TYPES:
        raise BadOutput(f"unknown stone_type {stone!r}")
    if (background == "black") != stone.startswith("white"):
        raise BadOutput(f"background {background!r} contradicts stone_type {stone!r}")
    return {"mapping": mapping, "background": background, "stone_type": stone,
            "confidence": data.get("confidence") or {}, "reasoning": data.get("reasoning", "")}


def classify(client, cfg, sources, rules):
    names = [s.name for s in sources]
    content = [{"type": "input_text", "text": CLASSIFY_PROMPT.format(names=", ".join(names), rules=rules)}]
    for s in sources:
        content.append({"type": "input_text", "text": f"Photograph: {s.name}"})
        content.append({"type": "input_image", "image_url": preview_data_url(s)})
    resp = client.responses.create(
        model=cfg.vision_model,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "gemstone_classification",
                         "schema": classification_schema(names), "strict": True}},
    )
    try:
        data = json.loads(resp.output_text)
    except (TypeError, ValueError) as e:
        raise BadOutput(f"classification is not valid JSON: {e}") from e
    return validate_classification(data, names)


EDIT_PROMPT = """{rules}

--- CURRENT EDIT ---
Edit this exact source photograph according to the gemstone editing rules.

This is the {label} output.

Preserve the exact photographed gemstone and viewpoint.

Do not reconstruct or replace the gemstone.

Produce one 1080x1080 catalogue image only.

Batch decisions already made from all three photographs (apply them, do not revisit):
- Background: {background}
- Stone category: {stone}
"""


def edit_image(client, cfg, src, role, cls, rules):
    """One edit of one source photo. Returns a 1080x1080 RGB PIL image."""
    resp = client.images.edit(
        model=cfg.image_model,
        image=upload_bytes(src),
        prompt=EDIT_PROMPT.format(rules=rules, label=ROLE_LABEL[role],
                                  background=BACKGROUNDS[cls["background"]][1],
                                  stone=cls["stone_type"].replace("_", " ")),
        size=cfg.image_size,
        quality=cfg.image_quality,
        output_format="png",
        background="opaque",
        n=1,
    )
    try:
        raw = base64.b64decode(resp.data[0].b64_json, validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.width != im.height:
                raise BadOutput(f"model returned non-square image {im.size}")
            # uniform scale only (API sizes are multiples of 16, so 1080 can't be requested directly)
            return im.convert("RGB").resize((OUT_PX, OUT_PX), Image.Resampling.LANCZOS)
    except (AttributeError, IndexError, TypeError, ValueError, OSError) as e:
        raise BadOutput(f"unusable image in response: {e}") from e


def save_png(img, path):
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, "PNG")
    os.replace(tmp, path)  # atomic: an interrupted run never leaves a half-written output
    if not is_valid_output(path):
        path.unlink(missing_ok=True)
        raise BadOutput(f"{path.name} failed validation after save")


def with_retries(fn, stage, batch, attempts):
    import openai
    kinds = {openai.RateLimitError: "rate_limit", openai.APITimeoutError: "timeout",
             openai.APIConnectionError: "network_error", openai.InternalServerError: "server_error",
             BadOutput: "malformed_output",
             OSError: "file_error"}  # e.g. Windows refusing to replace a PNG that is open in a viewer
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except tuple(kinds) as e:
            err = e
            log.warning("%s %s RETRY %d/%d %s: %s", batch, stage.upper(), attempt, attempts, type(e).__name__, e)
            if attempt < attempts:
                time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
        except openai.APIError as e:  # auth, bad request, content policy: retrying won't help
            raise StageError(stage, "api_error", f"{type(e).__name__}: {e}", attempt) from e
    kind = next(v for k, v in kinds.items() if isinstance(err, k))
    raise StageError(stage, kind, f"{type(err).__name__}: {err}", attempts) from err


# ---------------------------------------------------------------- batch processing

def record_failure(cfg, name, out_name, err, manifest):
    d = cfg.failed_dir / out_name
    write_json(d / "manifest.json", manifest)
    write_json(d / "error.json", {"batch": name, "stage": err.stage, "error_type": err.error_type,
                                  "message": err.message, "attempts": err.attempts, "timestamp": now()})


def saved_classification(manifest, names):
    m = manifest.get("mapping")
    if not isinstance(m, dict):
        return None
    try:
        return validate_classification(
            {**{f"{r}_source": m.get(r) for r in ROLES},
             "background": manifest.get("background"), "stone_type": manifest.get("stone_type"),
             "confidence": manifest.get("confidence"), "reasoning": manifest.get("reasoning")}, names)
    except BadOutput:
        return None


def process_batch(client, cfg, batch_dir, rules, force=False):
    """Returns (status, images_generated, images_failed); status is complete|failed|invalid."""
    name, out_name = batch_dir.name, sanitize(batch_dir.name)
    out_dir = cfg.output_dir / out_name
    log.info("%s STARTED", name)

    sources, reason = validate_batch(batch_dir)
    if reason:
        log.error("%s INVALID %s", name, reason)
        return "invalid", 0, 0
    names = [s.name for s in sources]
    identity = batch_identity(sources)

    manifest = read_json(out_dir / "manifest.json") or {}
    # forced, first run, or sources changed: nothing on disk can be trusted to match these photos
    fresh = force or manifest.get("identity") != identity
    if fresh:
        manifest = {}
    manifest.update(batch=name, output_name=out_name, identity=identity, sources=names, updated=now())

    cls = saved_classification(manifest, names)
    if cls:
        log.info("%s CLASSIFICATION_REUSED", name)
    else:
        try:
            cls = with_retries(lambda: classify(client, cfg, sources, rules), "classification", name, cfg.max_retries)
        except StageError as e:
            log.error("%s FAILED stage=%s attempts=%d %s", name, e.stage, e.attempts, e.message)
            manifest["status"] = "failed"
            record_failure(cfg, name, out_name, e, manifest)
            return "failed", 0, 0
        manifest.update(cls)
        log.info("%s CLASSIFICATION_COMPLETE front=%s back=%s side=%s background=%s stone=%s", name,
                 *cls["mapping"].values(), cls["background"], cls["stone_type"])

    expected = {r: output_filename(out_name, r, cls["background"]) for r in ROLES}
    manifest.update(outputs=expected, status="in_progress")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    # drop leftovers: temp files from interrupted runs, outputs from an older background decision
    for p in out_dir.glob(f"{glob.escape(out_name)}_*_BG.png*"):
        if p.name not in expected.values():
            log.info("%s REMOVED_STALE %s", name, p.name)
            p.unlink()

    by_name = {s.name: s for s in sources}
    generated, errors = 0, []
    for role, fname in expected.items():
        path = out_dir / fname
        if not fresh and is_valid_output(path):
            log.info("%s %s_EXISTS %s", name, role.upper(), fname)
            continue
        src = by_name[cls["mapping"][role]]
        try:
            with_retries(lambda: save_png(edit_image(client, cfg, src, role, cls, rules), path),
                         f"{role}_edit", name, cfg.max_retries)
        except StageError as e:
            log.error("%s %s_FAILED attempts=%d %s", name, role.upper(), e.attempts, e.message)
            errors.append(e)
            continue
        generated += 1
        log.info("%s %s_COMPLETE %s <- %s", name, role.upper(), fname, src.name)

    manifest.update(status="failed" if errors else "complete", updated=now())
    write_json(out_dir / "manifest.json", manifest)
    if errors:
        record_failure(cfg, name, out_name, errors[0], manifest)
        log.error("%s FAILED %d/3 outputs failed", name, len(errors))
        return "failed", generated, len(errors)
    shutil.rmtree(cfg.failed_dir / out_name, ignore_errors=True)
    log.info("%s COMPLETE", name)
    return "complete", generated, 0


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch-edit gemstone photos into catalogue images via the OpenAI API.")
    p.add_argument("--batch", action="append", metavar="NAME", help="process only this batch (repeatable)")
    p.add_argument("--retry-failed", action="store_true", help="retry batches recorded under failed/")
    p.add_argument("--dry-run", action="store_true", help="show what would be processed; no API calls")
    p.add_argument("--status", action="store_true", help="show batch counts; no API calls")
    p.add_argument("--force", action="append", metavar="NAME",
                   help="reclassify and regenerate this batch (inputs are never touched)")
    return p.parse_args(argv)


def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_dir / "processing.log", encoding="utf-8")
    console = logging.StreamHandler()
    for h in (file_handler, console):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def print_summary(s, cfg):
    rel = lambda p: p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
    print("\n".join([
        "=" * 40, "GEMSTONE PROCESSING COMPLETE" if not s.get("interrupted") else "GEMSTONE PROCESSING INTERRUPTED",
        "=" * 40, "",
        f"{'Batches discovered:':<22}{s['discovered']:>6}",
        f"{'Completed:':<22}{s['complete']:>6}",
        f"{'Skipped:':<22}{s['skipped']:>6}",
        f"{'Failed:':<22}{s['failed']:>6}",
        f"{'Invalid:':<22}{s['invalid']:>6}", "",
        f"{'Images generated:':<22}{s['generated']:>6}",
        f"{'Images failed:':<22}{s['images_failed']:>6}", "",
        "See:", f"  {rel(cfg.log_dir / 'processing.log')}", f"  {rel(cfg.failed_dir)}/", "=" * 40,
    ]))


def run(args, cfg, make_client):
    rules = RULES_FILE.read_text(encoding="utf-8")  # Windows defaults to cp1252
    if not cfg.input_dir.is_dir():
        print(f"Input directory not found: {cfg.input_dir}", file=sys.stderr)
        return 2
    batches = discover(cfg)
    forced = {n for n in args.force or []}

    # two input folders that sanitize to the same output name would overwrite each other
    seen = {}
    for b in batches:
        seen.setdefault(sanitize(b.name).lower(), []).append(b.name)  # Windows/macOS paths ignore case
    clashes = {n for names in seen.values() if len(names) > 1 for n in names}

    wanted = set(args.batch or []) | forced
    if wanted:
        unknown = wanted - {b.name for b in batches} - {sanitize(b.name) for b in batches}
        if unknown:
            print(f"Unknown batch(es): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        batches = [b for b in batches if b.name in wanted or sanitize(b.name) in wanted]
        forced = {b.name for b in batches if b.name in forced or sanitize(b.name) in forced}

    stats = dict(discovered=len(batches), complete=0, skipped=0, failed=0, invalid=0, generated=0, images_failed=0)
    plan = []
    for b in batches:
        state, detail = ("invalid", f"output name {sanitize(b.name)!r} clashes with another batch") \
            if b.name in clashes else batch_state(cfg, b)
        if b.name in forced and state != "invalid":
            action = "force"
        elif state == "invalid":
            action = "invalid"
        elif args.retry_failed:
            action = "process" if state == "failed" else "skip"
        elif state == "complete" or (state == "failed" and not wanted):
            action = "skip"
        else:
            action = "process"
        plan.append((b, state, detail, action))

    if args.status or args.dry_run:
        counts = {}
        for b, state, detail, action in plan:
            counts[state] = counts.get(state, 0) + 1
            if args.dry_run:
                print(f"{b.name:<30} {state:<10} {'REFUSE' if action == 'invalid' else 'WOULD ' + action.upper():<14} {detail}")
        if args.status:
            print(f"Total batches: {len(plan)}")
            for k in ("complete", "pending", "incomplete", "failed", "invalid"):
                print(f"  {k:<11} {counts.get(k, 0)}")
        return 0

    client = make_client() if any(a in ("process", "force") for *_, a in plan) else None
    try:
        for b, state, detail, action in plan:
            if action == "skip":
                stats["skipped"] += 1
                log.info("%s SKIPPED %s%s", b.name, state, " (use --retry-failed)" if state == "failed" else "")
                continue
            if action == "invalid":
                stats["invalid"] += 1
                log.error("%s INVALID %s", b.name, detail)
                continue
            status, gen, bad = process_batch(client, cfg, b, rules, force=action == "force")
            stats[status] += 1
            stats["generated"] += gen
            stats["images_failed"] += bad
    except KeyboardInterrupt:
        log.warning("INTERRUPTED by user; completed outputs are kept, rerun to resume")
        stats["interrupted"] = True
        print_summary(stats, cfg)
        return 130
    print_summary(stats, cfg)
    return 1 if stats["failed"] or stats["invalid"] else 0


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config()
    setup_logging(cfg.log_dir)

    def make_client():
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key == "your_api_key_here":
            raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        from openai import OpenAI
        return OpenAI(api_key=key, max_retries=0, timeout=cfg.timeout)  # retries are ours, so counts are honest

    return run(args, cfg, make_client)


if __name__ == "__main__":
    sys.exit(main())
