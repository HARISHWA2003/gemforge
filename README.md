# Gemstone Editor

A local Python app that turns batches of three gemstone photos into three 1080×1080 catalogue images. It uses the OpenAI API directly. It does not drive the ChatGPT website.

The editing behaviour is written down in [`rules/gemstone_editor.md`](rules/gemstone_editor.md), based on the existing Custom GPT's instructions. **It has not been tested for identical results to the Custom GPT.** Review the output of a few real batches before running a large job.

## How it works

For each batch folder:

1. **Validate (Python):** the folder must hold exactly 3 images (`.jpg`, `.jpeg`, `.png`, `.webp`). Hidden and non-image files are ignored. Batches with the wrong count are refused, and no API call is made.
2. **Classify (1 vision call, `OPENAI_VISION_MODEL`):** all three photos are sent together. The model returns a structured JSON answer: which file is the Front, Back and Side / Angle view, the stone category, and a black or white background. Python rejects any answer that doesn't use each file exactly once or that is self-contradictory, and retries.
3. **Edit (3 image calls, `OPENAI_IMAGE_MODEL`):** one edit per view. Each edit receives **only its own source photo**, plus the rules and the background chosen for the batch.
4. **Finish (Python):** each result is scaled to exactly 1080×1080 (the API only accepts sizes in multiples of 16, so it generates 1088×1088 and Python shrinks it evenly). Python checks it is a valid PNG, saves it atomically and writes `manifest.json`.

Input photos are only ever read. Nothing is written to, moved in or deleted from `input/`.

Models checked against the OpenAI docs on 2026-09-30: `gpt-image-2.5-sunburst` for image edits and `gpt-6-astra` for classification. Both can be changed in `.env`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (cmd / PowerShell)
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Put your key in `OPENAI_API_KEY`. `.env` is git-ignored, and the key is never logged or printed.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2.5-sunburst` | image editing model |
| `OPENAI_VISION_MODEL` | `gpt-6-astra` | Front/Back/Side + background classification |
| `OPENAI_IMAGE_SIZE` | `1088x1088` | size requested from the API (multiples of 16), scaled to 1080 afterwards. `2048x2048` gives a sharper downscale but costs more |
| `OPENAI_IMAGE_QUALITY` | `high` | `low` / `medium` / `high` / `xhigh` / `max` / `auto` |
| `OPENAI_TIMEOUT` | `300` | seconds per API request |
| `MAX_RETRIES` | `3` | attempts per classification and per edit |
| `INPUT_DIR` `OUTPUT_DIR` `FAILED_DIR` `LOG_DIR` | `./input` … | folders, relative to the project root |

## Directory structure

```text
input/
  Ruby_001/          <- one folder per gemstone = one batch
    a.jpg            <- exactly three photos; file names don't matter,
    b.jpg               the model decides which is Front / Back / Side
    c.jpg
output/
  Ruby_001/
    Ruby_001_Front_White_BG.png
    Ruby_001_Back_White_BG.png
    Ruby_001_Side_Angle_White_BG.png
    manifest.json    <- source -> role mapping, background, status
failed/
  Ruby_003/
    error.json       <- stage, error type, message, attempts, timestamp
    manifest.json
logs/processing.log
```

White and warm-white stones get `_Black_BG` (#080808). All other stones get `_White_BG` (#FFFFFF). In folder names, spaces and unsupported characters become `_` (for example, `Blue Sapphire #7` becomes `Blue_Sapphire_7`). The original name is kept in `manifest.json`. If two input folders would end up with the same cleaned name, both are refused so neither overwrites the other.

## First run

```bash
python processor.py --status       # counts: complete / pending / incomplete / failed / invalid
python processor.py --dry-run      # per-batch plan, no API calls
python processor.py                # process everything pending
```

Each batch costs 1 classification call and 3 image edits. Try a single batch first and look at the images before running a large job.

## Processing one batch

```bash
python processor.py --batch Ruby_001
python processor.py --force Ruby_001   # reclassify and regenerate all 3 (inputs untouched)
```

## Retry failures

```bash
python processor.py --retry-failed
```

A normal run **skips** batches that already have a record in `failed/`, so the same failure isn't paid for on every run. `--retry-failed` processes only those batches. It reuses any outputs and classification that already succeeded, and it removes the `failed/<batch>/` record once the batch completes.

## Resume behaviour

- Outputs that already exist and are valid are never regenerated. If one file is deleted, only that file is recreated.
- A saved classification is reused, so there is no second classification charge.
- If any source photo changes (content or name), the batch is treated as new: it is reclassified and all three images are regenerated.
- Ctrl-C is safe. Files are written atomically, so an interrupted run never leaves a half-written PNG. Run the command again to continue.

## What is and isn't verified

Python checks that each output exists, is not empty, decodes fully, is a PNG, is exactly 1080×1080, and has the role and background in its name. **Python does not check visual fidelity:** whether the gem is unchanged, the viewpoint is correct or the background colour is exact. That depends on the model and on your review.

## Tests

```bash
python -m unittest discover tests                  # mocked API, no network, no cost
GEMSTONE_INTEGRATION=1 GEMSTONE_INTEGRATION_BATCH=input/Ruby_001 \
  python -m unittest tests.test_integration        # one real batch, costs money
```

Windows PowerShell equivalent for the integration test:

```powershell
$env:GEMSTONE_INTEGRATION="1"; $env:GEMSTONE_INTEGRATION_BATCH="input/Ruby_001"
python -m unittest tests.test_integration
```

Integration-test outputs go to `output/_integration/` for you to review.

## Windows notes

- The commands are the same (`python processor.py ...`). Activate the environment with `.venv\Scripts\activate`. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- Close output PNGs in image viewers before a `--force` run. If Windows won't let a locked file be replaced, that output is retried and then recorded under `failed/` rather than crashing the run.
- Batch folders named like Windows device names (`CON`, `NUL`, `COM1` …) are saved as `_CON` and so on.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example` and set the key. |
| `INVALID expected exactly 3 images, found N` | Fix the folder so it holds exactly three photos. The app will not pick three from a larger set. |
| `INVALID unreadable image` | A photo is corrupt or not really an image. Re-export it. |
| `RETRY n/3 RateLimitError` | Rate limit reached. The app backs off (2s, 4s, …). If batches still fail, lower your throughput or raise the account limits, then run `--retry-failed`. |
| `error_type: network_error / timeout / server_error` | A temporary problem. Run `--retry-failed`. Raise `OPENAI_TIMEOUT` if high-quality edits time out. |
| `error_type: malformed_output` | The model returned invalid JSON, an inconsistent mapping or an unusable image 3 times in a row. See `failed/<batch>/error.json` and the log. Usually `--retry-failed` fixes it. |
| `error_type: api_error` (attempts: 1) | A request that won't succeed on retry: invalid key, unknown model name, content policy, or an image that is too large. Read the message, fix the cause, then run `--retry-failed`. |
| Wrong Front/Back/Side mapping | Check `mapping` and `reasoning` in `manifest.json`. Run `--force <batch>` to reclassify. |
