"""FLEURS regression eval set extractor.

PLAN §7 wants 100 FLEURS test-split utts/lang as the regression check
(ensures Praxy-STT LoRA doesn't break general read-prose performance).

FLEURS is on HF (`google/fleurs`) but ships as data-script — we download
the per-language test.tsv + audio tarball directly (verified working in
the ml-preflight check 2026-04-29).

Output: ``data/stt_flywheel/holdouts/{lang}/fleurs_regression.jsonl``
with fields:
  {id, lang, entity_class, text, audio_path, entity_tokens, source}

`source = "fleurs"` and `entity_class = "general"` so the EHR scorer
treats these as the regression baseline (no entity targets).

CLI::

    # Plan only
    uv run python -m paper.stt_flywheel.fleurs_regression_extractor plan \\
        --lang te

    # Build (downloads tar.gz on first run, cached)
    uv run python -m paper.stt_flywheel.fleurs_regression_extractor build \\
        --lang te --n 100 --execute
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_DIR = ROOT / "data" / "stt_flywheel" / "holdouts"
FLEURS_CACHE = ROOT / "data" / "fleurs_cache"
FLEURS_REPO = "google/fleurs"

LANG_TO_FLEURS = {"te": "te_in", "ta": "ta_in", "hi": "hi_in"}
HOLDOUT_N = 100


def get_token() -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def download_fleurs_split(lang: str, split: str = "test") -> tuple[Path, Path]:
    """Download tsv + audio tarball for a (lang, split). Returns (tsv, audio_dir)."""
    from huggingface_hub import hf_hub_download

    iv = LANG_TO_FLEURS[lang]
    token = get_token()

    tsv_path = hf_hub_download(
        FLEURS_REPO, f"data/{iv}/{split}.tsv", repo_type="dataset", token=token,
    )
    tar_path = hf_hub_download(
        FLEURS_REPO, f"data/{iv}/audio/{split}.tar.gz", repo_type="dataset", token=token,
    )

    audio_dir = FLEURS_CACHE / iv / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    if not any(audio_dir.glob("*.wav")):
        print(f"[fleurs] extracting {tar_path} → {audio_dir}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(audio_dir)
    return Path(tsv_path), audio_dir


def parse_fleurs_tsv(tsv: Path) -> list[dict]:
    """FLEURS test.tsv columns: id, filename, transcript, normalized_transcript,
    phoneme_string, duration_ms?, gender."""
    rows = []
    with open(tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for r in reader:
            if len(r) < 4:
                continue
            rows.append({
                "id": r[0], "filename": r[1],
                "transcript": r[2], "normalized": r[3],
                "gender": r[-1] if len(r) >= 7 else "",
            })
    return rows


def plan(lang: str) -> dict:
    iv = LANG_TO_FLEURS[lang]
    return {
        "lang": lang, "fleurs_split": iv,
        "fleurs_repo": FLEURS_REPO,
        "holdout_target": HOLDOUT_N,
        "fields_emitted": ["id", "lang", "entity_class", "text", "audio_path", "entity_tokens", "source"],
        "entity_class": "general",
        "source_label": "fleurs",
        "extraction_strategy": [
            "1. download data/{iv}/test.tsv + audio/test.tar.gz (HF cached)",
            "2. extract WAVs locally to data/fleurs_cache/{iv}/test/",
            "3. random sample 100 with seed=42",
            "4. write JSONL with audio_path pointing to extracted WAV",
        ],
    }


def build(lang: str, n: int = HOLDOUT_N, *, execute: bool, seed: int = 42) -> dict:
    if not execute:
        return {"lang": lang, "executed": False, **plan(lang)}

    out_dir = HOLDOUT_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fleurs_regression.jsonl"

    tsv, audio_dir = download_fleurs_split(lang, "test")
    rows = parse_fleurs_tsv(tsv)
    print(f"[fleurs] {lang}: {len(rows)} rows in test.tsv, sampling {n}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = []
    for r in rows:
        if len(selected) >= n:
            break
        # Audio file lives at audio_dir/test/{filename}
        audio_path = audio_dir / "test" / r["filename"]
        if not audio_path.exists():
            # try without 'test' subdir (depends on tar structure)
            alt = audio_dir / r["filename"]
            if alt.exists():
                audio_path = alt
            else:
                continue
        selected.append({
            "id": f"fleurs_{lang}_{r['id']}",
            "lang": lang,
            "entity_class": "general",
            "text": r["transcript"],
            "audio_path": str(audio_path.relative_to(ROOT)),
            "entity_tokens": [],
            "source": "fleurs",
            "fleurs_id": r["id"],
            "gender": r.get("gender", ""),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "lang": lang, "executed": True,
        "rows_written": len(selected),
        "out_path": str(out_path.relative_to(ROOT)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--lang", choices=list(LANG_TO_FLEURS), required=True)
    b = sub.add_parser("build")
    b.add_argument("--lang", choices=list(LANG_TO_FLEURS), required=True)
    b.add_argument("--n", type=int, default=HOLDOUT_N)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if args.cmd == "plan":
        print(json.dumps(plan(args.lang), indent=2))
        return 0
    if args.cmd == "build":
        print(json.dumps(build(args.lang, n=args.n, execute=args.execute, seed=args.seed), indent=2))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
