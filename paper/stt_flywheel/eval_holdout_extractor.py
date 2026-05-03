"""Build the entity-dense eval holdout from real recordings.

PLAN §7 wants 300 utts/lang from real human recordings:
  - 100 from IndicVoices held-back speakers
  - 100 newly recorded by 1 native speaker per language
  - 100 from CC-licensed Indian podcasts/news with verified transcripts

This module covers the IndicVoices held-back extraction (the biggest, most
automatable chunk). The other 200 utts/lang are handled by:
  - `record_native_holdout.py` (sister script — recording prompts + UI; manual)
  - manual YouTube curation (out-of-scope for v1; see RUNBOOK §Phase 4)

IndicVoices held-back extraction strategy:

  1. Pick speakers NOT used in Phase 3 training. Use the manifest from
     `data/stt_flywheel/manifests/{lang}_train.jsonl` to get the SET of
     IndicVoices speakers we DID use, then take complement from the
     IndicVoices full speaker list.
  2. From held-back speakers' clips, filter for entity-dense content
     using the same `entity_token_tagger` from `data_pipeline`. A clip is
     "entity-dense" if it has ≥1 tagged entity from {digits, currency,
     pincode, brand, proper_noun}.
  3. Sample 100 clips per language, balanced across entity classes.
  4. Output JSONL compatible with `eval_ehr.score_jsonl_pair`:
       {id, lang, entity_class, text, audio_path, entity_tokens, source}

Default --dry-run: prints sample-size estimates without actually iterating
the IndicVoices parquet shards. --execute downloads + filters.

CLI::

    # Plan: how many held-back speakers / clips per lang
    uv run python -m paper.stt_flywheel.eval_holdout_extractor plan

    # Build (downloads IV shards if not cached)
    uv run python -m paper.stt_flywheel.eval_holdout_extractor build \\
        --lang te --n 100 --execute
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_DIR = ROOT / "data" / "stt_flywheel" / "holdouts"
TRAIN_MANIFEST_DIR = ROOT / "data" / "stt_flywheel" / "manifests"

INDIC_VOICES_REPO = "ai4bharat/IndicVoices"
LANG_TO_INDIC_VOICES = {"te": "telugu", "ta": "tamil", "hi": "hindi"}

# Per-language holdout target
HOLDOUT_N = 100

# Entity classes to extract — same set as training
ENTITY_CLASSES = ("digits", "currency", "addresses", "brands", "codemix", "proper_nouns")


def used_speaker_ids(lang: str) -> set[str]:
    """Set of IndicVoices speaker_ids used in this language's training manifest."""
    train_path = TRAIN_MANIFEST_DIR / f"{lang}_train.jsonl"
    if not train_path.exists():
        return set()
    used = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # IndicVoices speaker_id is exposed in the row when source=='real'
        if r.get("source") == "real" and "speaker_id" in r:
            used.add(r["speaker_id"])
    return used


def list_indicvoices_shards(lang: str, token: str | None = None) -> list[str]:
    """List parquet shards for a language in the IndicVoices repo."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(INDIC_VOICES_REPO, repo_type="dataset", token=token)
    iv_lang = LANG_TO_INDIC_VOICES[lang]
    return sorted(f for f in files if f.startswith(f"{iv_lang}/") and f.endswith(".parquet"))


def stream_indicvoices_rows(lang: str, max_shards: int = 10, token: str | None = None) -> Iterator[dict]:
    """Stream rows from IndicVoices parquets. Returns dicts with at least
    {audio_bytes, transcript, speaker_id} (exact field names depend on
    upstream schema)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    shards = list_indicvoices_shards(lang, token=token)[:max_shards]
    for shard in shards:
        local = hf_hub_download(INDIC_VOICES_REPO, shard, repo_type="dataset", token=token)
        table = pq.read_table(local)
        for batch in table.to_batches():
            for r in batch.to_pylist():
                yield r


def is_entity_dense(text: str, lang: str) -> tuple[bool, list[str]]:
    """Return (True, [entity_class_hits]) if text contains at least one
    entity from any of our classes, using the SAME `entity_token_tagger`
    that we used at training time so train/eval coverage matches."""
    from paper.stt_flywheel.data_pipeline import entity_token_tagger
    classes_with_hit: list[str] = []
    for cls in ENTITY_CLASSES:
        try:
            toks = entity_token_tagger(text, cls)
        except Exception:
            toks = []
        if toks:
            classes_with_hit.append(cls)
    return (len(classes_with_hit) > 0, classes_with_hit)


def plan(lang: str, token: str | None = None) -> dict:
    used = used_speaker_ids(lang)
    shards = list_indicvoices_shards(lang, token=token)
    return {
        "lang": lang,
        "indicvoices_repo": INDIC_VOICES_REPO,
        "iv_lang_dir": LANG_TO_INDIC_VOICES[lang],
        "n_train_speakers_used": len(used),
        "n_iv_shards_total": len(shards),
        "first_5_shards": shards[:5],
        "holdout_target": HOLDOUT_N,
        "entity_classes": list(ENTITY_CLASSES),
        "extraction_strategy": [
            "1. iterate IV parquet shards",
            "2. filter rows by held-back speaker (NOT in train manifest)",
            "3. filter rows by entity-density (≥1 entity_token in text)",
            "4. sample 100 with class-balance (≈17 per class)",
            "5. write JSONL with audio_bytes → wav file + ground-truth entity_tokens",
        ],
    }


def build_holdout(lang: str, n: int = HOLDOUT_N, *, execute: bool, max_shards: int = 10) -> dict:
    """Build the entity-dense holdout for a language."""
    if not execute:
        print(f"[dry-run] would extract {n} entity-dense IndicVoices clips for {lang} "
              f"from up to {max_shards} parquet shards")
        return {"lang": lang, "executed": False}

    used = used_speaker_ids(lang)
    out_dir = HOLDOUT_DIR / lang / "iv_held_back"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = HOLDOUT_DIR / lang / "iv_held_back.jsonl"

    token = None
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"')
                break

    by_class: dict = {c: [] for c in ENTITY_CLASSES}
    seen_ids = set()

    print(f"[hldout] {lang}: scanning up to {max_shards} IV shards, holdout target n={n}")
    rows_seen = 0
    rows_kept = 0
    for r in stream_indicvoices_rows(lang, max_shards=max_shards, token=token):
        rows_seen += 1
        # IV schema (verified empirically — adjust if upstream changes):
        # {audio_filepath, audio_bytes, transcript, speaker_id, ...}
        spk = r.get("speaker_id") or r.get("speaker") or r.get("client_id") or ""
        if spk in used:
            continue
        text = (r.get("transcript") or r.get("text") or "").strip()
        if not text:
            continue
        is_dense, classes = is_entity_dense(text, lang)
        if not is_dense:
            continue
        # Pick the first matching class for this row (avoids double-counting
        # in class budgets; later we balance).
        cls = classes[0]
        if len(by_class[cls]) >= n // len(ENTITY_CLASSES) + 5:
            continue  # bucket full
        rid = f"iv_holdout_{lang}_{cls}_{rows_kept:04d}"
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        # Save audio
        audio_path = out_dir / f"{rid}.wav"
        audio_bytes = r.get("audio_bytes") or r.get("audio", {}).get("bytes")
        if audio_bytes is None:
            continue
        audio_path.write_bytes(audio_bytes)
        from paper.stt_flywheel.data_pipeline import entity_token_tagger
        by_class[cls].append({
            "id": rid, "lang": lang, "entity_class": cls, "text": text,
            "audio_path": str(audio_path.relative_to(ROOT)),
            "entity_tokens": entity_token_tagger(text, cls),
            "source": "indicvoices_held_back",
            "speaker_id": spk,
        })
        rows_kept += 1
        if rows_kept >= n + 30:  # over-fetch a bit; balance below
            break

    # Class-balance sample to exactly n
    final = []
    per_class = max(1, n // len(ENTITY_CLASSES))
    rng = random.Random(42)
    for cls, rows in by_class.items():
        rng.shuffle(rows)
        final.extend(rows[:per_class])
    # Top up to exactly n if some classes were short
    while len(final) < n:
        # Pick any unused row
        pool = [r for cls, rs in by_class.items() for r in rs if r not in final]
        if not pool:
            break
        rng.shuffle(pool)
        final.extend(pool[:n - len(final)])

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in final[:n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[hldout] {lang}: scanned {rows_seen} rows, kept {rows_kept}, wrote {len(final[:n])} → {out_jsonl.relative_to(ROOT)}")
    return {
        "lang": lang, "executed": True,
        "rows_seen": rows_seen, "rows_kept": rows_kept,
        "rows_written": len(final[:n]),
        "out_path": str(out_jsonl.relative_to(ROOT)),
        "by_class": {cls: len(rs[:per_class]) for cls, rs in by_class.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan", help="Plan-only.")
    pp.add_argument("--lang", choices=("te", "ta", "hi"), required=True)
    bp = sub.add_parser("build", help="Build the holdout JSONL.")
    bp.add_argument("--lang", choices=("te", "ta", "hi"), required=True)
    bp.add_argument("--n", type=int, default=HOLDOUT_N)
    bp.add_argument("--max-shards", type=int, default=10,
                    help="Cap shards scanned for cost / time control.")
    bp.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    token = None
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"')
                break

    if args.cmd == "plan":
        s = plan(args.lang, token=token)
        print(json.dumps(s, indent=2))
        return 0
    if args.cmd == "build":
        s = build_holdout(args.lang, n=args.n, execute=args.execute, max_shards=args.max_shards)
        print(json.dumps(s, indent=2, ensure_ascii=False))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
