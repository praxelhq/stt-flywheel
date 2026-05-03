"""Build a 'general conversational' IV-Te / IV-Hi / IV-Ta holdout for eval.

The original eval_holdout_extractor filters for entity-dense rows, but IV
transcripts are fully spelled-out in native script — 0 digits / addresses
hit the tagger. We instead sample ``n`` random rows from non-Read scenarios
(Conversation + Extempore) for a 'natural conversational' holdout.

This holdout provides a 3rd evaluation dimension beyond FLEURS-read-prose
and CV25-read-prose: real conversational speech from held-out speakers.

CLI::
    uv run python -m paper.stt_flywheel.build_iv_general_holdout \\
        --lang te --n 100 --max-shards 5 --execute
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_DIR = ROOT / "data" / "stt_flywheel" / "holdouts"

INDIC_VOICES_REPO = "ai4bharat/IndicVoices"
LANG_DIR = {"te": "telugu", "hi": "hindi", "ta": "tamil"}


def _hf_token() -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def build(lang: str, n: int, max_shards: int, *, execute: bool, seed: int = 1337) -> dict:
    if not execute:
        print(f"[dry-run] would extract {n} conversational rows from {lang} ({max_shards} shards)")
        return {"executed": False}

    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    token = _hf_token()
    out_dir = HOLDOUT_DIR / lang / "iv_general"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = HOLDOUT_DIR / lang / "iv_general.jsonl"

    rng = random.Random(seed)
    candidates: list[dict] = []
    rows_seen = 0

    files = list_repo_files(INDIC_VOICES_REPO, repo_type="dataset", token=token)
    shards = sorted(f for f in files if f.startswith(f"{LANG_DIR[lang]}/") and f.endswith(".parquet"))
    print(f"[hldout] {lang}: found {len(shards)} parquets, scanning first {max_shards}")

    for i, fname in enumerate(shards[:max_shards]):
        try:
            local = hf_hub_download(INDIC_VOICES_REPO, fname, repo_type="dataset", token=token)
        except Exception as e:
            print(f"[hldout] {lang}: shard {fname} download failed: {e}; stopping")
            break
        table = pq.read_table(local)
        for batch in table.to_batches():
            for r in batch.to_pylist():
                rows_seen += 1
                if r.get("scenario") == "Read":
                    continue
                text = (r.get("verbatim") or r.get("text") or "").strip()
                if not text or len(text) < 10:
                    continue
                duration = r.get("duration") or 0
                if duration < 1.0 or duration > 25.0:
                    continue
                audio = r.get("audio_filepath") or {}
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not audio_bytes:
                    continue
                candidates.append({
                    "text": text,
                    "duration": duration,
                    "speaker_id": r.get("speaker_id", ""),
                    "scenario": r.get("scenario", ""),
                    "task_name": r.get("task_name", ""),
                    "audio_bytes": audio_bytes,
                })
        print(f"[hldout] {lang}: shard {i} done; rows_seen={rows_seen}, candidates={len(candidates)}")
        if len(candidates) >= n * 5:
            # 5x oversample is enough for random selection
            break

    if not candidates:
        print(f"[hldout] {lang}: NO candidates found", file=sys.stderr)
        return {"executed": True, "n_kept": 0, "rows_seen": rows_seen}

    rng.shuffle(candidates)
    chosen = candidates[:n]

    rows_out = []
    for i, c in enumerate(chosen):
        rid = f"iv_general_{lang}_{i:04d}"
        audio_path = out_dir / f"{rid}.wav"
        audio_path.write_bytes(c["audio_bytes"])
        rows_out.append({
            "id": rid,
            "lang": lang,
            "entity_class": "general",
            "text": c["text"],
            "audio_path": str(audio_path.relative_to(ROOT)),
            "entity_tokens": [],
            "source": "indicvoices_general",
            "speaker_id": c["speaker_id"],
            "scenario": c["scenario"],
            "task_name": c["task_name"],
            "duration_s": c["duration"],
        })

    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out))
    print(f"[hldout] {lang}: wrote {len(rows_out)} rows → {out_jsonl}")
    return {
        "executed": True,
        "rows_seen": rows_seen,
        "n_kept": len(rows_out),
        "out_path": str(out_jsonl.relative_to(ROOT)),
        "scenarios": {s: sum(1 for r in rows_out if r["scenario"] == s)
                      for s in {r["scenario"] for r in rows_out}},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["te", "ta", "hi"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-shards", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    summary = build(args.lang, args.n, args.max_shards, execute=args.execute, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
