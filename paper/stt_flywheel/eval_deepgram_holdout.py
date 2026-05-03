"""Eval Deepgram Nova-3 on the FLEURS or CV25 holdouts (paper #3 baseline).

Reads our existing predictions JSONL (which already has GT text per row)
and adds a Deepgram hypothesis column, then scores EHR + SFR + WER locally
via paper.stt_flywheel.eval_ehr.

Audio source:
  - FLEURS: stream from HF datasets (google/fleurs te_in / hi_in / ta_in test).
  - CV25:   read mp3 from local Downloads dir (or a path you pass in).

Default --dry-run prints API call count + cost. --execute pays.

Pricing: Deepgram Nova-3 ~$0.0043/min. ~85 utts × ~5s avg = ~$0.03 per
holdout per language. Far cheaper than expected — go ahead and run all 3.

CLI::

    uv run python -m paper.stt_flywheel.eval_deepgram_holdout fleurs --lang te --n 100
    uv run python -m paper.stt_flywheel.eval_deepgram_holdout fleurs --lang te --n 100 --execute
    uv run python -m paper.stt_flywheel.eval_deepgram_holdout cv25   --lang te \\
        --cv25-dir /Users/pushpak/Downloads/cv-corpus-25.0-2026-03-09 --execute
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORECARD_DIR = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"

DEEPGRAM_PRICE_PER_MIN = 0.0043


def _wav_bytes_from_array(audio, sr: int = 16_000) -> bytes:
    """Encode a float32 array as a WAV blob for Deepgram POST."""
    import soundfile as sf
    import numpy as np
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype="float32"), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _load_fleurs(lang: str, n: int):
    from datasets import load_dataset, Audio
    fleurs_lang = {"te": "te_in", "hi": "hi_in", "ta": "ta_in"}[lang]
    ds = load_dataset("google/fleurs", fleurs_lang, split="test", trust_remote_code=True)
    if n is not None and n < len(ds):
        ds = ds.select(range(n))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    rows = []
    for i, row in enumerate(ds):
        rows.append({
            "id": f"fleurs_{lang}_{row.get('id', i)}",
            "lang": lang,
            "entity_class": "general",
            "text": row["transcription"],
            "entity_tokens": [],
            "source": "fleurs",
            "_audio_bytes": _wav_bytes_from_array(row["audio"]["array"]),
            "_duration_s": len(row["audio"]["array"]) / 16_000,
        })
    return rows


def _load_cv25(lang: str, cv25_dir: Path, n: int | None):
    """Read CV25 test split. cv25_dir is the parent that contains <lang>/test.tsv + clips/."""
    import librosa
    import sys as _sys
    csv.field_size_limit(_sys.maxsize)  # Tamil rows can exceed default 128k limit
    lang_dir = cv25_dir / lang
    tsv_path = lang_dir / "test.tsv"
    clips_dir = lang_dir / "clips"
    if not tsv_path.exists():
        raise FileNotFoundError(tsv_path)
    rows_meta: list[dict] = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if not r.get("path") or not r.get("sentence"):
                continue
            rows_meta.append({"path": r["path"], "sentence": r["sentence"]})
    if n is not None and n < len(rows_meta):
        rows_meta = rows_meta[:n]
    rows = []
    for r in rows_meta:
        audio_path = clips_dir / r["path"]
        try:
            audio, _sr = librosa.load(str(audio_path), sr=16_000, mono=True)
        except Exception as e:
            print(f"WARN: load fail {audio_path}: {e}", file=sys.stderr)
            continue
        rows.append({
            "id": f"cv25_{lang}_{Path(r['path']).stem}",
            "lang": lang,
            "entity_class": "general",
            "text": r["sentence"],
            "entity_tokens": [],
            "source": "cv25",
            "_audio_bytes": _wav_bytes_from_array(audio),
            "_duration_s": len(audio) / 16_000,
        })
    return rows


def _transcribe_all(rows: list[dict], lang: str) -> list[dict]:
    from stt.backends import deepgram as dg
    out = []
    for i, r in enumerate(rows):
        try:
            hyp = dg.transcribe(r["_audio_bytes"], language=lang)
        except Exception as e:
            print(f"WARN: deepgram failed on {r['id']}: {e}", file=sys.stderr)
            hyp = ""
        out_row = {k: v for k, v in r.items() if not k.startswith("_")}
        out_row["deepgram_hyp"] = hyp
        out.append(out_row)
        if (i + 1) <= 3 or (i + 1) % 25 == 0:
            print(f"[dg] [{i+1}/{len(rows)}] GT: {r['text'][:60]}")
            print(f"[dg]               HY: {hyp[:60]}")
    return out


def _score(rows: list[dict], lang: str) -> dict:
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None  # type: ignore
    scores, sfrs, gts, hyps = [], [], [], []
    for r in rows:
        hyp = r["deepgram_hyp"]
        scores.append(score_row(r, hyp))
        sfrs.append(script_fidelity_rate(hyp, lang))
        gts.append(r["text"]); hyps.append(hyp)
    agg = aggregate(scores)
    out = {
        "system": "deepgram_nova_3",
        "n": len(rows),
        "ehr": agg["ehr"],
        "sfr_mean": round(sum(sfrs) / len(sfrs), 4) if sfrs else None,
    }
    if jiwer:
        try:
            out["wer"] = round(jiwer.wer(gts, hyps), 4)
            out["cer"] = round(jiwer.cer(gts, hyps), 4)
        except Exception as e:
            out["wer_error"] = str(e)
    return out


def _load_jsonl_holdout(jsonl_path: Path, n: int | None):
    """Load a holdout from a JSONL with audio_path + text fields."""
    import librosa
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ap = r["audio_path"]
        if not Path(ap).is_absolute():
            ap = REPO_ROOT / ap
        try:
            audio, _sr = librosa.load(str(ap), sr=16_000, mono=True)
        except Exception as e:
            print(f"WARN: load fail {ap}: {e}", file=sys.stderr)
            continue
        rows.append({
            "id": r["id"],
            "lang": r["lang"],
            "entity_class": r.get("entity_class", "general"),
            "text": r["text"],
            "entity_tokens": r.get("entity_tokens", []),
            "source": r.get("source", "jsonl"),
            "_audio_bytes": _wav_bytes_from_array(audio),
            "_duration_s": len(audio) / 16_000,
        })
        if n is not None and len(rows) >= n:
            break
    return rows


def _run(holdout: str, lang: str, n: int | None, cv25_dir: Path | None, execute: bool, jsonl_path: Path | None = None):
    if holdout == "fleurs":
        rows = _load_fleurs(lang, n=n or 100)
    elif holdout == "cv25":
        if cv25_dir is None:
            print("ERROR: --cv25-dir required for cv25 holdout", file=sys.stderr)
            return 2
        rows = _load_cv25(lang, cv25_dir, n=n)
    elif holdout == "jsonl":
        if jsonl_path is None:
            print("ERROR: --jsonl required for jsonl holdout", file=sys.stderr)
            return 2
        rows = _load_jsonl_holdout(jsonl_path, n=n)
    else:
        print(f"unknown holdout {holdout}", file=sys.stderr)
        return 2

    total_min = sum(r["_duration_s"] for r in rows) / 60.0
    cost = total_min * DEEPGRAM_PRICE_PER_MIN
    print(f"=== Plan ===")
    print(f"  holdout       : {holdout} ({lang}, n={len(rows)})")
    print(f"  total minutes : {total_min:.2f}")
    print(f"  est. cost     : ${cost:.4f}")

    if not execute:
        print("\n[dry-run] no API calls. Pass --execute to run.")
        return 0

    print(f"\n[exec] calling Deepgram Nova-3 × {len(rows)} clips...")
    hyp_rows = _transcribe_all(rows, lang)

    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    holdout_label = holdout if holdout != "jsonl" else jsonl_path.stem
    pred_path = SCORECARD_DIR / f"deepgram_{lang}_{holdout_label}_n{len(rows)}_predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in hyp_rows))
    print(f"[exec] predictions → {pred_path}")

    summary = _score(hyp_rows, lang)
    summary["holdout"] = f"{holdout_label}_{lang}_test"
    score_path = SCORECARD_DIR / f"deepgram_{lang}_{holdout_label}_n{len(rows)}_scorecard.json"
    score_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[exec] scorecard → {score_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fleurs", "cv25", "jsonl"):
        p = sub.add_parser(name)
        p.add_argument("--lang", required=True, choices=["te", "hi", "ta"])
        p.add_argument("--n", type=int, default=None)
        p.add_argument("--execute", action="store_true")
        if name == "cv25":
            p.add_argument("--cv25-dir", type=Path, required=True,
                           help="Parent dir containing {lang}/test.tsv + {lang}/clips/")
        if name == "jsonl":
            p.add_argument("--jsonl", type=Path, required=True,
                           help="Holdout JSONL with id/text/audio_path fields")
        p.set_defaults(func=lambda args, holdout=name: _run(
            holdout, args.lang, args.n,
            getattr(args, "cv25_dir", None),
            args.execute,
            jsonl_path=getattr(args, "jsonl", None),
        ))
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
