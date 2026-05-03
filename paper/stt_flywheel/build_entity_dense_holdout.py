"""Build a class-balanced entity-dense holdout from one held-out synth system.

Training manifests at data/stt_flywheel/manifests/{lang}_train.jsonl mix
audio from {praxy_lora, praxy_indicf5, elevenlabs, cartesia}. For β-paper
we retrain LoRA WITHOUT cartesia rows; cartesia becomes the entity-dense
held-out set.

Outputs holdout JSONL with audio_path/text/entity_tokens/entity_class so
that all our evals (eval_te_jsonl_holdout, eval_deepgram_holdout, vasista
baseline) can score it.

CLI::
    uv run python -m paper.stt_flywheel.build_entity_dense_holdout \\
        --lang te --hold-system cartesia --n-per-class 17 --execute
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_DIR = ROOT / "data" / "stt_flywheel" / "holdouts"


def build(lang: str, hold_system: str, n_per_class: int, *, execute: bool, seed: int = 1337) -> dict:
    manifest = ROOT / "data" / "stt_flywheel" / "manifests" / f"{lang}_train.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    held = [r for r in rows if r.get("synth_system") == hold_system]
    by_class: dict[str, list] = defaultdict(list)
    for r in held:
        by_class[r["entity_class"]].append(r)
    print(f"[holdout] {lang}: {len(held)} rows from {hold_system}, classes:")
    for c, lst in by_class.items():
        print(f"  {c}: {len(lst)}")

    rng = random.Random(seed)
    sampled: list = []
    for c, lst in by_class.items():
        rng.shuffle(lst)
        sampled.extend(lst[:n_per_class])
    rng.shuffle(sampled)

    if not execute:
        print(f"\n[dry-run] would sample {len(sampled)} rows; pass --execute to write")
        return {"executed": False, "n_sampled": len(sampled)}

    out_jsonl = HOLDOUT_DIR / lang / f"entity_dense_{hold_system}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for r in sampled:
        out_rows.append({
            "id": Path(r["audio_path"]).stem,
            "lang": lang,
            "entity_class": r["entity_class"],
            "text": r["text"],
            "audio_path": r["audio_path"],
            "entity_tokens": r.get("entity_tokens", []),
            "source": f"synth_{hold_system}",
            "duration_s": r.get("duration_s"),
        })
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))
    print(f"\n[holdout] wrote {len(out_rows)} rows → {out_jsonl}")
    return {
        "executed": True,
        "n_kept": len(out_rows),
        "by_class": {c: sum(1 for r in out_rows if r["entity_class"] == c)
                     for c in {r["entity_class"] for r in out_rows}},
        "out_path": str(out_jsonl.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["te", "ta", "hi"])
    ap.add_argument("--hold-system", required=True,
                    choices=["cartesia", "elevenlabs", "praxy_lora", "praxy_indicf5", "praxy_vanilla"])
    ap.add_argument("--n-per-class", type=int, default=17)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    summary = build(args.lang, args.hold_system, args.n_per_class, execute=args.execute, seed=args.seed)
    print(json.dumps(summary, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
