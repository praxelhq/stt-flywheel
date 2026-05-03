"""Eval Te-r2 LoRA vs vanilla Whisper-v3 on FLEURS te_in test split.

Self-contained: downloads FLEURS inside the Modal container (no upload of audio
needed), loads vanilla whisper-large-v3 + Te-r2 LoRA from the volume, runs
both, returns per-utterance predictions. Scoring (EHR + SFR + WER) happens
locally via paper.stt_flywheel.eval_ehr + jiwer.

Default --dry-run prints plan + cost. --execute submits the Modal job.

CLI::

    uv run python -m paper.stt_flywheel.eval_te_fleurs plan
    uv run python -m paper.stt_flywheel.eval_te_fleurs run --n 50 --execute   # smoke
    uv run python -m paper.stt_flywheel.eval_te_fleurs run --n 200 --execute  # full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import modal  # type: ignore
    _HAS_MODAL = True
except Exception:  # noqa: BLE001
    modal = None  # type: ignore
    _HAS_MODAL = False

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "praxy-stt-eval-te-fleurs"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"

LORA_CKPT_DEFAULT = "/cache/stt_r2/te/r2/checkpoint-6000"
BASE_MODEL = "openai/whisper-large-v3"

if _HAS_MODAL:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .pip_install(
            "torch==2.4.0",
            "torchaudio==2.4.0",
            "numpy<2",
            "transformers==4.49.0",
            "accelerate>=0.34",
            "peft>=0.13",
            "datasets>=2.20,<3.0",
            "soundfile",
            "librosa",
            "huggingface_hub>=0.25",
        )
    )


def _eval_impl(
    n: int,
    lora_ckpt: str,
    base_model: str = BASE_MODEL,
    num_beams: int = 1,
) -> dict:
    """Runs inside the Modal container. Returns dict with predictions per system."""
    import os
    import torch
    from datasets import load_dataset, Audio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

    os.environ.setdefault("HF_HOME", "/cache/hf")

    print(f"[eval] loading FLEURS te_in test split (cap={n})", flush=True)
    ds = load_dataset("google/fleurs", "te_in", split="test", trust_remote_code=True)
    if n is not None and n < len(ds):
        ds = ds.select(range(n))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    print(f"[eval] {len(ds)} utterances", flush=True)

    print(f"[eval] loading base {base_model}", flush=True)
    processor = WhisperProcessor.from_pretrained(base_model, language="telugu", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    base.eval()
    base.generation_config.language = "telugu"
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    base.generation_config.suppress_tokens = []

    def _transcribe(model) -> list[str]:
        outs = []
        for i, row in enumerate(ds):
            audio = row["audio"]["array"]
            feats = processor.feature_extractor(
                audio, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                pred_ids = model.generate(
                    feats,
                    max_new_tokens=400,
                    num_beams=num_beams,
                    do_sample=False,
                    language="telugu",
                    task="transcribe",
                )
            text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
            outs.append(text)
            if (i + 1) <= 3 or (i + 1) % 25 == 0:
                print(f"[eval] [{i+1}/{len(ds)}] GT: {row['transcription'][:60]}", flush=True)
                print(f"[eval]                 HY: {text[:60]}", flush=True)
        return outs

    print("[eval] transcribing with VANILLA whisper-v3", flush=True)
    vanilla_hyps = _transcribe(base)

    print(f"[eval] loading LoRA adapter from {lora_ckpt}", flush=True)
    lora_model = PeftModel.from_pretrained(base, lora_ckpt)
    lora_model.eval()

    print("[eval] transcribing with Te-r2 LoRA", flush=True)
    lora_hyps = _transcribe(lora_model)

    rows = []
    for i, row in enumerate(ds):
        rows.append({
            "id": f"fleurs_te_{row.get('id', i)}",
            "lang": "te",
            "entity_class": "general",
            "text": row["transcription"],
            "entity_tokens": [],
            "source": "fleurs",
            "vanilla_hyp": vanilla_hyps[i],
            "lora_hyp": lora_hyps[i],
        })
    return {"n": len(rows), "rows": rows}


if _HAS_MODAL:
    @app.function(
        image=image,
        gpu="A10G",
        volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)],
        timeout=60 * 60 * 2,
    )
    def eval_fleurs(n: int, lora_ckpt: str = LORA_CKPT_DEFAULT, num_beams: int = 1) -> dict:
        return _eval_impl(n=n, lora_ckpt=lora_ckpt, num_beams=num_beams)


# ---------------------------------------------------------------------------
# Local scoring
# ---------------------------------------------------------------------------


def _score_local(rows: list[dict], lang: str = "te") -> dict:
    """Compute EHR + SFR + WER for vanilla and lora hypotheses."""
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer  # type: ignore
    except Exception:
        jiwer = None  # type: ignore

    def _score_system(hyp_key: str) -> dict:
        scores = []
        sfrs = []
        wers = []
        gts = []
        hyps = []
        for r in rows:
            hyp = r[hyp_key]
            scores.append(score_row(r, hyp))
            sfrs.append(script_fidelity_rate(hyp, lang))
            gts.append(r["text"])
            hyps.append(hyp)
        agg = aggregate(scores)
        out = {
            "n": len(rows),
            "ehr": agg["ehr"],
            "sfr_mean": round(sum(sfrs) / len(sfrs), 4) if sfrs else None,
            "by_class": agg.get("by_class"),
        }
        if jiwer is not None:
            try:
                out["wer"] = round(jiwer.wer(gts, hyps), 4)
                out["cer"] = round(jiwer.cer(gts, hyps), 4)
            except Exception as e:
                out["wer_error"] = str(e)
        return out

    return {
        "vanilla_whisper_v3": _score_system("vanilla_hyp"),
        "praxy_te_r2": _score_system("lora_hyp"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    n = args.n
    cost = round(n / 200 * 1.10, 2)  # ~1 A10G-hr per 200 utts (rough)
    print("=== Plan ===")
    print(f"  holdout         : FLEURS te_in test (cap={n})")
    print(f"  base            : openai/whisper-large-v3")
    print(f"  lora ckpt       : {args.lora_ckpt}")
    print(f"  metrics (local) : EHR, SFR, WER, CER")
    print(f"  est. duration   : ~{max(5, n // 30)} min on A10G")
    print(f"  est. cost       : ~${cost} (@ $1.10/hr A10G)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.execute:
        return _cmd_plan(args)
    if not _HAS_MODAL:
        print("ERROR: modal package not installed; cannot --execute", file=sys.stderr)
        return 2

    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exec] launching Modal eval (n={args.n}, ckpt={args.lora_ckpt}, beams={args.num_beams})", flush=True)
    with app.run():
        result = eval_fleurs.remote(n=args.n, lora_ckpt=args.lora_ckpt, num_beams=args.num_beams)

    rows = result["rows"]
    suffix = f"_b{args.num_beams}" if args.num_beams > 1 else ""
    pred_path = out_dir / f"praxy_te_r2_fleurs_n{len(rows)}{suffix}_predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"[exec] predictions → {pred_path}")

    summary = _score_local(rows, lang="te")
    summary["n_rows"] = len(rows)
    summary["holdout"] = "fleurs_te_in_test"
    summary["lora_ckpt"] = args.lora_ckpt
    summary["num_beams"] = args.num_beams
    score_path = out_dir / f"praxy_te_r2_fleurs_n{len(rows)}{suffix}_scorecard.json"
    score_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[exec] scorecard → {score_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--lora-ckpt", default=LORA_CKPT_DEFAULT)
    p.set_defaults(func=_cmd_plan, execute=False)

    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--lora-ckpt", default=LORA_CKPT_DEFAULT)
    r.add_argument("--num-beams", type=int, default=1)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
