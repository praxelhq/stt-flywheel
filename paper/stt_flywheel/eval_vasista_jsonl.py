"""vasista22 baseline on a JSONL+audio-prefix holdout (e.g., entity-dense).

Mirrors eval_te_jsonl_holdout.py shape but uses vasista22/whisper-{lang}-large-v2
as the model. Works with audio living on the Modal volume at /cache/stt_r2/...
or any path; we pass the rows + audio_prefix as args.

CLI::
    uv run python -m paper.stt_flywheel.eval_vasista_jsonl run \\
        --lang te \\
        --jsonl data/stt_flywheel/holdouts/te/entity_dense_cartesia.jsonl \\
        --audio-prefix /cache/stt_r2/audio/te/cartesia --execute
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

try:
    import modal  # type: ignore
    _HAS_MODAL = True
except Exception:
    modal = None
    _HAS_MODAL = False

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "praxy-stt-vasista-jsonl"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"

VASISTA_MODEL = {
    "te": "vasista22/whisper-telugu-large-v2",
    "ta": "vasista22/whisper-tamil-large-v2",
    "hi": "vasista22/whisper-hindi-large-v2",
}

if _HAS_MODAL:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .pip_install(
            "torch==2.4.0", "torchaudio==2.4.0", "numpy<2",
            "transformers==4.36.2", "accelerate>=0.30",
            "soundfile", "librosa", "huggingface_hub>=0.25,<0.30",
        )
    )


def _impl(rows: list[dict], lang: str, audio_prefix: str) -> dict:
    import os, torch, librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    os.environ.setdefault("HF_HOME", "/cache/hf")
    model_id = VASISTA_MODEL[lang]
    lang_name = {"te": "telugu", "ta": "tamil", "hi": "hindi"}[lang]
    processor = WhisperProcessor.from_pretrained(model_id, language=lang_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    forced = processor.tokenizer.get_decoder_prompt_ids(language=lang_name, task="transcribe")
    model.config.forced_decoder_ids = forced
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.suppress_tokens = []

    out = []
    for i, r in enumerate(rows):
        ap = f"{audio_prefix}/{Path(r['audio_path']).name}"
        try:
            audio, _ = librosa.load(ap, sr=16_000, mono=True)
            feats = processor.feature_extractor(
                audio, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                pred_ids = model.generate(feats, max_new_tokens=400, num_beams=1, do_sample=False)
            hyp = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[vasista] WARN {ap}: {e}", flush=True)
            hyp = ""
        out.append({**{k: v for k, v in r.items() if not k.startswith("_")}, "vasista_hyp": hyp})
        if (i + 1) <= 3 or (i + 1) % 25 == 0:
            print(f"[vasista] [{i+1}] GT: {r['text'][:60]}", flush=True)
            print(f"[vasista]      HY: {hyp[:60]}", flush=True)
    return {"n": len(out), "rows": out}


if _HAS_MODAL:
    @app.function(
        image=image, gpu="A10G", volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60 * 60 * 2,
    )
    def eval_jsonl(rows: list[dict], lang: str, audio_prefix: str) -> dict:
        return _impl(rows, lang, audio_prefix)


def _score_local(rows: list[dict], lang: str) -> dict:
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None
    scores, sfrs, gts, hyps = [], [], [], []
    for r in rows:
        scores.append(score_row(r, r["vasista_hyp"]))
        sfrs.append(script_fidelity_rate(r["vasista_hyp"], lang))
        gts.append(r["text"]); hyps.append(r["vasista_hyp"])
    agg = aggregate(scores)
    out = {"n": len(rows), "ehr": agg["ehr"],
           "sfr_mean": round(sum(sfrs)/len(sfrs),4) if sfrs else None,
           "by_class": agg.get("by_class")}
    if jiwer:
        try:
            out["wer"] = round(jiwer.wer(gts, hyps), 4)
            out["cer"] = round(jiwer.cer(gts, hyps), 4)
        except Exception as e:
            out["wer_error"] = str(e)
    return out


def _cmd_run(args):
    if not args.execute:
        print(f"[dry-run] would run vasista22-{args.lang} on {args.jsonl} via Modal A10G (~$0.55)")
        return 0
    if not _HAS_MODAL:
        return 2
    raw = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    rows = [{
        "id": r["id"], "lang": r.get("lang", args.lang),
        "entity_class": r.get("entity_class", "general"),
        "text": r["text"], "entity_tokens": r.get("entity_tokens", []),
        "source": r.get("source"), "audio_path": r["audio_path"],
    } for r in raw]
    print(f"[exec] launching vasista22 on {args.jsonl} (n={len(rows)}, prefix={args.audio_prefix})")
    with app.run():
        result = eval_jsonl.remote(rows=rows, lang=args.lang, audio_prefix=args.audio_prefix)
    out_rows = result["rows"]
    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.jsonl.stem
    pred = out_dir / f"vasista_{args.lang}_{label}_n{len(out_rows)}_predictions.jsonl"
    pred.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))
    summary = _score_local(out_rows, args.lang)
    summary["holdout"] = label
    summary["model"] = VASISTA_MODEL[args.lang]
    score = out_dir / f"vasista_{args.lang}_{label}_n{len(out_rows)}_scorecard.json"
    score.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[exec] predictions → {pred}")
    print(f"[exec] scorecard → {score}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lang", required=True, choices=["te", "ta", "hi"])
    r.add_argument("--jsonl", type=Path, required=True)
    r.add_argument("--audio-prefix", required=True,
                   help="Modal volume path prefix, e.g., /cache/stt_r2/audio/te/cartesia")
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
