"""β-paper eval on an arbitrary JSONL holdout (e.g. pushpak_native).

Same model loading recipe as eval_beta.py (vasista22 + LoRA from
/cache/stt_rb/{lang}/rb/checkpoint-4000) but takes a JSONL + audio_prefix
instead of the fixed 4-holdout bundle.

CLI::
    uv run python -m paper.stt_flywheel.eval_beta_jsonl run \\
        --lang te \\
        --jsonl data/stt_flywheel/holdouts/te/pushpak_native.jsonl \\
        --audio-prefix /cache/pushpak_native/te --execute
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

try:
    import modal  # type: ignore
    _HAS_MODAL = True
except Exception:
    modal = None  # type: ignore
    _HAS_MODAL = False

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "praxy-stt-eval-beta-jsonl"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"

VASISTA_MODEL = {
    "te": "vasista22/whisper-telugu-large-v2",
    "ta": "vasista22/whisper-tamil-large-v2",
    "hi": "vasista22/whisper-hindi-large-v2",
}
LANG_NAME = {"te": "telugu", "ta": "tamil", "hi": "hindi"}

if _HAS_MODAL:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .pip_install(
            "torch==2.4.0", "torchaudio==2.4.0", "numpy<2",
            "transformers==4.36.2", "accelerate==0.30.0", "peft==0.10.0",
            "soundfile", "librosa", "huggingface_hub>=0.25,<0.30",
        )
    )


def _impl(rows: list[dict], lang: str, audio_prefix: str, lora_ckpt: str) -> dict:
    import os, torch, librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    os.environ.setdefault("HF_HOME", "/cache/hf")
    base_model = VASISTA_MODEL[lang]
    lang_name = LANG_NAME[lang]

    print(f"[β-jsonl] loading {base_model}", flush=True)
    processor = WhisperProcessor.from_pretrained(base_model, language=lang_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    forced = processor.tokenizer.get_decoder_prompt_ids(language=lang_name, task="transcribe")
    model.config.forced_decoder_ids = forced
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.suppress_tokens = []

    print(f"[β-jsonl] loading LoRA from {lora_ckpt}", flush=True)
    model_lora = PeftModel.from_pretrained(model, lora_ckpt)
    model_lora.eval().to("cuda")

    out = []
    for i, r in enumerate(rows):
        ap = f"{audio_prefix}/{Path(r['audio_path']).name}"
        try:
            audio, _ = librosa.load(ap, sr=16_000, mono=True)
            feats = processor.feature_extractor(
                audio, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                pred_ids = model_lora.generate(feats, max_new_tokens=400, num_beams=1, do_sample=False)
            hyp = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[β-jsonl] WARN {ap}: {e}", flush=True); hyp = ""
        out.append({**{k: v for k, v in r.items() if not k.startswith("_")}, "beta_hyp": hyp})
        print(f"[β-jsonl] [{i+1}/{len(rows)}] {r.get('entity_class','?')} GT: {r['text'][:60]}", flush=True)
        print(f"[β-jsonl]      HY: {hyp[:60]}", flush=True)
    return {"n": len(out), "rows": out}


if _HAS_MODAL:
    @app.function(
        image=image, gpu="A10G", volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60 * 60 * 1,
    )
    def eval_beta_jsonl(rows: list[dict], lang: str, audio_prefix: str, lora_ckpt: str) -> dict:
        return _impl(rows, lang, audio_prefix, lora_ckpt)


def _score(rows: list[dict], lang: str) -> dict:
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None
    scores, sfrs, gts, hyps = [], [], [], []
    for r in rows:
        scores.append(score_row(r, r["beta_hyp"]))
        sfrs.append(script_fidelity_rate(r["beta_hyp"], lang))
        gts.append(r["text"]); hyps.append(r["beta_hyp"])
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
        print(f"[dry-run] would eval β-{args.lang} on {args.jsonl} via Modal A10G (~$0.30)")
        return 0
    if not _HAS_MODAL:
        return 2
    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    lora_ckpt = args.lora_ckpt or f"/cache/stt_rb/{args.lang}/rb/checkpoint-4000"
    print(f"[exec] launching β-{args.lang} on {args.jsonl} (n={len(rows)})")
    with app.run():
        result = eval_beta_jsonl.remote(rows=rows, lang=args.lang,
                                         audio_prefix=args.audio_prefix, lora_ckpt=lora_ckpt)
    out_rows = result["rows"]
    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.jsonl.stem
    pred = out_dir / f"beta_{args.lang}_{label}_n{len(out_rows)}_predictions.jsonl"
    pred.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))
    summary = _score(out_rows, args.lang)
    summary["holdout"] = label
    summary["lora_ckpt"] = lora_ckpt
    score = out_dir / f"beta_{args.lang}_{label}_n{len(out_rows)}_scorecard.json"
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
    r.add_argument("--audio-prefix", required=True)
    r.add_argument("--lora-ckpt", default=None)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
