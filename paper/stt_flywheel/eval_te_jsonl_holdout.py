"""Generic eval: Te-r2 LoRA + vanilla Whisper-v3 on a JSONL holdout where
audio lives at /cache/iv_holdouts/te/<id>.wav inside Modal.

Useful for the IV-general / IV-held-back / any-extracted-from-HF holdouts.

Reads the holdout JSONL locally to know id+text+audio_filename mapping;
audio bytes load inside the container from the volume.

CLI::
    uv run python -m paper.stt_flywheel.eval_te_jsonl_holdout run \\
        --jsonl data/stt_flywheel/holdouts/te/iv_general.jsonl \\
        --audio-prefix /cache/iv_holdouts/te \\
        --execute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import modal  # type: ignore
    _HAS_MODAL = True
except Exception:
    modal = None  # type: ignore
    _HAS_MODAL = False

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "praxy-stt-eval-jsonl"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"
LORA_CKPT = "/cache/stt_r2/te/r2/checkpoint-6000"
BASE_MODEL = "openai/whisper-large-v3"

if _HAS_MODAL:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .pip_install(
            "torch==2.4.0", "torchaudio==2.4.0", "numpy<2",
            "transformers==4.49.0", "accelerate>=0.34", "peft>=0.13",
            "soundfile", "librosa", "huggingface_hub>=0.25",
        )
    )


def _eval_impl(rows: list[dict], audio_prefix: str, lora_ckpt: str, num_beams: int = 1) -> dict:
    import os, torch, librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel
    os.environ.setdefault("HF_HOME", "/cache/hf")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="telugu", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to("cuda")
    base.eval()
    base.generation_config.language = "telugu"
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    base.generation_config.suppress_tokens = []

    audios = []
    for r in rows:
        path = f"{audio_prefix}/{r['_audio_basename']}"
        try:
            audio, _ = librosa.load(path, sr=16_000, mono=True)
            audios.append(audio)
        except Exception as e:
            print(f"WARN: load fail {path}: {e}", flush=True)
            audios.append(None)

    def _transcribe(model) -> list[str]:
        outs = []
        for i, (r, audio) in enumerate(zip(rows, audios)):
            if audio is None:
                outs.append("")
                continue
            feats = processor.feature_extractor(
                audio, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                pred_ids = model.generate(
                    feats, max_new_tokens=400, num_beams=num_beams, do_sample=False,
                    language="telugu", task="transcribe",
                )
            text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
            outs.append(text)
            if (i + 1) <= 3 or (i + 1) % 25 == 0:
                print(f"[eval] [{i+1}/{len(rows)}] GT: {r['text'][:60]}", flush=True)
                print(f"[eval]                 HY: {text[:60]}", flush=True)
        return outs

    print("[eval] vanilla v3 transcribing", flush=True)
    vanilla = _transcribe(base)
    print(f"[eval] loading LoRA from {lora_ckpt}", flush=True)
    lora = PeftModel.from_pretrained(base, lora_ckpt)
    lora.eval()
    print("[eval] Te-r2 transcribing", flush=True)
    lora_hyps = _transcribe(lora)

    out = []
    for i, r in enumerate(rows):
        out.append({
            **{k: v for k, v in r.items() if not k.startswith("_")},
            "vanilla_hyp": vanilla[i],
            "lora_hyp": lora_hyps[i],
        })
    return {"n": len(out), "rows": out}


if _HAS_MODAL:
    @app.function(
        image=image, gpu="A10G", volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60 * 60 * 2,
    )
    def eval_jsonl(rows: list[dict], audio_prefix: str, lora_ckpt: str = LORA_CKPT, num_beams: int = 1) -> dict:
        return _eval_impl(rows, audio_prefix, lora_ckpt, num_beams)


def _score_local(rows: list[dict], lang: str = "te") -> dict:
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None
    def _score(hyp_key):
        scores, sfrs, gts, hyps = [], [], [], []
        for r in rows:
            hyp = r[hyp_key]
            scores.append(score_row(r, hyp))
            sfrs.append(script_fidelity_rate(hyp, lang))
            gts.append(r["text"]); hyps.append(hyp)
        agg = aggregate(scores)
        out = {"n": len(rows), "ehr": agg["ehr"], "sfr_mean": round(sum(sfrs)/len(sfrs),4) if sfrs else None}
        if jiwer:
            try:
                out["wer"] = round(jiwer.wer(gts, hyps), 4)
                out["cer"] = round(jiwer.cer(gts, hyps), 4)
            except Exception as e:
                out["wer_error"] = str(e)
        return out
    return {"vanilla_whisper_v3": _score("vanilla_hyp"), "praxy_te_r2": _score("lora_hyp")}


def _cmd_run(args):
    if not args.execute:
        print(f"[dry-run] would eval {args.jsonl} via Modal A10G (~$1)")
        return 0
    if not _HAS_MODAL:
        return 2

    raw_rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    rows = []
    for r in raw_rows:
        rows.append({
            "id": r["id"],
            "lang": r.get("lang", "te"),
            "entity_class": r.get("entity_class", "general"),
            "text": r["text"],
            "entity_tokens": r.get("entity_tokens", []),
            "source": r.get("source", "iv"),
            "_audio_basename": Path(r["audio_path"]).name,
        })

    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_label = args.jsonl.stem
    print(f"[exec] launching Modal eval (n={len(rows)}, holdout={holdout_label}, beams={args.num_beams})")
    with app.run():
        result = eval_jsonl.remote(rows=rows, audio_prefix=args.audio_prefix,
                                   lora_ckpt=args.lora_ckpt, num_beams=args.num_beams)
    out_rows = result["rows"]
    suffix = f"_b{args.num_beams}" if args.num_beams > 1 else ""
    pred_path = out_dir / f"praxy_te_r2_{holdout_label}_n{len(out_rows)}{suffix}_predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))
    summary = _score_local(out_rows, lang="te")
    summary["n_rows"] = len(out_rows)
    summary["holdout"] = holdout_label
    summary["num_beams"] = args.num_beams
    score_path = out_dir / f"praxy_te_r2_{holdout_label}_n{len(out_rows)}{suffix}_scorecard.json"
    score_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[exec] predictions → {pred_path}")
    print(f"[exec] scorecard → {score_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jsonl", type=Path, required=True)
    r.add_argument("--audio-prefix", required=True,
                   help="Modal volume path prefix where audio basenames resolve, e.g. /cache/iv_holdouts/te")
    r.add_argument("--lora-ckpt", default=LORA_CKPT)
    r.add_argument("--num-beams", type=int, default=1)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
