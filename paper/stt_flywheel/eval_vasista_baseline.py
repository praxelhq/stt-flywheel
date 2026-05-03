"""Eval vasista22/whisper-{te,ta,hi}-large-v2 on FLEURS + CV25 + IV holdouts.

Earlier attempts (per STATUS.md) used model.generate() directly with broken
forced_decoder_ids handling and got blank output. The model card explicitly
documents the *only* working recipe:

    pipe = pipeline(task="automatic-speech-recognition", model=...)
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
        language="te", task="transcribe")
    out = pipe(audio)["text"]

Apache-2.0. Trained on CSTD-IIIT-H + ULCA + Shrutilipi + MS-Indic + FLEURS-train +
Babel. Reports FLEURS-Te WER 9.65 on the model card.

CLI::
    uv run python -m paper.stt_flywheel.eval_vasista_baseline plan
    uv run python -m paper.stt_flywheel.eval_vasista_baseline run --lang te --holdout fleurs --execute
"""
from __future__ import annotations

import argparse
import csv
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
APP_NAME = "praxy-stt-vasista"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"

VASISTA_MODEL = {
    "te": "vasista22/whisper-telugu-large-v2",
    "ta": "vasista22/whisper-tamil-large-v2",
    "hi": "vasista22/whisper-hindi-large-v2",
}

CV25_DIR = "/cache/cv25"  # already uploaded for te + hi
IV_HOLDOUT_DIR = "/cache/iv_holdouts"  # already uploaded for te

if _HAS_MODAL:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    # vasista22's whisper-{te,ta,hi}-large-v2 was saved with an older
    # generation_config that breaks under transformers >=4.40. Pin to a
    # compatible version. This image is eval-only, separate from the LoRA
    # training image.
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .pip_install(
            "torch==2.4.0", "torchaudio==2.4.0", "numpy<2",
            "transformers==4.36.2", "accelerate>=0.30",
            "soundfile", "librosa", "huggingface_hub>=0.25,<0.30",
            "datasets>=2.20,<3.0",
        )
    )


def _eval_impl(lang: str, holdout: str, n: int | None = None) -> dict:
    """Runs inside Modal. Loads the right audio source, transcribes, returns rows."""
    import os, torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    os.environ.setdefault("HF_HOME", "/cache/hf")
    model_id = VASISTA_MODEL[lang]
    lang_name = {"te": "telugu", "ta": "tamil", "hi": "hindi"}[lang]
    print(f"[vasista] loading {model_id}", flush=True)
    processor = WhisperProcessor.from_pretrained(model_id, language=lang_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    # vasista22's generation_config is from older transformers; doesn't support
    # `language=` arg. Use explicit forced_decoder_ids (HF issue #25084).
    forced = processor.tokenizer.get_decoder_prompt_ids(language=lang_name, task="transcribe")
    model.config.forced_decoder_ids = forced
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.suppress_tokens = []
    print(f"[vasista] forced_decoder_ids={forced}", flush=True)

    def _transcribe(audio_arr) -> str:
        feats = processor.feature_extractor(
            audio_arr, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            pred_ids = model.generate(
                feats, max_new_tokens=400, num_beams=1, do_sample=False,
            )
        return processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

    rows: list[dict] = []

    if holdout == "fleurs":
        from datasets import load_dataset, Audio
        fleurs_lang = {"te": "te_in", "ta": "ta_in", "hi": "hi_in"}[lang]
        ds = load_dataset("google/fleurs", fleurs_lang, split="test", trust_remote_code=True)
        if n is not None and n < len(ds):
            ds = ds.select(range(n))
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        for i, row in enumerate(ds):
            audio = row["audio"]["array"]
            try:
                hyp = _transcribe(audio)
            except Exception as e:
                print(f"[vasista] WARN row {i}: {e}", flush=True)
                hyp = ""
            rows.append({
                "id": f"fleurs_{lang}_{row.get('id', i)}",
                "lang": lang, "entity_class": "general",
                "text": row["transcription"], "entity_tokens": [],
                "source": "fleurs", "vasista_hyp": hyp,
            })
            if (i + 1) <= 3 or (i + 1) % 25 == 0:
                print(f"[vasista] [{i+1}] GT: {row['transcription'][:60]}", flush=True)
                print(f"[vasista]      HY: {hyp[:60]}", flush=True)

    elif holdout == "cv25":
        import librosa
        import sys as _sys
        csv.field_size_limit(_sys.maxsize)  # Ta rows can exceed 128k default
        tsv = f"{CV25_DIR}/{lang}/test.tsv"
        clips = f"{CV25_DIR}/{lang}/clips"
        with open(tsv) as f:
            reader = csv.DictReader(f, delimiter="\t")
            meta = [r for r in reader if r.get("path") and r.get("sentence")]
        if n is not None and n < len(meta):
            meta = meta[:n]
        for i, m in enumerate(meta):
            try:
                audio, _ = librosa.load(f"{clips}/{m['path']}", sr=16_000, mono=True)
                hyp = _transcribe(audio)
            except Exception as e:
                print(f"[vasista] WARN row {i}: {e}", flush=True)
                hyp = ""
            rows.append({
                "id": f"cv25_{lang}_{Path(m['path']).stem}",
                "lang": lang, "entity_class": "general",
                "text": m["sentence"], "entity_tokens": [],
                "source": "cv25", "vasista_hyp": hyp,
            })
            if (i + 1) <= 3 or (i + 1) % 25 == 0:
                print(f"[vasista] [{i+1}] GT: {m['sentence'][:60]}", flush=True)
                print(f"[vasista]      HY: {hyp[:60]}", flush=True)

    elif holdout == "iv_general":
        import librosa
        ivdir = f"{IV_HOLDOUT_DIR}/{lang}"
        wavs = sorted(Path(ivdir).glob("*.wav"))
        # Read text from local-side JSONL via the rid match
        # Caller supplies metadata via JSONL; but for a self-contained run we
        # only need rids. We'll pair text post-hoc on the laptop.
        if n is not None:
            wavs = wavs[:n]
        for i, w in enumerate(wavs):
            try:
                audio, _ = librosa.load(str(w), sr=16_000, mono=True)
                hyp = _transcribe(audio)
            except Exception as e:
                print(f"[vasista] WARN row {i}: {e}", flush=True)
                hyp = ""
            rows.append({"id": w.stem, "lang": lang, "vasista_hyp": hyp})
            if (i + 1) <= 3 or (i + 1) % 25 == 0:
                print(f"[vasista] [{i+1}] {w.stem}", flush=True)
                print(f"[vasista]      HY: {hyp[:60]}", flush=True)
    else:
        raise ValueError(f"unknown holdout: {holdout}")

    return {"n": len(rows), "rows": rows}


if _HAS_MODAL:
    @app.function(
        image=image, gpu="A10G", volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60 * 60 * 2,
    )
    def eval_vasista(lang: str, holdout: str, n: int | None = None) -> dict:
        return _eval_impl(lang, holdout, n)


def _score_local(rows: list[dict], lang: str) -> dict:
    from paper.stt_flywheel.eval_ehr import script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None
    sfrs, gts, hyps = [], [], []
    for r in rows:
        if not r.get("text"):
            continue
        hyp = r["vasista_hyp"]
        sfrs.append(script_fidelity_rate(hyp, lang))
        gts.append(r["text"]); hyps.append(hyp)
    out = {"n": len(gts), "sfr_mean": round(sum(sfrs)/len(sfrs),4) if sfrs else None}
    if jiwer and gts:
        try:
            out["wer"] = round(jiwer.wer(gts, hyps), 4)
            out["cer"] = round(jiwer.cer(gts, hyps), 4)
        except Exception as e:
            out["wer_error"] = str(e)
    return out


def _attach_iv_text(rows: list[dict], lang: str) -> list[dict]:
    """Pair iv_general rows (which only have id from filename) with text from local JSONL."""
    jsonl = REPO_ROOT / "data" / "stt_flywheel" / "holdouts" / lang / "iv_general.jsonl"
    if not jsonl.exists():
        return rows
    text_map = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        text_map[r["id"]] = r["text"]
    for r in rows:
        if "text" not in r and r["id"] in text_map:
            r["text"] = text_map[r["id"]]
            r["entity_class"] = "general"
            r["entity_tokens"] = []
    return rows


def _cmd_run(args):
    if not args.execute:
        print(f"[dry-run] would run vasista22/whisper-{args.lang}-large-v2 on {args.holdout} via Modal A10G (~$0.55)")
        return 0
    if not _HAS_MODAL:
        return 2
    print(f"[exec] launching vasista22 eval (lang={args.lang}, holdout={args.holdout}, n={args.n})")
    with app.run():
        result = eval_vasista.remote(lang=args.lang, holdout=args.holdout, n=args.n)
    rows = result["rows"]
    if args.holdout == "iv_general":
        rows = _attach_iv_text(rows, args.lang)

    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"vasista_{args.lang}_{args.holdout}_n{len(rows)}_predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summary = _score_local(rows, args.lang)
    summary["holdout"] = f"{args.holdout}_{args.lang}"
    summary["model"] = VASISTA_MODEL[args.lang]
    score_path = out_dir / f"vasista_{args.lang}_{args.holdout}_n{len(rows)}_scorecard.json"
    score_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[exec] predictions → {pred_path}")
    print(f"[exec] scorecard → {score_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(func=lambda a: print("vasista eval; ~$0.55 per holdout × lang on Modal A10G"))
    r = sub.add_parser("run")
    r.add_argument("--lang", required=True, choices=["te", "ta", "hi"])
    r.add_argument("--holdout", required=True, choices=["fleurs", "cv25", "iv_general"])
    r.add_argument("--n", type=int, default=None)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)
    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
