"""Generic eval — Whisper-v3 + per-language LoRA on any holdout.

Replaces eval_te_fleurs / eval_te_cv25 / eval_te_jsonl_holdout for the Hi/Ta
LoRAs which followed Te-r2's recipe. Same code path; --lang and --holdout
parameterized.

CLI::
    uv run python -m paper.stt_flywheel.eval_lora run \\
        --lang ta --holdout fleurs --n 100 --execute
    uv run python -m paper.stt_flywheel.eval_lora run \\
        --lang ta --holdout cv25 --execute
    uv run python -m paper.stt_flywheel.eval_lora run \\
        --lang hi --holdout iv_general --execute
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
APP_NAME = "praxy-stt-eval-lora"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"
BASE_MODEL = "openai/whisper-large-v3"

LANG_NAME = {"te": "telugu", "ta": "tamil", "hi": "hindi"}
LANG_FLEURS = {"te": "te_in", "ta": "ta_in", "hi": "hi_in"}

if _HAS_MODAL:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1", "git")
        .pip_install(
            "torch==2.4.0", "torchaudio==2.4.0", "numpy<2",
            "transformers==4.49.0", "accelerate>=0.34", "peft>=0.13",
            "datasets>=2.20,<3.0", "soundfile", "librosa",
            "huggingface_hub>=0.25",
        )
    )


def _eval_impl(lang: str, holdout: str, lora_ckpt: str, n: int | None,
               jsonl_rows: list[dict] | None = None,
               audio_prefix: str | None = None) -> dict:
    import os, torch, librosa, sys as _sys
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

    os.environ.setdefault("HF_HOME", "/cache/hf")
    csv.field_size_limit(_sys.maxsize)
    lang_name = LANG_NAME[lang]

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=lang_name, task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to("cuda")
    base.eval()
    base.generation_config.language = lang_name
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    base.generation_config.suppress_tokens = []

    audios: list = []
    rows_meta: list[dict] = []

    if holdout == "fleurs":
        from datasets import load_dataset, Audio
        ds = load_dataset("google/fleurs", LANG_FLEURS[lang], split="test", trust_remote_code=True)
        if n is not None and n < len(ds):
            ds = ds.select(range(n))
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        for i, row in enumerate(ds):
            audios.append(row["audio"]["array"])
            rows_meta.append({
                "id": f"fleurs_{lang}_{row.get('id', i)}",
                "lang": lang, "entity_class": "general",
                "text": row["transcription"], "entity_tokens": [],
                "source": "fleurs",
            })

    elif holdout == "cv25":
        tsv = f"/cache/cv25/{lang}/test.tsv"
        clips = f"/cache/cv25/{lang}/clips"
        with open(tsv) as f:
            meta = [r for r in csv.DictReader(f, delimiter="\t") if r.get("path") and r.get("sentence")]
        if n is not None and n < len(meta):
            meta = meta[:n]
        for m in meta:
            try:
                a, _ = librosa.load(f"{clips}/{m['path']}", sr=16_000, mono=True)
            except Exception as e:
                print(f"[lora-eval] WARN load {m['path']}: {e}", flush=True)
                continue
            audios.append(a)
            rows_meta.append({
                "id": f"cv25_{lang}_{Path(m['path']).stem}",
                "lang": lang, "entity_class": "general",
                "text": m["sentence"], "entity_tokens": [],
                "source": "cv25",
            })

    elif holdout == "jsonl":
        # Caller provides rows + audio_prefix
        for r in jsonl_rows or []:
            ap = f"{audio_prefix}/{Path(r['audio_path']).name}"
            try:
                a, _ = librosa.load(ap, sr=16_000, mono=True)
            except Exception as e:
                print(f"[lora-eval] WARN {ap}: {e}", flush=True)
                continue
            audios.append(a)
            rows_meta.append({k: v for k, v in r.items() if not k.startswith("_")})

    print(f"[lora-eval] {len(rows_meta)} utterances loaded", flush=True)

    def _transcribe(model) -> list[str]:
        outs = []
        for i, (r, audio) in enumerate(zip(rows_meta, audios)):
            feats = processor.feature_extractor(
                audio, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                pred_ids = model.generate(
                    feats, max_new_tokens=400, num_beams=1, do_sample=False,
                    language=lang_name, task="transcribe",
                )
            text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
            outs.append(text)
            if (i + 1) <= 3 or (i + 1) % 25 == 0:
                print(f"[lora-eval] [{i+1}/{len(rows_meta)}] GT: {r['text'][:60]}", flush=True)
                print(f"[lora-eval]               HY: {text[:60]}", flush=True)
        return outs

    print("[lora-eval] vanilla v3 transcribing", flush=True)
    vanilla = _transcribe(base)
    print(f"[lora-eval] loading LoRA from {lora_ckpt}", flush=True)
    lora = PeftModel.from_pretrained(base, lora_ckpt)
    lora.eval()
    print(f"[lora-eval] {lang}-r2 transcribing", flush=True)
    lora_hyps = _transcribe(lora)

    out_rows = []
    for i, r in enumerate(rows_meta):
        out_rows.append({**r, "vanilla_hyp": vanilla[i], "lora_hyp": lora_hyps[i]})
    return {"n": len(out_rows), "rows": out_rows}


if _HAS_MODAL:
    @app.function(
        image=image, gpu="A10G", volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60 * 60 * 2,
    )
    def eval_lora(lang: str, holdout: str, lora_ckpt: str, n: int | None = None,
                  jsonl_rows: list[dict] | None = None, audio_prefix: str | None = None) -> dict:
        return _eval_impl(lang, holdout, lora_ckpt, n, jsonl_rows, audio_prefix)


def _score_local(rows: list[dict], lang: str) -> dict:
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None
    def _score(hyp_key):
        scores, sfrs, gts, hyps = [], [], [], []
        for r in rows:
            scores.append(score_row(r, r[hyp_key]))
            sfrs.append(script_fidelity_rate(r[hyp_key], lang))
            gts.append(r["text"]); hyps.append(r[hyp_key])
        agg = aggregate(scores)
        out = {"n": len(rows), "ehr": agg["ehr"],
               "sfr_mean": round(sum(sfrs)/len(sfrs),4) if sfrs else None}
        if jiwer:
            try:
                out["wer"] = round(jiwer.wer(gts, hyps), 4)
                out["cer"] = round(jiwer.cer(gts, hyps), 4)
            except Exception as e:
                out["wer_error"] = str(e)
        return out
    return {"vanilla_whisper_v3": _score("vanilla_hyp"),
            f"praxy_{lang}_r2": _score("lora_hyp")}


def _cmd_run(args):
    if not args.execute:
        print(f"[dry-run] would eval {args.lang}-r2 on {args.holdout} via Modal (~$0.55)")
        return 0
    if not _HAS_MODAL:
        return 2
    lora_ckpt = args.lora_ckpt or f"/cache/stt_r2/{args.lang}/r2/checkpoint-6000"

    jsonl_rows = None
    audio_prefix = None
    if args.holdout == "jsonl":
        if not args.jsonl or not args.audio_prefix:
            print("ERROR: jsonl holdout needs --jsonl + --audio-prefix", file=sys.stderr)
            return 2
        jsonl_rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
        audio_prefix = args.audio_prefix

    print(f"[exec] launching {args.lang}-r2 on {args.holdout} (n={args.n}, ckpt={lora_ckpt})")
    with app.run():
        result = eval_lora.remote(
            lang=args.lang, holdout=args.holdout, lora_ckpt=lora_ckpt,
            n=args.n, jsonl_rows=jsonl_rows, audio_prefix=audio_prefix,
        )
    rows = result["rows"]
    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.holdout if args.holdout != "jsonl" else args.jsonl.stem
    pred = out_dir / f"praxy_{args.lang}_r2_{label}_n{len(rows)}_predictions.jsonl"
    pred.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summary = _score_local(rows, args.lang)
    summary["holdout"] = label
    summary["lora_ckpt"] = lora_ckpt
    score = out_dir / f"praxy_{args.lang}_r2_{label}_n{len(rows)}_scorecard.json"
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
    r.add_argument("--holdout", required=True, choices=["fleurs", "cv25", "jsonl"])
    r.add_argument("--n", type=int, default=None)
    r.add_argument("--lora-ckpt", default=None)
    r.add_argument("--jsonl", type=Path, default=None)
    r.add_argument("--audio-prefix", default=None)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
