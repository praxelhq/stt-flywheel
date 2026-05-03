"""β-paper eval — vasista22 + entity-LoRA on all 4 Te holdouts.

Loads vasista22/whisper-{lang}-large-v2 base + the rb LoRA at
/cache/stt_rb/{lang}/rb/checkpoint-NNNN, transcribes each of 4 holdouts:
  1. entity_dense_cartesia  — held-out (β eval target)
  2. fleurs                 — read-prose regression check
  3. cv25                   — read-prose regression check
  4. iv_general             — conversational regression check

CLI::
    uv run python -m paper.stt_flywheel.eval_beta run --lang te --execute
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

try:
    import modal  # type: ignore
    _HAS_MODAL = True
except Exception:
    modal = None  # type: ignore
    _HAS_MODAL = False

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "praxy-stt-eval-beta"
VOLUME_NAME = "praxy-voice-vol"
HF_SECRET = "praxy-hf"

VASISTA_MODEL = {
    "te": "vasista22/whisper-telugu-large-v2",
    "ta": "vasista22/whisper-tamil-large-v2",
    "hi": "vasista22/whisper-hindi-large-v2",
}
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
            "transformers==4.36.2", "accelerate==0.30.0", "peft==0.10.0",
            "datasets>=2.20,<3.0", "soundfile", "librosa",
            "huggingface_hub>=0.25,<0.30",
        )
    )


def _eval_impl(lang: str, lora_ckpt: str, entity_dense_rows: list[dict],
               entity_dense_audio_prefix: str, iv_rows: list[dict],
               iv_audio_prefix: str, n_fleurs: int = 100, n_cv25: int | None = None) -> dict:
    import os, torch, librosa, sys as _sys
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

    os.environ.setdefault("HF_HOME", "/cache/hf")
    csv.field_size_limit(_sys.maxsize)
    lang_name = LANG_NAME[lang]
    base_model = VASISTA_MODEL[lang]

    print(f"[β-eval] loading {base_model}", flush=True)
    processor = WhisperProcessor.from_pretrained(base_model, language=lang_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    forced = processor.tokenizer.get_decoder_prompt_ids(language=lang_name, task="transcribe")
    model.config.forced_decoder_ids = forced
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.suppress_tokens = []

    print(f"[β-eval] loading LoRA from {lora_ckpt}", flush=True)
    # The trainer saves LoRA-only state at lora_ckpt/lora_state.pt OR uses adapter_*.safetensors
    # Try PeftModel.from_pretrained first; fallback to manual load.
    try:
        model_lora = PeftModel.from_pretrained(model, lora_ckpt)
        model_lora.eval()
    except Exception as e:
        print(f"[β-eval] PeftModel.from_pretrained failed ({e}); falling back to lora_state.pt", flush=True)
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
        model_lora = get_peft_model(model, lora_cfg)
        sd = torch.load(f"{lora_ckpt}/lora_state.pt", map_location="cpu")
        model_lora.load_state_dict(sd, strict=False)
        model_lora.eval()
    model_lora.to("cuda")

    def _transcribe(model_in, audio):
        feats = processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            pred_ids = model_in.generate(feats, max_new_tokens=400, num_beams=1, do_sample=False)
        return processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

    results: dict = {}

    # 1. Entity-dense cartesia (β target)
    print("[β-eval] entity_dense_cartesia", flush=True)
    out = []
    for i, r in enumerate(entity_dense_rows):
        ap = f"{entity_dense_audio_prefix}/{Path(r['audio_path']).name}"
        try:
            audio, _ = librosa.load(ap, sr=16_000, mono=True)
            hyp = _transcribe(model_lora, audio)
        except Exception as e:
            print(f"[β-eval] WARN {ap}: {e}", flush=True); hyp = ""
        out.append({**{k: v for k, v in r.items() if not k.startswith("_")}, "beta_hyp": hyp})
        if (i + 1) <= 3 or (i + 1) % 25 == 0:
            print(f"[β-eval] entity[{i+1}] GT: {r['text'][:60]}", flush=True)
            print(f"[β-eval]                HY: {hyp[:60]}", flush=True)
    results["entity_dense_cartesia"] = out

    # 2. FLEURS
    print("[β-eval] fleurs", flush=True)
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", LANG_FLEURS[lang], split="test", trust_remote_code=True)
    if n_fleurs is not None and n_fleurs < len(ds):
        ds = ds.select(range(n_fleurs))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    out = []
    for i, row in enumerate(ds):
        try:
            hyp = _transcribe(model_lora, row["audio"]["array"])
        except Exception as e:
            print(f"[β-eval] WARN fleurs row {i}: {e}", flush=True); hyp = ""
        out.append({
            "id": f"fleurs_{lang}_{row.get('id', i)}", "lang": lang,
            "entity_class": "general", "text": row["transcription"],
            "entity_tokens": [], "source": "fleurs", "beta_hyp": hyp,
        })
        if (i + 1) <= 3 or (i + 1) % 25 == 0:
            print(f"[β-eval] fleurs[{i+1}] GT: {row['transcription'][:60]}", flush=True)
            print(f"[β-eval]                HY: {hyp[:60]}", flush=True)
    results["fleurs"] = out

    # 3. CV25
    print("[β-eval] cv25", flush=True)
    tsv = f"/cache/cv25/{lang}/test.tsv"
    clips = f"/cache/cv25/{lang}/clips"
    with open(tsv, encoding="utf-8") as f:
        meta = [r for r in csv.DictReader(f, delimiter="\t") if r.get("path") and r.get("sentence")]
    if n_cv25 is not None and n_cv25 < len(meta):
        meta = meta[:n_cv25]
    out = []
    for i, m in enumerate(meta):
        try:
            audio, _ = librosa.load(f"{clips}/{m['path']}", sr=16_000, mono=True)
            hyp = _transcribe(model_lora, audio)
        except Exception as e:
            print(f"[β-eval] WARN cv25 row {i}: {e}", flush=True); hyp = ""
        out.append({
            "id": f"cv25_{lang}_{Path(m['path']).stem}", "lang": lang,
            "entity_class": "general", "text": m["sentence"],
            "entity_tokens": [], "source": "cv25", "beta_hyp": hyp,
        })
        if (i + 1) <= 3 or (i + 1) % 25 == 0:
            print(f"[β-eval] cv25[{i+1}] GT: {m['sentence'][:60]}", flush=True)
            print(f"[β-eval]              HY: {hyp[:60]}", flush=True)
    results["cv25"] = out

    # 4. IV-general
    print("[β-eval] iv_general", flush=True)
    out = []
    for i, r in enumerate(iv_rows):
        ap = f"{iv_audio_prefix}/{Path(r['audio_path']).name}"
        try:
            audio, _ = librosa.load(ap, sr=16_000, mono=True)
            hyp = _transcribe(model_lora, audio)
        except Exception as e:
            print(f"[β-eval] WARN iv {ap}: {e}", flush=True); hyp = ""
        out.append({**{k: v for k, v in r.items() if not k.startswith("_")}, "beta_hyp": hyp})
        if (i + 1) <= 3 or (i + 1) % 25 == 0:
            print(f"[β-eval] iv[{i+1}] GT: {r['text'][:60]}", flush=True)
            print(f"[β-eval]            HY: {hyp[:60]}", flush=True)
    results["iv_general"] = out

    return results


if _HAS_MODAL:
    @app.function(
        image=image, gpu="A10G", volumes={"/cache": volume},
        secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60 * 60 * 3,
    )
    def eval_beta_all(lang: str, lora_ckpt: str, entity_dense_rows: list[dict],
                      entity_dense_audio_prefix: str, iv_rows: list[dict],
                      iv_audio_prefix: str, n_fleurs: int = 100, n_cv25: int | None = None) -> dict:
        return _eval_impl(lang, lora_ckpt, entity_dense_rows, entity_dense_audio_prefix,
                          iv_rows, iv_audio_prefix, n_fleurs, n_cv25)


def _score(rows: list[dict], lang: str, hyp_key: str = "beta_hyp") -> dict:
    from paper.stt_flywheel.eval_ehr import score_row, aggregate, script_fidelity_rate
    try:
        import jiwer
    except Exception:
        jiwer = None
    scores, sfrs, gts, hyps = [], [], [], []
    for r in rows:
        scores.append(score_row(r, r[hyp_key]))
        sfrs.append(script_fidelity_rate(r[hyp_key], lang))
        gts.append(r["text"]); hyps.append(r[hyp_key])
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
        print(f"[dry-run] would eval β-{args.lang} on 4 holdouts via Modal A10G (~$1.50)")
        return 0
    if not _HAS_MODAL:
        return 2

    # Load entity-dense holdout
    ed_path = REPO_ROOT / "data" / "stt_flywheel" / "holdouts" / args.lang / "entity_dense_cartesia.jsonl"
    iv_path = REPO_ROOT / "data" / "stt_flywheel" / "holdouts" / args.lang / "iv_general.jsonl"
    ed_rows = [json.loads(l) for l in ed_path.read_text().splitlines() if l.strip()] if ed_path.exists() else []
    iv_rows = [json.loads(l) for l in iv_path.read_text().splitlines() if l.strip()] if iv_path.exists() else []

    lora_ckpt = args.lora_ckpt or f"/cache/stt_rb/{args.lang}/rb/checkpoint-4000"
    print(f"[exec] launching β-{args.lang} eval (lora={lora_ckpt}, ed={len(ed_rows)}, iv={len(iv_rows)})")
    with app.run():
        result = eval_beta_all.remote(
            lang=args.lang, lora_ckpt=lora_ckpt,
            entity_dense_rows=ed_rows,
            entity_dense_audio_prefix=f"/cache/stt_r2/audio/{args.lang}/cartesia",
            iv_rows=iv_rows,
            iv_audio_prefix=f"/cache/iv_holdouts/{args.lang}",
            n_fleurs=args.n_fleurs, n_cv25=args.n_cv25,
        )

    out_dir = REPO_ROOT / "evaluation" / "scorecards" / "stt_flywheel"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"lang": args.lang, "lora_ckpt": lora_ckpt, "by_holdout": {}}
    for holdout, rows in result.items():
        pred = out_dir / f"beta_{args.lang}_{holdout}_n{len(rows)}_predictions.jsonl"
        pred.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        s = _score(rows, args.lang)
        summary["by_holdout"][holdout] = s
        print(f"\n=== β-{args.lang} on {holdout} (n={len(rows)}) ===")
        print(json.dumps(s, indent=2, ensure_ascii=False))

    score = out_dir / f"beta_{args.lang}_full_scorecard.json"
    score.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[exec] full scorecard → {score}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lang", required=True, choices=["te", "ta", "hi"])
    r.add_argument("--lora-ckpt", default=None)
    r.add_argument("--n-fleurs", type=int, default=100)
    r.add_argument("--n-cv25", type=int, default=None)
    r.add_argument("--execute", action="store_true")
    r.set_defaults(func=_cmd_run)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
