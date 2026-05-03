# STT Flywheel — Runbook

Step-by-step "what to run when, in what order, with what flag" for the STT flywheel paper. Sections track Phase 1 → 2a → 2b → 3 → 4 in execution order. Resume-friendly: every step skips work it can detect already done.

> **ALWAYS run `/ml-preflight paper/stt_flywheel/PLAN.md` before flipping any `--execute` flag.** See `.claude/skills/ml-preflight/SKILL.md`.

## Phase 1 — Entity-dense text generation (Anthropic Haiku 4.5, ~$15)

**Done as of 2026-04-30.** 22,193 cleaned rows across te/ta/hi × 6 entity classes.

```bash
# Plan-only (free): print synth targets + cost estimate
uv run python -m paper.stt_flywheel.data_pipeline build \
    --plan paper/stt_flywheel/plan.yaml

# Execute (only if greenlit): generates ~27k raw rows → ~22k after cleanup
uv run python -m paper.stt_flywheel.data_pipeline build \
    --plan paper/stt_flywheel/plan.yaml --execute

# Single (lang, class) regeneration (e.g., after expanding entity dict)
uv run python -m paper.stt_flywheel.data_pipeline text \
    --lang ta --class proper_nouns --n 1500 --execute

# Audit + clean
uv run python paper/stt_flywheel/audit_corpus.py
uv run python paper/stt_flywheel/clean_corpus.py
```

**Skip-≥70% threshold** in `build_synth_corpus` ensures pairs already at ≥70% of target are not regenerated. Drops the truly bad rows via `clean_corpus`.

## Phase 1.5 — Tamil low-yield redo (incremental, ~$1.50)

If Tamil pairs underperform (proper_nouns hit 50/1500 = 3.3% on the first run), expand the entity dict and re-run only the low-yield pairs:

```bash
# 1. Expand stt/data/entities/{class}/ta.jsonl with broader seed entities
# 2. Move underperformer manifests aside
mv data/stt_flywheel/text/ta_proper_nouns.jsonl data/stt_flywheel/text/ta_proper_nouns.jsonl.lowyield
mv data/stt_flywheel/text/ta_addresses.jsonl    data/stt_flywheel/text/ta_addresses.jsonl.lowyield
mv data/stt_flywheel/text/ta_brands.jsonl       data/stt_flywheel/text/ta_brands.jsonl.lowyield

# 3. Re-run (orchestrator skips the 15 other pairs via skip-≥70%)
rm -rf data/stt_flywheel/_cache
uv run python -m paper.stt_flywheel.data_pipeline build \
    --plan paper/stt_flywheel/plan.yaml --execute
```

## Phase 2a — Free-credit synth (ElevenLabs + Cartesia, $0)

```bash
# Plan
uv run python -m paper.stt_flywheel.phase2_synth plan

# Canary (5 clips, ~$0)
uv run python -m paper.stt_flywheel.phase2_synth shard \
    --lang te --system elevenlabs --max-rows 5 --execute

# Full free buckets (skip Praxy)
uv run python -m paper.stt_flywheel.phase2_synth all --skip-praxy --execute

# Background-friendly: redirect stdout, capture PID
uv run python -m paper.stt_flywheel.phase2_synth all --skip-praxy --execute \
    > data/stt_flywheel/logs/phase2_free.log 2>&1 &
```

Resume-friendly: existing WAVs at `data/stt_flywheel/audio/{lang}/{system}/{id}.wav` are skipped.

## Phase 2b — Modal Praxy synth (~$15)

Awaiting greenlight. Routes per `serving/praxy_router.py`:
- `praxy_lora`     — te/ta non-codemix → R6 LoRA + Config B
- `praxy_vanilla`  — hi non-codemix → vanilla Chatterbox + Config B
- `praxy_indicf5`  — any codemix → translit→IndicF5

```bash
# Build per-shard test_set JSONs (free, file-shuffling only)
uv run python -m paper.stt_flywheel.phase2b_praxy_modal build-test-sets

# Plan: print modal CLI commands + cost
uv run python -m paper.stt_flywheel.phase2b_praxy_modal plan

# Canary one shard (~$0.65–3.00)
uv run python -m paper.stt_flywheel.phase2b_praxy_modal shard \
    --lang te --branch praxy_lora --max-rows 50 --execute

# Full Phase 2b
uv run python -m paper.stt_flywheel.phase2b_praxy_modal all --execute

# Rebuild unified manifest after harvest
uv run python -m paper.stt_flywheel.phase2b_praxy_modal manifest
```

## Phase 3 — Whisper-large-v3 LoRA training per language (~$45-55)

```bash
# Build training manifests (joins text + audio + duration)
uv run python -m paper.stt_flywheel.build_training_manifest build

# Validate manifest (audio files exist, durations sane, fields complete)
uv run python -m paper.stt_flywheel.build_training_manifest validate \
    --manifest data/stt_flywheel/manifests/te_train.jsonl

# Smoke test on Modal (10 utts, 50 steps, ~$0.10)
uv run modal run paper/stt_flywheel/modal_stt_train.py::train_lora \
    --lang te --max-rows 10 --max-steps 50

# Full per-lang training (≈12 A10G-hr/lang × 3 langs)
uv run modal run paper/stt_flywheel/modal_stt_train.py::train_lora \
    --lang te --max-steps 6000
uv run modal run paper/stt_flywheel/modal_stt_train.py::train_lora \
    --lang ta --max-steps 6000
uv run modal run paper/stt_flywheel/modal_stt_train.py::train_lora \
    --lang hi --max-steps 6000
```

**Divergence-abort** is built into the trainer: if eval-WER goes up 2 consecutive 500-step checkpoints, training auto-aborts. Per-lang LoRAs (no Hi-proxy — explicit fix to R1 failure mode).

## Phase 4 — Eval + paper draft (~$5-10)

```bash
# 1. Build eval holdout sets (300 entity-dense + 100 FLEURS regression + 100 codemix per lang)
uv run python -m paper.stt_flywheel.eval_set_builder build \
    --lang te --execute

# 2. Run baselines (vanilla Whisper + IndicConformer + commercial)
uv run modal run paper/stt_flywheel/modal_stt_train.py::eval_checkpoint \
    --lang te --ckpt vanilla --eval-set entity_dense_holdout

# 3. Score Entity-Hit-Rate (precise per-class normalisation per PLAN §7)
uv run python -m paper.stt_flywheel.eval_ehr score \
    --gt data/stt_flywheel/holdouts/te_entity_dense.jsonl \
    --hyp data/stt_flywheel/eval_outputs/te_vanilla_whisper.jsonl

# 4. Compare vs baselines, write paper section §V
```

## Healthchecks

| Check | Command | Expected |
|---|---|---|
| Anthropic spend | `cat ~/.praxy/anthropic_spend.json` | `total_usd` < $100 |
| OpenRouter spend | `cat ~/.praxy/openrouter_spend.json` | `total_usd` < $3 |
| Modal billing | https://modal.com/billing | Alert if >$15/night |
| Phase 1 corpus integrity | `uv run python paper/stt_flywheel/audit_corpus.py` | 0 critical issues per pair |
| Tests | `uv run python -m pytest paper/stt_flywheel/ -q` | All passing |
| Background jobs | `ps aux \| grep paper.stt` | one active python per running phase |

## Abort triggers

Per PLAN.md §9 and `feedback_cost_discipline.md`:

1. **Cost cap $130** — STOP all paid work; reassess plan.
2. **Divergence-abort during training** — auto-built into trainer; manifests remain on disk.
3. **All 3 langs <2pp EHR improvement after R1** — abort, reassess data mix.
4. **Real-corpus blocked for 1 lang** — ship 2-language paper (Te+Hi), document Ta as future work.
5. **Network blip in middle of paid run** — exponential-backoff retry built in (6 attempts before giving up on a pair); spend ledger preserved.

## What NOT to do

- ❌ Skip `/ml-preflight` before `--execute` — burns credits on bad routing.
- ❌ Wipe `data/stt_flywheel/_cache/` mid-run — orphans the in-progress (lang, class) pair.
- ❌ Run two `data_pipeline build` processes simultaneously — they race on the cache.
- ❌ Edit `data/stt_flywheel/text/*.jsonl` files by hand — use `clean_corpus.py` so renumbering stays consistent.
- ❌ Trust `wave.open(p).getnframes()` for Cartesia outputs — header is broken (returns INT32_MAX); use `_wav_duration()` from `build_training_manifest.py`.
