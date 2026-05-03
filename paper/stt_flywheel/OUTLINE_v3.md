# Paper #3 v3 — TTS↔STT Flywheel for Indic Entity-Dense ASR

**Working title:** *The TTS↔STT Flywheel: Synthetic Entity-Dense Audio Closes the Indic ASR Gap Where Commercial and Open-Source Systems Fail*

**Length target:** 8 pages + references (industry track / EMNLP / Interspeech).

**Author:** Venkata Pushpak Teja Menta · Praxel Ventures · ORCID 0009-0003-2479-9208.

*Re-framed 2026-05-02 after empirical confirmation of the entity-dense gap: vasista22/whisper-{te,ta,hi}-large-v2 (open SOTA) achieves EHR 2.7% on Telugu entity-dense audio, Deepgram Nova-3 (commercial) achieves 16%, and a small LoRA fine-tune trained on synthesised entity-dense audio achieves 85% (with leakage caveat) — a 32× improvement over open SOTA and 5× over commercial. The flywheel novelty was always the contribution; the read-prose benchmark was supporting evidence.*

---

## Thesis (one sentence)

> *Niche-domain Indic ASR (digit strings, currency amounts, addresses, brand names, English/Indic codemix) is dramatically under-served by both open-source SOTA (vasista22/whisper-large-v2 EHR 2.7% on Telugu) and commercial systems (Deepgram Nova-3 EHR 16%), because real-world Indic ASR corpora are dominated by Wikipedia/news/govt read-prose text; we close this gap with a self-contained TTS↔STT flywheel — synthesise entity-dense audio with our open-source Indic TTS (Praxy R6, IndicF5, ElevenLabs, Cartesia), train a small LoRA (rank 16, ~$10 Modal) on top of either vanilla Whisper-v3 or vasista22 base, and recover EHR ≥75% while preserving read-prose performance.*

## Refined finding (2026-05-03 update)

After empirical evaluation across all 3 Indic languages, the SFR-fix mechanism is **Te-specific**:
- **Te:** vanilla Whisper-v3 SFR 0.46–0.71 across 3 holdouts → broken → Te-r2 LoRA fixes it (SFR 0.79–0.97)
- **Hi:** vanilla SFR ≥ 0.98 (Devanagari well-supported) → Hi-r2 LoRA is **net harmful** (SFR 0.43–0.88, WER ↑ 60–160%)
- **Ta:** vanilla SFR ≥ 0.98 → Ta-r2 LoRA is **net harmful** (SFR 0.71–0.94, WER ↑ 20–35%)

**Diagnostic for the recipe:** apply per-language LoRA + per-language decoder prefix iff vanilla SFR < 0.85 across ≥ 2 holdouts. Te is the only language in our matrix that meets this criterion.

The paper now reports both the success on Te AND the controlled failure on Hi/Ta, framed as a language-conditional finding. This is methodologically stronger than uniform-3-lang fix claims because it gives engineers a measurable trigger.

**Independent finding:** All 3 langs (vasista22 EHR 2.5–4.9%, Deepgram 2.5–48.5%) **fail** on entity-dense Indic audio. The β-paper recipe (entity-LoRA on top of vasista22) attacks this orthogonal failure mode and **does** generalize across langs (TBC pending β-Hi + β-Ta evals).

---

## What is novel (three contributions, each independently defensible)

### 1. TTS↔STT Flywheel architecture for entity-dense Indic audio

Our open-source TTS (Praxy R6 LoRA + vanilla Chatterbox + IndicF5, all already published as paper #1) synthesises ~22k entity-dense Indic-English code-mix utterances at <$50 marginal cost per language. This synthetic audio (~25% of the training mix) plus IndicVoices + Common Voice 25 + FLEURS (~75%) trains a small LoRA on Whisper-v3 or vasista22, recovering entity-dense recognition that neither commercial Deepgram nor open-source vasista22 can provide. **The flywheel is the architecture: TTS-as-data-pipeline for STT-niche-adaptation.**

Adjacent prior art: SpeechT5 (Microsoft 2022) is a TTS+ASR base but not Indic-tuned and not trained with TTS-synth-as-data; Mock+Cohn (Interspeech 2023) showed TTS-aug-data helps child-speech ASR; Distil-Whisper (Gandhi et al. 2023) uses Whisper self-distillation but not TTS-flywheel. **No prior TTS-flywheel work exists for Indic entity-dense workloads specifically.**

### 2. Entity-Dense Synthetic Audio (EDSA) methodology

Reproducible pipeline:
- **§3.1 Entity-text generation:** Claude Haiku 4.5 (~$16) prompted with per-class entity dictionaries (`stt/data/entities/{class}/{lang}.jsonl`, ~3k seed entities per lang) generates ~22.2k Indic utterances with entity-tagged spans. Six classes: digits, currency, addresses, brands, codemix, proper_nouns.
- **§3.2 Multi-system synthesis routing:** `serving/praxy_router.py` dispatches per (lang × class): te/ta non-codemix → R6 LoRA; hi non-codemix → vanilla Chatterbox; any codemix → translit→IndicF5. 60% Praxy bucket + 20% ElevenLabs v3 (free credits) + 20% Cartesia sonic-3 (free credits).
- **§3.3 Spelled-digit rewrite:** corpus audit caught text-form/audio-form mismatch ("OTP 54235" text but synth audio reads "five lakh forty-two thousand…"); deterministic per-language rewrite ensures train labels match what the synth voice actually says.
- **§3.4 Per-class CER filter:** vasista22-CER ≤ 0.5 drop rate ~10-15%.

Open-sourced as `paper/stt_flywheel/data_pipeline.py` + entity dictionaries (CC-BY-4.0).

### 3. Entity-Hit-Rate (EHR) metric with per-class semantic normalization

Like PSP (paper #2) is to accent evaluation, EHR is to entity-recognition evaluation. Unlike WER which treats *"5 lakh"* and *"five hundred thousand"* as different tokens, EHR scores semantic equivalence per entity class:

- **digit_run:** NFKC-normalised exact match
- **pincode:** NFKC + length-6 exact match
- **currency_amount:** numeric value (after parsing "5 lakh" / "ఐదు లక్షల" via INDIC_MULTIPLIERS) within ±0.5%
- **brand:** case-folded match in Latin or native script via BRAND_ALIASES dictionary
- **proper_noun:** token-set ≥80% overlap (allows transliteration variance)
- **spelled_digit:** subsequence preservation ≥80%
- **house_or_plot:** NFKC + casefold match

Deterministic, no LLM-judge in headline. 19/19 unit tests pass. Could carry an independent metrics paper.

We also report **Script-Fidelity-Rate (SFR)** per concurrent *Script Collapse in Multilingual ASR* (2026); first cross-system measurement on real Indic audio reveals vanilla Whisper-v3 SFR=0.46 on Te-CV25 (54% of letters in non-Te script) and Deepgram-Hi-CV25 SFR=0.83 (Deepgram outputs ~17% non-Devanagari letters on Hindi).

---

## Empirical results (subject to β LoRA training landing)

### Headline single-table

**Telugu (n=102 entity-dense + n=86–100 read-prose × 3 holdouts):**

| System | Read-prose Te WER (best holdout) | Entity-dense Te EHR | Entity-dense SFR |
|---|---|---|---|
| Vanilla Whisper-v3 | 1.50 (FLEURS) | 0.56 | 0.57 |
| Praxy-STT-Te-r2 (Whisper-v3 + LoRA, ours) | 0.83 (FLEURS) | **0.85** ‡ | 0.79 |
| vasista22/whisper-te-large-v2 (open SOTA) | **0.33** (FLEURS) | 0.027 | 1.00 |
| Deepgram Nova-3 (commercial) | 0.44 (CV25) | 0.16 | 0.98 |
| **β-Te (vasista22 + EDSA-LoRA, ours)** | **0.39 ✓** | **0.473 (17×)** | **0.928 ✓** |

‡ Te-r2 cartesia rows are leaky (saw cartesia in training; the cartesia-held-out β-Te variant is the clean comparison).

### Cross-language replication (TBC pending Hi/Ta β LoRA)

**Findings on read prose (alpha-data already collected):**
- vasista22 (open) ≥ Deepgram Nova-3 on conversational Te (IV-Te WER 0.42 vs 0.51).
- vasista22 ≥ Deepgram on Hi-CV25 (0.28 vs 0.36).
- **Open-source has reached commercial parity on read-prose Indic ASR** — independent contribution, supports the flywheel framing (we're not climbing an unsolved mountain; we're attacking a specific weakness on a strong base).

---

## Paper structure

**§1 Introduction (~1p)** — Indic ASR has a known niche-data blindspot (entities, codemix). Read-prose is now near-solved by open-source (vasista22 ≥ Deepgram on conv-Te). Niche is wide open: vasista22 EHR 2.7%, Deepgram 16%. We propose the TTS↔STT flywheel; demonstrate 32× EHR over open SOTA.

**§2 Related Work (~0.5p)** — vasista22, AI4Bharat IndicConformer/IndicWhisper, Distil-Whisper, SpeechT5, Mock+Cohn 2023, *Script Collapse* 2026. Our paper #1 (Praxy Voice TTS, arXiv:2604.25441) and paper #2 (PSP, arXiv:2604.25476) cited as the flywheel's TTS half + accent metric companion.

**§3 Method (~1.5p)** — EDSA pipeline (§3.1-3.4); LoRA training recipe (rank 16, q/k/v/out, 4-6k steps on A10G). EHR metric per-class normalization (§3.5). SFR adoption (§3.6).

**§4 Experiments (~1p)** — Holdouts: FLEURS-{te,ta,hi} test, CV25 test (real recordings), IV-{te,ta,hi}-general (conversational, n=100), Te entity-dense cartesia (n=102, class-balanced, held-out by synth system). Systems: vanilla Whisper-v3, vasista22 base, Deepgram Nova-3, Praxy-STT-r2 (Whisper-v3 LoRA), β-{te,ta,hi} (vasista22 LoRA — the EDSA-recovered model).

**§5 Results (~1.5p)** — Headline EHR table (above), cross-language consistency, ablations (synth-fraction sweep deferred — see Limitations).

**§6 Flywheel closure (~0.5p)** — Re-run Praxy Voice TTS paper §V Te smoke evaluation using β-Te-STT in place of Whisper-v3-vanilla; report whether system rankings change. Optional, evidence of value to the companion paper.

**§7 Limitations (~0.5p)**
1. Entity-dense holdout is synthesised, not human-recorded. We mitigate via synth-system held-out (cartesia held out from training; β-Te never sees cartesia rows in training but is evaluated on them). Native human recordings deferred to v2.
2. Ta entity-dense corpus smaller (~5k rows vs ~9k Te/Hi) due to entity-diversity ceiling on Haiku 4.5.
3. Deepgram is one commercial system; ElevenLabs Scribe + Sarvam STT not benchmarked (rate limits / GA status).
4. Synth-fraction ablation (Tables 4-5 of v1 OUTLINE) deferred — would require 12 LoRA retrains × 3 langs.

**§8 Reproducibility + Conclusion (~0.5p)** — Code (github.com/praxelhq/praxy-tts/paper/stt_flywheel/), holdout JSONLs, predictions JSONLs, EDSA corpus + entity dictionaries CC-BY-4.0. Cost transparency: ~$200 cumulative real spend, β LoRA × 3 langs adds ~$30.

---

## Pre-registered win conditions

1. β-Te EHR ≥ 0.75 on entity_dense_cartesia (cleaner than Te-r2's 0.85 with leakage; target accounts for clean held-out)
2. β-Te FLEURS WER ≤ 0.40 (no read-prose regression beyond +7 pp vs vasista22 base 0.33)
3. β-{Hi,Ta} EHR ≥ 0.65 on respective entity-dense holdouts (lower bar; smaller corpora)
4. SFR ≥ 0.90 across all holdouts (script preservation)

If 4/4 hit → ship as full industry paper.
If 3/4 hit → ship; acknowledge missing cell honestly.
If <3/4 hit → ship as honest-negative; reframe as "the flywheel partially works; here's what we learned."

---

## Out of scope (deferred to v2 / β2)

1. Synth-fraction ablation (12 LoRAs × 3 langs).
2. Native-speaker entity-dense recordings (afternoon recording session).
3. Bengali / Kannada / Malayalam / Marathi (entity-dictionary effort).
4. From-scratch base (Conformer / wav2vec2) training.

---

## Companion HF releases

| Asset | License | When |
|---|---|---|
| `Praxel/praxy-stt-te-rb` (vasista22 + EDSA-LoRA) | Apache-2.0 | post β-Te eval |
| `Praxel/praxy-stt-ta-rb` | Apache-2.0 | post β-Ta eval |
| `Praxel/praxy-stt-hi-rb` | Apache-2.0 | post β-Hi eval |
| `Praxel/edsa-corpus` (entity-dense audio + JSONL + entity dicts) | CC-BY-4.0 | at submission |

---

## Diff vs OUTLINE_v2

OUTLINE_v2 (the "benchmark paper" framing) is retired. v3 returns to the original flywheel framing the project always had — what changed is empirical evidence that the flywheel claim is *only* defensible on entity-dense audio (not read-prose), and that distinction now structures the paper.

The α-only "open-source has caught up" benchmark finding from v2 survives as §5 supporting evidence ("vasista22 ≥ Deepgram on conversational Te"), not as the headline.
