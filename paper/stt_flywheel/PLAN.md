# STT Flywheel — Experiment Plan (R2, scoped 2026-04-28)

**Working title:** *The TTS ↔ STT Flywheel: Entity-Dense Synthetic Audio as a Free Training Signal for Indic ASR*

**Status:** scoping doc. No Modal jobs run, no API calls made. Gate to execute: Praxy TTS paper (arXiv:2604.25441) and PSP paper (arXiv:2604.25476) are live; R6 weights on HF (`Praxel/praxy-voice-r6`); ~$200 of remaining $400 project credit reserved for this sprint.

**Target venue:** Interspeech 2027 (deadline ~March 2027) primary; EMNLP 2026 industry track as fallback.

---

## 1. Hypothesis & paper claim

An open Indic TTS (Praxy R6) + free commercial-credit TTS (ElevenLabs, Cartesia) can manufacture an **entity-dense Indian-English code-mix corpus** at near-zero marginal cost. A LoRA-fine-tuned `whisper-large-v3` trained on this synth corpus mixed with real audio (IndicVoices + Common Voice) **beats vanilla Whisper-v3, AI4Bharat IndicConformer, Deepgram Nova-3, and Sarvam-STT on entity-dense Indic speech** (digits, currency, addresses, brands, code-mix) for at least two of {Te, Ta, Hi}, while **not regressing** on FLEURS read-prose by more than 2 pp WER.

The flywheel is the contribution: the same TTS we shipped is reused as a training-data factory; the resulting STT then becomes the canonical Indic-WER evaluator for our TTS paper's table — closing the loop without circular training.

## 2. R1 failure modes the design must fix

| R1 failure | R2 fix |
|---|---|
| Hindi-proxy `<|hi|>` for Tamil destroyed Whisper-v3's strong native Tamil capability (CER 7.4% → 102%) | Use **per-language decoder prefix** (`<|te|>`, `<|ta|>`, `<|hi|>`) — no proxy. |
| 100% synthetic, single-source (Sarvam Bulbul) → domain mismatch with FLEURS | **Mix synth ≤ 30%**, real ≥ 70%; multi-source synth (Praxy R6 + 11labs + Cartesia). |
| Single shared adapter across 3 langs caused Tamil bleed-through | **Per-language LoRA adapters**; one model, three swappable adapters. |
| n=5 eval was undersized; ranking unreliable | **300-utt entity-dense holdout per language** + 100-utt FLEURS regression set. |
| Script contamination in Tamil training rows (Devanagari/Kannada glyphs) | Strict per-row script validator; reject any row whose `text` field has non-target-script chars except whitelisted Latin spans. |

## 3. Entity classes & utterance recipes

Six classes × 3 languages × ~500 utts/class = ~9,000 entity-dense base utterances per language; each rendered through ~6 voices/configs → ~54k clips/lang.

| Class | Definition | Telugu example | Hindi example | Tamil example |
|---|---|---|---|---|
| `digits` | Phone numbers, OTPs, PIN codes, account numbers | "నా మొబైల్ నంబర్ తొమ్మిది ఎనిమిది ఏడు ఆరు ఐదు నాలుగు మూడు రెండు ఒకటి సున్నా" | "मेरा OTP है 4 7 2 9 1 6" | "என் ஃபோன் நம்பர் 9876543210" |
| `currency` | INR amounts (lakh/crore + decimals) | "ఈ నెల జీతం యాభై ఆరు వేల ఎనిమిది వందల రూపాయలు" | "किराया पच्चीस हज़ार छह सौ रुपये है" | "என் சம்பளம் நாற்பது ஆயிரம் ரூபாய்" |
| `addresses` | Street + locality + city + PIN | "ప్లాట్ నంబర్ 42, బంజారా హిల్స్, హైదరాబాద్ 500034" | "B-204, सेक्टर 62, नोएडा 201301" | "5/7, அண்ணா நகர், சென்னை 600040" |
| `brands` | Indian + global brand names (Swiggy, Zomato, Flipkart, Paytm, HDFC, Tata) | "Swiggy లో biryani order చేశాను" | "Flipkart पर sale चल रही है" | "Zomato-ல order பண்ணினேன்" |
| `codemix` | Hinglish/Tenglish/Tanglish call-centre style — 30-50% English tokens | "మా CEO meeting ఇవాళ schedule అయింది, అతను confirm చేశాడు" | "मैंने uber book कर liya है, ETA 5 minutes है" | "Project deadline நாளைக்கு, team-ஆ standup-ல update பண்ணுங்க" |
| `proper_nouns` | Indian person + place names (Telugu/Tamil/Hindi diaspora across all three) | "శ్రీనివాస రామానుజన్ తిరువళ్ళూర్ లో పుట్టారు" | "अरविंद केजरीवाल दिल्ली के मुख्यमंत्री थे" | "முத்துசாமி திருவாரூரில் பிறந்தார்" |

**Generation pipeline for text:** Claude Haiku 4.5 via Anthropic-direct (existing `evaluation/anthropic_client.py` budget-capped client). Per-class prompt templates with seeded entity dicts (`stt/build_entity_corpus.py` already has 7,541 entries on disk — we extend to ~3,000/class/lang). Cost: ~$2 against the $100 Anthropic cap (Pushpak has $500 grant credit, we use Anthropic-direct so the $3 OpenRouter cap is not the bottleneck).

## 4. Synthetic audio generation pipeline

**Volume target:** 50k clips per language (Te/Ta/Hi), avg 6s each = ~83 hrs/lang = ~250 hrs total synth.

**Source mix per language (60% Praxy bucket dispatched per production router `serving/praxy_router.py`):**

| Source | Share | Routing | Rationale | Cost |
|---|---|---|---|---|
| **Praxy bucket** (60%, 30k) | dispatched per (lang, codemix-or-not): | | | |
| ↳ Praxy R6 LoRA on Chatterbox | (te \| ta, non-codemix) ≈ 35% | Modal A10G, Config B, 4 ref voices/lang | Our model; primary flywheel loop | $40-60 Modal (50-70 A10G-hr at $1.10/hr — earlier $30 was 2× under-estimated) |
| ↳ Vanilla Chatterbox + Config B | (hi, non-codemix) ≈ 17% | Modal A10G, no LoRA | R6 regresses Hi LLM-WER 0.025 → 0.334 (paper §V.B) | included above |
| ↳ Translit → IndicF5 | (any, codemix) ≈ 8% | Modal, Haiku translit + IndicF5 | Only working codemix recipe per `project_chatterbox_codemix_dead_end_2026-04-27` | included above |
| **ElevenLabs v3** (`eleven_v3` model_id; 8 voices verified Indic-capable per `data/codeswitch_gen.ELEVENLABS_MULTILINGUAL_VOICES`) | 20% (10k) | All (lang, class) | Free credits (~32M); native codemix support | $0 (credits) |
| **Cartesia sonic-3** (Indic voices in `serving/commercial_baselines.CARTESIA_VOICES`; Te/Ta feminine-only) | 20% (10k) | All (lang, class) | Free credits (~16M); different acoustic prior | $0 (credits) |

**Voice diversity:** ≥6 distinct speaker references per language across all sources combined (Praxy: 4 ref voices, 11labs: 8 voices, Cartesia: 1-2 Indic voices/lang). Augmentation pass — **applied at training time inside Whisper's preprocessor (NOT pre-baked)**: SpecAugment on the log-mel spectrogram (`time_mask=10`, `freq_mask=27`, `num_time_masks=2`); offline-only ops are bandwidth/gain normalisation, not SpecAugment. Speed perturb (0.9–1.1×) is applied offline at audio-domain via `torchaudio.transforms.Resample` to keep alignment exact.

**Audio resampling: explicit step.** All sources output 24 kHz; Whisper-v3 expects 16 kHz mono. Pipeline includes a `torchaudio.functional.resample(orig_freq=24000, new_freq=16000, lowpass_filter_width=64, resampling_method='kaiser_window')` step before manifest write. Stereo → mono via channel mean.

**Script validator:** every synth clip is re-transcribed and CER-filtered. **Per-language judge** because Whisper-v3 is known weak on Telugu (CLAUDE.md: "outputs in Kannada script sometimes"):

- **Te / Ta**: AI4Bharat IndicWhisper (`vasista22/whisper-telugu-large-v2`, `vasista22/whisper-tamil-large-v2`) at CER ≤ 0.5
- **Hi**: Whisper-large-v3 at CER ≤ 0.5 (Whisper-v3 is competitive on Hi)
- **Codemix (any lang)**: Whisper-large-v3 + LLM-judge (Sonnet 4.6) for semantic equivalence — STT alone is unreliable on codemix per TTS paper §VI.C.

Drop any clip that fails the relevant filter. Estimate ~10–15% rejection rate based on R6 PSP findings.

## 5. Real-audio mixing

Real corpora (must stay ≥70% of training mix):

| Corpus | Languages | Hours (verified 2026-04-29) | Access | License | Notes |
|---|---|---|---|---|---|
| **IndicVoices** (`ai4bharat/IndicVoices`) | Te 56 train shards, Ta 78, Hi 64 | est. ~120h after speaker-cap sampling | ✅ HF API confirms 1,438 files accessible to this account | CC-BY 4.0 | Conversational. Per-speaker cap of 25 clips to avoid voice-id dominating. |
| **Common Voice 17** (Mozilla, `commonvoice.mozilla.org/datasets`) | Te ~3h, Ta ~30h, Hi ~17h (CV17 stats page) | ~50h | ⚠️ NOT on HuggingFace — must download `cv-corpus-17.0-yyyy-mm-dd-{lang}.tar.gz` from commonvoice.mozilla.org with email-gated link | CC0 | Read-prose; helps FLEURS regression. Earlier plan over-estimated CV hours by ~2× — corrected. |
| **FLEURS** train split (`google/fleurs`) | Te/Ta/Hi ~3-7h each | ~15h total | ✅ HF accessible | CC-BY-SA | In-distribution for FLEURS regression test set. Earlier plan said "~10h each = 30h" — actual is ~half that. |

**Final per-language manifest:** ~80h synth + ~50-60h real ≈ 130-140h × 3 langs = **~400h total** (revised down from ~470h). Synth share: ~25% / real ~75%. Still meets the "≥70% real" constraint.

**Sequencing risk:** Common Voice download is email-gated (24-48h delivery). Schedule the download at W1, not W3, so it's ready when training starts.

## 6. Training recipe

**Base model:** `openai/whisper-large-v3` (R1 found `ai4bharat/indic-whisper-large` does not exist on HF; AI4Bharat's actual public artefact is the Whisper-medium-fine-tune `vasista22/whisper-hindi-large-v2` etc., language-specific). Whisper-v3 with per-language prefix is the cleanest baseline.

**Adapter strategy:** **3 separate LoRAs** (one per language), shared base weights. Inference selects adapter by language tag.

**LoRA config:**
- Rank 16, α 32
- Target modules: `q_proj k_proj v_proj out_proj` on **both encoder self-attn and decoder self/cross-attn** (R1 only hit one side)
- `modules_to_save=[]` (no embedding edits — R1's `<|translit|>` token added complexity without payoff)
- bf16 on A10G

**Hyperparams:**
- Per-language: 6,000 optimizer steps, bs=4 grad_accum=4 (eff 16), peak LR 8e-5 cosine, warmup 300
- Optimizer: AdamW (β=0.9, 0.95, weight-decay 0.01)
- Seed: 1337 (use 42 + 1337 + 7000 for triplicate runs in Phase 4 ablation)
- ~12 A10G-hr per language × 3 = 36 A10G-hr total
- Modal cost: ~$45-55 for all 3 LoRA rounds

**Software pin chain** (matches `project_indicf5_unblock_recipe_2026-04-27` chain that works on Modal):
```
torch==2.4.0
transformers==4.49.0
peft==0.13.0
accelerate==0.30.0
datasets
librosa
jiwer
soundfile
```
CUDA 12.1.

**Divergence-abort:** save every 500 steps, eval entity-WER on a 30-utt smoke holdout; abort if WER goes up 2 consecutive checkpoints (R1 lesson — Tamil destruction was visible at step 1500 had we checked).

## 7. Evaluation design

**Holdout sets (per language, mutually exclusive from training):**

1. **Entity-dense holdout** — 300 utts curated from real human recordings:
   - 100 from IndicVoices held-back speakers
   - 100 newly recorded by 1 native speaker per language (Pushpak for Te; collaborator afternoons for Ta + Hi) — 30 mins each, $0
   - 100 from YouTube creative-commons Indian podcasts/news with verified human transcripts
2. **FLEURS regression set** — 100 utts, read-prose; ensures we didn't break general capability
3. **Code-mix subset** — 100 utts subset of (1), specifically the `codemix` class

**Metrics:**
- **Word-WER** (standard, normalized via Whisper's English+Indic norm)
- **Character-WER** for short-form (digits, brands)
- **Entity-Hit-Rate** (headline metric, precise definition):
  - For each ground-truth row, `entity_tokens` is a list of `EntityToken{surface, start, end, type}` produced by `paper.stt_flywheel.data_pipeline.entity_token_tagger`.
  - For each predicted hypothesis, run the same `entity_token_tagger(hypothesis_text, entity_class)` to get hypothesis spans.
  - **Hit normalisation rules per `type`** (deterministic, pre-registered):
    - `digit_run` (numeric "9876543210"): NFKC-normalised exact match against any extracted digit_run in hypothesis.
    - `spelled_digit` (e.g., "ఐదు ఆరు సున్నా"): wordwise match against vocabulary list of native digit-words for that language (`SPELLED_DIGITS_BY_LANG` in `clean_corpus.py`); a hit requires ≥80% of spelled digits in correct order in hypothesis.
    - `currency_amount`: numeric value (after parsing "5 lakh" / "ఐదు లక్షల" via `praxy.linguistics.indic_numbers.parse_amount`) within ±0.5% of GT.
    - `pincode` (6-digit Indian PIN): exact match.
    - `house_or_plot` (e.g., "B-204", "5/7"): NFKC + case-fold match.
    - `brand`: case-folded match in either Latin or native script form (uses `BRAND_ALIASES` table).
    - `proper_noun`: token-set match with ≥80% overlap (allows transliteration variance).
  - **Mixed numeric / spelled scoring**: "5" and "five" are *both* counted as a hit if either produces the correct numeric value. Same for "5 lakh" vs "five lakh" vs "ఐదు లక్షల".
  - **EHR formula**: `EHR = sum(hits_per_row) / sum(total_entity_tokens_per_row)`. Reported per (lang, class) cell + macro-averaged.
  - **Hypothesis-side parser**: same `entity_token_tagger` is run on the STT output text (after Whisper text-normalisation). LLM-judge is NOT used in the headline EHR — too non-deterministic for a paper number.
- **LLM-judge semantic equivalence** (existing `evaluation/llm_wer.py`, Sonnet 4.6) — secondary metric; captures synonym-level matches commercial systems get penalized for under raw WER. Reported alongside EHR for completeness, not as the headline.

**Baselines (cost):**
| System | Cost on 900 utts × 3 langs | Notes |
|---|---|---|
| Vanilla Whisper-v3 | $0 (Modal, ~1hr A10G) | $1 |
| AI4Bharat IndicConformer | $0 (Modal, public weights) | $1 |
| Deepgram Nova-3 | ~900 × 3 × 6s = 16,200s = 270 min × $0.0043 = ~$1.20 | uses existing $500 credit |
| Sarvam-STT | API not GA — best-effort via their playground; cite as N/A if unavailable | $0 |
| ElevenLabs Scribe | ~$0.30/hr × 4.5h ≈ $1.40 | optional |

**"Win" definition:** Praxy-STT-R2 beats vanilla Whisper-v3 by ≥10 pp Entity-Hit-Rate on entity-dense holdout for ≥2 of 3 langs, AND beats Deepgram Nova-3 by ≥5 pp on the same. FLEURS WER regression ≤2 pp absolute.

## 8. Budget breakdown (USD)

| Line item | Cost | Notes |
|---|---|---|
| Anthropic Haiku 4.5 for entity-dense text gen | $2.00 | within $100 Anthropic-direct cap (Pushpak has $500 grant) |
| TTS gen via Praxy bucket on Modal A10G (50-70 A10G-hr/lang × 3 = 150-210 A10G-hr total) | $60-80 | 60% of audio; routed per `serving/praxy_router.py` (R6 LoRA on Te/Ta non-codemix; vanilla Chatterbox on Hi non-codemix; translit→IndicF5 on all codemix). Earlier $30 was 2× under-estimated. |
| TTS gen via ElevenLabs v3 (10k clips × 3 langs) | $0.00 | ~32M credits available |
| TTS gen via Cartesia sonic-3 (10k clips × 3 langs) | $0.00 | ~16M credits available |
| Synth audio post-processing (CPU) | $2.00 | augmentation + Whisper-v3 re-transcribe filter |
| Real-corpus download + manifest build (CPU) | $2.00 | IndicVoices + CV + FLEURS |
| LoRA training: 3 × A10G × ~12 hr | $50.00 | Modal A10G ~$1.10/hr |
| Eval on Modal (Whisper-v3 + IndicConformer baselines + LoRA) | $5.00 | A10G batched |
| Deepgram Nova-3 baseline | $1.50 | uses $500 credit |
| ElevenLabs Scribe baseline (optional) | $1.50 | uses 11labs credit |
| Buffer for divergence reruns / debug | $20.00 | 1 LoRA reroll |
| **Total** | **~$145-165** | revised up after correcting Praxy throughput estimate; still under $200 cap |

Modal $15/night cap honoured: TTS-gen split across 6 nights (~$5/night), training across 4 nights (~$15/night).

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Praxy R6 synth quality too low for Te entities (PSP showed 26.7% retroflex collapse) | Med | Entity-Hit-Rate drops; LoRA learns wrong phonemes | Whisper-v3 CER filter @ 0.5; fall back to higher 11labs share for Te |
| 11labs/Cartesia rate limits or Indic voice quality drift | Low | Audio variety reduced | Cap requests at 5k/day/provider; pre-validate 100-clip sample |
| Entity-dense ground truth has ASR-judgement disagreements (numerals: "5" vs "five") | High | EHR metric noisy | Pre-define normalisation rules per class; LLM-judge as secondary |
| Tamil destruction reappears (R1 mode) | Low (per-lang adapter + per-lang prefix fix it) | Paper claim collapses on Ta | Divergence-abort at 500-step checkpoints; per-lang train decoupled |
| Sarvam-STT releases public API mid-sprint and beats us | Med | Lose "vs commercial Indic" framing | Reposition: still SOTA among **open** + competitive with Western commercial; cite Sarvam if results available |
| Real-corpus license issues (IndicVoices terms) | Low | Can't redistribute manifests | Distribute manifest as `(corpus_id, utt_id)` tuples + load script — no audio redistribution |
| Modal A10G availability / preemption | Low | Schedule slip | Pre-warm volume; `serving/modal_app.py` patterns already handle re-attach |
| Synth audio leaks into real holdout (data leakage) | Med | Inflated EHR | Holdout text strings hashed and excluded from synth gen step |

**Abort triggers:**
1. After 1 LoRA round per lang, if EHR-vs-base improvement is <2 pp on **all 3** languages → abort, reassess data mix
2. If Modal spend hits $130 (15% over budget) → freeze training, ship with current best
3. If real-corpus access blocked for ≥1 language → ship 2-language paper (Te + Hi) and document Ta as future work

## 10. Timeline (post-TTS-paper-ship)

| Week | Deliverables | Spend |
|---|---|---|
| W1 | Entity dicts extended to 3k/class/lang; Haiku 4.5 text gen run; manifest validator | $4 |
| W2 | Synth audio generation (Praxy R6 + 11labs + Cartesia) + filter pass | $35 |
| W3 | Real-corpus pull + unified manifest; per-lang LoRA round 1 (Te first, single-lang sanity) | $25 |
| W4 | Ta + Hi LoRA rounds; eval against vanilla Whisper-v3 | $30 |
| W5 | Commercial baselines (Deepgram, IndicConformer); ablation on synth fraction (0%, 30%, 60%) | $15 |
| W6 | Paper draft, figures, HF release as `Praxel/praxy-stt-r2-{te,ta,hi}` | $5 |

Total elapsed: ~6 weeks; total spend: ~$114 + buffer = ~$135 ceiling.

---

## Open questions (not blockers)

1. Whether to include Bengali/Kannada as a "scaling-out" appendix table once the recipe is locked. Out of scope for v1; would push budget over $150.
2. Whether to LoRA the encoder only vs encoder+decoder. Per IndicWhisper community findings, decoder-side LoRA helps Indic decoding more; we include both. Ablate if budget allows.
3. Diarization — explicitly out of scope for this paper; logged as STT v2 follow-up.
