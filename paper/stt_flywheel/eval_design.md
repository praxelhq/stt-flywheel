# STT-Flywheel — Evaluation Design

## Holdout sets (per language; Te / Ta / Hi)

| Set | Size | Composition | Source | Audio license | Notes |
|---|---|---|---|---|---|
| **Entity-dense** | 300 utts | 60 digits / 60 currency / 60 addresses / 60 brands / 60 codemix | 100 IV-held-back speakers + 100 newly recorded by native speaker (Te=Pushpak; Ta + Hi = collaborator afternoons) + 100 CC-licensed YouTube podcasts/news with verified human transcripts | mixed (cite per row) | **Headline metric set.** Mutually exclusive from train. |
| **FLEURS regression** | 100 utts | uniform sample of FLEURS test split | FLEURS (CC-BY-SA) | open | Guards against catastrophic forgetting. |
| **Code-mix subset** | 100 utts | strict subset of `codemix` rows in entity-dense | (same as above) | mixed | Reported separately to isolate Hinglish/Tenglish/Tanglish gain. |

Total: 500 utts × 3 langs = 1500 utts. Audio is human-recorded (no synthetic).
The 100 newly-recorded clips per language are licensed CC-BY by the speaker
to Praxel HQ for redistribution alongside the paper.

## Ground-truth schema

Each holdout row JSON:

```json
{
  "id": "te_entity_001",
  "lang": "te",
  "audio_path": "holdouts/te/entity/001.wav",
  "duration_s": 5.7,
  "text": "నా మొబైల్ నంబర్ తొమ్మిది ఎనిమిది ఏడు ...",
  "text_norm": "...",                          // post-normalisation form
  "entity_class": "digits",
  "entity_tokens": [
    {"surface": "తొమ్మిది", "start": 19, "end": 27, "type": "digit_word"},
    ...
  ],
  "speaker_id": "iv_te_023",
  "source": "indicvoices",
  "source_license": "CC-BY-4.0"
}
```

## Metrics

### 1. Word-WER (`jiwer`, Whisper-norm)

Standard normalisation:
- Lowercase Latin
- Strip punctuation except in-word apostrophes
- Collapse whitespace
- Indic: drop ZWJ/ZWNJ; preserve native script

Numerals normalised both ways (digit-form `5` and word-form `ఐదు` accepted as
equivalent — pre-defined per-class rule, not learned). This is **critical**:
without it, R1-style failures look catastrophic on `digits`.

### 2. Char-WER

Standard CER over Whisper-normalised strings. Reported because:
- Short brand names hide in word-WER noise
- R1 found CER tracks Indic phoneme integrity better than WER

### 3. Entity-Hit-Rate (EHR) — **headline**

Defined per row:

    EHR_row = (# entity_tokens whose surface — under normalisation — appears in
               the model hypothesis, in any order) / (# entity_tokens in row)

Aggregated as macro-mean over rows in a holdout set. Reported per
`entity_class` and overall.

### 4. LLM-judge semantic equivalence

Re-uses `evaluation/llm_wer.py` (Qwen-2.5-72B). Asks: is the hypothesis
semantically equivalent to the reference? Returns scalar 0..1. Aggregated
as macro-mean. Used as a tiebreaker when WER and EHR disagree (often the
case for code-mix — commercial systems pay WER cost for synonym
substitution that's actually correct).

### 5. FLEURS regression delta

`WER(LoRA) - WER(base)` on FLEURS holdout. **Hard ceiling: +2 pp.** Anything
worse means the LoRA broke general capability and the run is rejected
regardless of EHR gains.

## Baseline systems

| System | Rationale | How invoked | Cost on 1500 utts |
|---|---|---|---|
| `whisper-large-v3` (vanilla, per-lang prefix) | Same base we LoRA on; isolates LoRA effect | Modal A10G batch | ~$1 |
| AI4Bharat Indic-specific Whisper variants (`vasista22/whisper-{hi,te,ta}-large-v2` if available) | Open Indic baselines | Modal A10G batch | ~$2 |
| AI4Bharat IndicConformer | Strong open Indic ASR; non-Whisper architecture | Modal A10G batch | ~$2 |
| Deepgram Nova-3 (`nova-3` model with `language=hi/ta/te` if supported, else `multi`) | Western commercial benchmark | Deepgram API; existing $500 credit | ~$1.20 |
| ElevenLabs Scribe (`scribe_v1`) | Newer commercial entrant | 11labs API; existing credits | ~$1.40 (free) |
| Sarvam-STT | Indic-specialist commercial; only if API GA at eval time | Sarvam API | $0 if free tier; else N/A |
| Google Chirp-v2 | Optional, only if budget permits | Google STT API | ~$5 (skipped unless under budget) |

For each system × language × holdout, we report all 4 metrics + 95% bootstrap
CI on EHR (1000 resamples).

## "Win" criteria for the paper claim

The paper claim is **conditional and narrow** (per Pushpak's intellectual-honesty constraint):

- **Primary claim:** Praxy-STT-R2 beats vanilla Whisper-v3 by **≥10 pp Entity-Hit-Rate** on the entity-dense holdout for **at least 2 of 3 languages** (Te + one of {Ta, Hi}).
- **Secondary claim:** Praxy-STT-R2 beats Deepgram Nova-3 by **≥5 pp EHR** on the entity-dense holdout for those same languages.
- **Non-regression:** FLEURS WER delta vs base ≤ +2 pp absolute for **all 3** languages.
- **Honesty caveats reported alongside results:**
  - We do NOT claim parity on FLEURS / general read-prose Hindi news.
  - We do NOT claim streaming/real-time capability.
  - If Sarvam-STT is GA and beats us, we cite that and pivot framing to "best open vs commercial Indic" rather than "best across the board".

## Ablation table

| Run | Synth fraction | Real fraction | Per-lang adapter? | Decoder prefix | Expected EHR (Te) |
|---|---|---|---|---|---|
| A0 | 0% | 100% | yes | per-lang | baseline |
| A1 | 30% | 70% | yes | per-lang | **target +10 pp** |
| A2 | 60% | 40% | yes | per-lang | likely degrades |
| A3 | 30% | 70% | shared (R1-style) | per-lang | tests adapter benefit |
| A4 | 30% | 70% | yes | shared `<|hi|>` (R1-style) | should be much worse on Ta |

A0/A1 + A3/A4 (4 runs total) are the budgeted ablation. A2 only if budget allows ($25 extra per lang).

## Reproducibility

- Holdout JSONL + audio: `paper/stt_flywheel/holdouts/{lang}/`
- Manifests: `data/stt_flywheel/manifests/`
- Scorecards: `evaluation/scorecards/stt_r2_*.json`
- LoRA weights: HF `Praxel/praxy-stt-r2-{te,ta,hi}` (Apache-2.0)
- Eval driver: `paper/stt_flywheel/modal_stt_train.py::eval_checkpoint`
- Bootstrap CI script: TODO `paper/stt_flywheel/bootstrap_ci.py` (not yet scaffolded)
