# stt-flywheel

Open-source companion repository for **"The TTS↔STT Flywheel for Entity-Dense Indic ASR"** (Praxel HQ, 2026).

This repo holds everything reviewers need to verify the paper's numbers without rebuilding the training pipeline: the Entity-Hit-Rate (EHR) metric, the held-out test sets, the per-utterance predictions from every system reported in the paper, the EDSA entity-dense corpus text, and the seed entity dictionaries.

> **Paper.** arXiv preprint: *(URL pending arXiv submission; will be linked here on appearance.)*
>
> **Companion repos.**
> - [`praxelhq/praxy-tts`](https://github.com/praxelhq/praxy-tts) — *private* — the TTS half of the flywheel (Praxy R6 LoRA training, Chatterbox / IndicF5 / commercial wrappers, Modal infra).
> - [`praxelhq/psp-eval`](https://github.com/praxelhq/psp-eval) — *public* — the Phoneme Substitution Profile accent metric used in the companion Praxy Voice paper.

## Headline result

Entity-dense (Cartesia held-out) Entity-Hit-Rate (EHR) — bold = best per row. *n* = 102 (Te, Ta), *n* = 86 (Hi).

| Lang | Vanilla Whisper-v3 | Praxy-STT-r2 | vasista22 | Deepgram Nova-3 | **Praxy-STT-rb** |
|------|-------------------:|-------------:|----------:|----------------:|-----------------:|
| Te   | 0.560              | 0.853        | 0.027     | 0.160           | **0.473**        |
| Hi   | —                  | —            | 0.049     | **0.485**       | 0.337            |
| Ta   | —                  | —            | 0.025     | 0.025           | **0.543**        |

vasista22 = `vasista22/whisper-{telugu,hindi,tamil}-large-v2` (open SOTA, IIT-Madras Speech Lab). Praxy-STT-rb = vasista22 + entity-dense LoRA (this work). Praxy-STT-r2 = Whisper-v3 + per-language LoRA (vanilla-base sibling reported for the Script Fidelity Rate analysis; see paper §5.3).

Native human-recorded sanity check (Telugu, *n* = 20 utterances): Praxy-STT-rb EHR **0.516** on real speech vs **0.473** on synth holdout — no synth-distribution overfit.

## Quick start: verify a single cell

```bash
git clone https://github.com/praxelhq/stt-flywheel.git
cd stt-flywheel
pip install jiwer regex  # only deps eval_ehr.py needs

# Score Praxy-STT-rb predictions against the entity-dense Telugu holdout
python paper/stt_flywheel/eval_ehr.py \
  --gt   data/stt_flywheel/holdouts/te/entity_dense_cartesia.jsonl \
  --hyp  evaluation/scorecards/stt_flywheel/beta_te_entity_dense_cartesia_n102_predictions.jsonl

# Run the metric's unit tests (19/19 should pass)
python -m pytest paper/stt_flywheel/test_eval_ehr.py -v
```

The exact CLI flags vary slightly between scripts; `--help` on any `eval_*.py` enumerates them.

## Repo layout

```
stt-flywheel/
├── README.md                 (this file)
├── LICENSE                   MIT (code) + CC-BY-4.0 / CC0 notes (data)
│
├── paper/stt_flywheel/
│   ├── stt_flywheel.tex      Full LaTeX source
│   ├── refs.bib              BibTeX references
│   ├── stt_flywheel.pdf      Compiled PDF (mirror of arXiv submission)
│   ├── fonts/                Noto Devanagari/Telugu/Tamil TTFs (paper compile)
│   ├── figures/              All paper figures (PDF + PNG) + make_figures.py
│   ├── eval_ehr.py           THE Entity-Hit-Rate metric (deterministic, 19/19 tests)
│   ├── test_eval_ehr.py      Unit tests for EHR
│   ├── data_pipeline.py      EDSA corpus generation methodology (text-only)
│   ├── eval_lora.py          Praxy-STT-r2 / -rb evaluation harness
│   ├── eval_vasista_baseline.py / eval_vasista_jsonl.py
│   │                         vasista22 baseline harness
│   ├── eval_beta.py / eval_beta_jsonl.py
│   │                         Praxy-STT-rb (β) harness for HF holdouts and JSONL
│   ├── eval_deepgram_holdout.py
│   │                         Deepgram Nova-3 commercial baseline harness
│   ├── eval_te_fleurs.py / eval_te_cv25.py / eval_te_jsonl_holdout.py
│   │                         FLEURS-Te + CV25-Te + held-out JSONL eval
│   ├── build_entity_dense_holdout.py / build_iv_general_holdout.py
│   │                         Holdout construction scripts
│   ├── eval_holdout_extractor.py / fleurs_regression_extractor.py
│   │                         Reference-text extraction utilities
│   ├── spelled_digit_rewriter.py
│   │                         Digit-spelling normaliser used by EHR
│   ├── audit_corpus.py / clean_corpus.py
│   │                         Corpus QA scripts (orphan WAV detector, etc.)
│   ├── RUNBOOK.md            How to reproduce each row in each table
│   ├── PLAN.md               Sprint plan
│   ├── eval_design.md        Eval-set design rationale
│   ├── OUTLINE_v3.md         Pre-registration document (canonical)
│   ├── RECORDING_SCRIPT.md / RECORDING_SCRIPT.jsonl
│   │                         Native sanity-check recording protocol + 20 sentences
│   └── __init__.py
│
├── data/stt_flywheel/
│   ├── holdouts/
│   │   ├── te/{iv_general,entity_dense_cartesia,pushpak_native}.jsonl
│   │   ├── te/pushpak_native/*.wav    20 native human Telugu recordings (CC0)
│   │   ├── hi/{iv_general,entity_dense_cartesia}.jsonl
│   │   └── ta/{iv_general,entity_dense_cartesia}.jsonl
│   └── text/                  EDSA entity-tagged corpus (~22k utterances, text-only)
│       └── {te,hi,ta}_{addresses,brands,codemix,currency,digits,proper_nouns}.jsonl
│
├── evaluation/scorecards/stt_flywheel/
│   ├── *_scorecard.json                Per-system per-holdout aggregate scorecards
│   └── *_predictions.jsonl             Per-utterance hypotheses for every system
│                                       in the paper. Allows re-scoring with any
│                                       alternative metric without re-running ASR.
│
└── stt/data/entities/
    ├── brands/        ta.jsonl (seed brand list, CC-BY-4.0)
    ├── addresses/     ta.jsonl
    └── proper_nouns/  ta.jsonl
```

### Notes on what is and is not in this repo

- **Synthesised holdout audio is not redistributed.** The `entity_dense_cartesia` holdout was synthesised with Cartesia `sonic-3` voices (and similarly the IV-general holdout, which references real IndicVoices speech). We ship only the ground-truth JSONL annotations (id / text / audio_path / entity_tokens / entity_class). To reproduce the audio, regenerate it from the transcript text using the TTS pipeline described in the paper §3.
- **Native Telugu recordings (`pushpak_native/*.wav`)** are included — 20 short clips (~5 MB total), released CC0 by the speaker (one of the authors).
- **FLEURS-Te regression rows** (Table 4) are evaluated by streaming `google/fleurs` directly from HuggingFace inside `eval_te_fleurs.py`; no static JSONL is needed.
- **The `data/stt_flywheel/audio/` synth corpus** (~22k WAVs used to train the LoRA) is not redistributed because it carries Cartesia / ElevenLabs / IndicF5 / Praxy R6 synthesised voice identities. The text-only counterpart in `data/stt_flywheel/text/` is sufficient to regenerate it.
- **Modal training scripts** are not included in this public repo — they live in the private `praxy-tts` repo. The released LoRA adapter weights (HuggingFace, below) are the artefact reviewers can directly load.

## Reproducing the LoRA training

The trained LoRA adapters are released on HuggingFace under Apache-2.0:

| Adapter | Base | Use case |
|---|---|---|
| [`Praxel/praxy-stt-te-rb`](https://huggingface.co/Praxel/praxy-stt-te-rb) | `vasista22/whisper-telugu-large-v2` | Telugu entity-dense (paper headline) |
| [`Praxel/praxy-stt-hi-rb`](https://huggingface.co/Praxel/praxy-stt-hi-rb) | `vasista22/whisper-hindi-large-v2`  | Hindi entity-dense |
| [`Praxel/praxy-stt-ta-rb`](https://huggingface.co/Praxel/praxy-stt-ta-rb) | `vasista22/whisper-tamil-large-v2`  | Tamil entity-dense |
| [`Praxel/praxy-stt-te-r2`](https://huggingface.co/Praxel/praxy-stt-te-r2) | `openai/whisper-large-v3`            | Telugu Script-Fidelity-Rate fix |
| [`Praxel/praxy-stt-hi-r2`](https://huggingface.co/Praxel/praxy-stt-hi-r2) | `openai/whisper-large-v3`            | Hindi (SFR analysis) |
| [`Praxel/praxy-stt-ta-r2`](https://huggingface.co/Praxel/praxy-stt-ta-r2) | `openai/whisper-large-v3`            | Tamil (SFR analysis) |

Inference snippet:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torch, librosa

base = "vasista22/whisper-telugu-large-v2"
adapter = "Praxel/praxy-stt-te-rb"

model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload().to("cuda")
processor = WhisperProcessor.from_pretrained(base)

audio, _ = librosa.load("your_clip.wav", sr=16000)
features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
ids = model.generate(features, language="te", task="transcribe")
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## License

- **Code:** MIT.
- **Entity dictionaries (`stt/data/entities/`) and EDSA corpus text (`data/stt_flywheel/text/`):** CC-BY-4.0.
- **Holdout JSONL ground truths:** CC-BY-4.0.
- **Native Telugu recordings (`data/stt_flywheel/holdouts/te/pushpak_native/`):** CC0 (public domain).
- **LoRA adapter weights (HuggingFace):** Apache-2.0.

See [LICENSE](./LICENSE) for the full text and per-asset breakdown.

## Citation

```bibtex
@misc{stt_flywheel_2026,
  title        = {The {TTS}{$\leftrightarrow$}{STT} Flywheel for Entity-Dense Indic {ASR}},
  author       = {Menta, Pushpak and {Praxel HQ}},
  year         = {2026},
  eprint       = {arXiv:TBA},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  howpublished = {\url{https://github.com/praxelhq/stt-flywheel}}
}
```

(arXiv ID will be filled in on appearance.)
