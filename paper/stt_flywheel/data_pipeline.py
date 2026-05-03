"""STT-flywheel entity-dense corpus generator (real implementation).

Produces a manifest of entity-dense Indic utterances and (optionally)
synthesizes audio for them via Praxy R6 / ElevenLabs / Cartesia.

Defaults to ``--dry-run``: prints the planned API calls + cost estimate,
spends $0. Pass ``--execute`` to actually run.

Pipeline:

    1. Seed entity dictionaries (loaded from stt/data/entities/{cls}/{lang}.jsonl
       if present; otherwise we ask Qwen for inline candidates).
    2. Qwen-2.5-72B (OpenRouter, budget-capped) generates utterances per
       (language, entity-class). Output JSONL is cached on disk so re-runs
       are idempotent.
    3. Validate each row: script-purity, entity-presence, length range.
    4. Tag entity spans for downstream Entity-Hit-Rate evaluation.
    5. (optional) synthesize audio for each row through 60% Praxy R6 /
       20% ElevenLabs / 20% Cartesia.
    6. Write training manifest JSONL (one row per audio clip).

Run:
    uv run python -m paper.stt_flywheel.data_pipeline build --plan paper/stt_flywheel/plan.yaml
    uv run python -m paper.stt_flywheel.data_pipeline build --plan ... --execute

Tests (pure-logic helpers, no API calls):
    uv run python -m paper.stt_flywheel.test_data_pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTITY_DIR = REPO_ROOT / "stt" / "data" / "entities"
TEMPLATE_DIR = REPO_ROOT / "stt" / "data" / "prompt_templates"
OUT_TEXT_DIR = REPO_ROOT / "data" / "stt_flywheel" / "text"
OUT_AUDIO_DIR = REPO_ROOT / "data" / "stt_flywheel" / "audio"
OUT_MANIFEST_DIR = REPO_ROOT / "data" / "stt_flywheel" / "manifests"
CACHE_DIR = REPO_ROOT / "data" / "stt_flywheel" / "_cache"

LANGS: tuple[str, ...] = ("te", "ta", "hi")
ENTITY_CLASSES: tuple[str, ...] = (
    "digits", "currency", "addresses", "brands", "codemix", "proper_nouns",
)

# Per (lang, class) generation target before audio fan-out.
ROWS_PER_LANG_CLASS = 1500

# Model + budget. Anthropic Haiku 4.5 via direct API (Pushpak has Anthropic
# grant credit; OpenRouter $3 cap was the bottleneck, Anthropic budget is
# enforced separately in evaluation/anthropic_client.py).
LLM_MODEL = "claude-haiku-4-5"
ANTHROPIC_BUDGET_USD = 5.00  # generous cap; Haiku is $1/$5 per Mtok, this run is ~$2 estimated

# Synth audio source-mix per language (PLAN.md §4).
SYNTH_MIX = {
    "praxy_r6": 0.60,
    "elevenlabs": 0.20,
    "cartesia": 0.20,
}
SYNTH_CLIPS_PER_LANG = 50_000  # 50k clips/lang (PLAN.md §4)

# Unicode block ranges (inclusive) for each target language's primary script.
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "hi": (0x0900, 0x097F),  # Devanagari
}

# Length bounds (token-approx, whitespace-split).
MIN_TOKENS = 3
MAX_TOKENS = 25

Lang = Literal["te", "ta", "hi"]
EntityClass = Literal[
    "digits", "currency", "addresses", "brands", "codemix", "proper_nouns",
]
SynthSystem = Literal["praxy_r6", "elevenlabs", "cartesia"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntityToken:
    surface: str
    start: int
    end: int
    type: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UtteranceRow:
    id: str
    lang: str
    entity_class: str
    text: str
    entity_tokens: list[EntityToken] = field(default_factory=list)
    source_template: str = ""
    llm_run_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entity_tokens"] = [e if isinstance(e, dict) else asdict(e) for e in self.entity_tokens]
        return d


@dataclass
class AudioClip:
    clip_id: str
    utterance_id: str
    lang: str
    entity_class: str
    text: str
    audio_path: str
    system: str
    voice: str
    duration_s: float | None
    sample_rate: int


# ---------------------------------------------------------------------------
# Step 1 — seed dictionaries / templates
# ---------------------------------------------------------------------------


def load_entity_dict(lang: str, entity_class: str) -> list[dict]:
    """Read seed entities from `stt/data/entities/{class}/{lang}.jsonl`.

    Returns [] if missing — caller asks Qwen to invent plausible entities
    inline in that case.
    """
    p = ENTITY_DIR / entity_class / f"{lang}.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_templates(lang: str, entity_class: str) -> list[str]:
    """Read sentence templates with `{e0}`, `{e1}` slots."""
    # Try class-specific first, then per-lang fallback.
    for stem in (f"{lang}_{entity_class}.txt", f"{lang}.txt"):
        p = TEMPLATE_DIR / stem
        if p.exists():
            return [
                ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
    return []


# ---------------------------------------------------------------------------
# Step 2 — Qwen prompt + call
# ---------------------------------------------------------------------------


_LANG_NAMES = {"te": "Telugu", "ta": "Tamil", "hi": "Hindi"}

_CLASS_INSTR = {
    "digits": (
        "Phone numbers, OTPs, PIN codes, account numbers spoken naturally. "
        "Mix word-form digits and numeral digits (5 / ఐదు / पाँच). 6-12 digits per utt."
    ),
    "currency": (
        "Indian rupee amounts using lakh/crore/thousand. Salary, rent, EMI, GST contexts."
    ),
    "addresses": (
        "Indian street addresses: plot/flat number, locality, city, 6-digit PIN code."
    ),
    "brands": (
        "Indian + global consumer brands (Swiggy, Zomato, Flipkart, Paytm, HDFC, Tata, "
        "Amazon). Brand name is usually code-mixed Latin script."
    ),
    "codemix": (
        "Hinglish/Tenglish/Tanglish call-centre and office speech. "
        "Matrix language is Indic; 30-50% English tokens at lexical hotspots "
        "(tech, work, brands)."
    ),
    "proper_nouns": (
        "Indian person + place names. Mix scripts naturally; do NOT translate "
        "proper nouns."
    ),
}


def build_generation_prompt(
    lang: str,
    entity_class: str,
    entities: list[dict],
    templates: list[str],
    n_target: int,
) -> list[dict]:
    """Build OpenAI-style chat messages for Qwen.

    Returns a list of {"role", "content"} dicts ready for ``chat_complete``.
    """
    lang_name = _LANG_NAMES.get(lang, lang)
    instr = _CLASS_INSTR.get(entity_class, "Generate diverse natural utterances.")

    seed_lines = []
    for e in entities[:30]:
        if isinstance(e, dict):
            surface = e.get("surface", "")
            if surface:
                seed_lines.append(f"- {surface}")
    seed_block = "\n".join(seed_lines) if seed_lines else "(none — invent plausible ones)"

    tmpl_block = "\n".join(f"- {t}" for t in templates[:10]) if templates else "(none)"

    sys_msg = (
        "You are an Indic-language data writer. Output STRICT JSONL — one JSON "
        "object per line, no prose, no fences. Each line must have keys: "
        '"text" (the utterance string) and "entities_used" (list of surface '
        "strings actually used).\n"
        f"Target script: {lang_name} primary; English Latin allowed only for "
        "explicit codemix / brands / proper-noun classes."
    )
    user_msg = (
        f"Generate {n_target} natural {lang_name} utterances for the entity "
        f"class '{entity_class}'.\n\n"
        f"Class definition: {instr}\n\n"
        f"Length: {MIN_TOKENS}-{MAX_TOKENS} whitespace tokens per utt.\n"
        "Rules:\n"
        "1. Vary entity surface position (start / middle / end).\n"
        "2. Each utterance must contain at least one entity surface.\n"
        "3. No duplicate utterances.\n"
        "4. Use the seed entities below verbatim where possible; otherwise "
        "invent realistic Indian-context entities.\n\n"
        f"Seed entities:\n{seed_block}\n\n"
        f"Optional templates (with {{e0}} {{e1}} slots):\n{tmpl_block}\n\n"
        "Respond with JSONL only."
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def _parse_jsonl_response(content: str) -> list[dict]:
    """Robustly extract JSONL records from a Qwen response."""
    rows: list[dict] = []
    for raw in content.splitlines():
        raw = raw.strip().strip("`")
        if not raw or raw.startswith("```"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "text" in obj:
            rows.append(obj)
    return rows


def call_llm_for_class(
    lang: str,
    entity_class: str,
    n_target: int = ROWS_PER_LANG_CLASS,
    batch_size: int = 50,
    execute: bool = False,
) -> list[UtteranceRow]:
    """Call Qwen-72B in batches of ``batch_size`` until ``n_target`` rows.

    With ``execute=False`` (default) returns [] and prints what would be sent.
    With ``execute=True`` actually invokes ``evaluation.anthropic_client.chat_complete``
    (Claude Haiku 4.5 — drop-in compatible with the OpenRouter shape).
    """
    entities = load_entity_dict(lang, entity_class)
    templates = load_templates(lang, entity_class)

    cache_path = CACHE_DIR / f"{lang}_{entity_class}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load any cached rows from previous partial runs.
    cached_rows: list[UtteranceRow] = []
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                cached_rows.append(_row_from_dict(d))
            except (json.JSONDecodeError, KeyError):
                continue

    if not execute:
        print(
            f"  [dry-run] {lang}/{entity_class}: cached={len(cached_rows)}, "
            f"would call Claude Haiku 4.5 ~{(n_target - len(cached_rows) + batch_size - 1) // max(batch_size, 1)} "
            f"times to reach {n_target}"
        )
        return cached_rows

    # Real execution path. Anthropic-direct (Haiku 4.5).
    from evaluation.anthropic_client import (  # noqa: WPS433  (import inside function = isolation)
        BudgetExceededError,
        chat_complete,
        extract_content,
    )

    rows: list[UtteranceRow] = list(cached_rows)
    seen_texts: set[str] = {_norm_text(r.text) for r in rows}
    run_id = f"haiku_{int(time.time())}"

    consecutive_failures = 0
    while len(rows) < n_target:
        remaining = n_target - len(rows)
        ask = min(batch_size, remaining)
        msgs = build_generation_prompt(lang, entity_class, entities, templates, ask)
        try:
            resp = chat_complete(
                messages=msgs,
                model=LLM_MODEL,
                temperature=0.8,
                max_tokens=2048,
            )
            consecutive_failures = 0
        except BudgetExceededError as e:
            print(f"  [budget] stopping at {len(rows)} rows: {e}", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001 — network/timeout/anthropic SDK errors
            consecutive_failures += 1
            wait_s = min(2 ** consecutive_failures, 60)
            print(f"  [net-retry] {lang}/{entity_class}: {type(e).__name__} ({e}); "
                  f"sleeping {wait_s}s, attempt {consecutive_failures}", file=sys.stderr)
            if consecutive_failures >= 6:
                print(f"  [net-fatal] {lang}/{entity_class}: 6 consecutive failures, "
                      f"stopping at {len(rows)} rows", file=sys.stderr)
                break
            time.sleep(wait_s)
            continue
        content = extract_content(resp)
        new_rows = _parse_jsonl_response(content)
        for nr in new_rows:
            text = nr.get("text", "").strip()
            if not text:
                continue
            key = _norm_text(text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            row = UtteranceRow(
                id=f"{lang}_{entity_class}_{len(rows):06d}",
                lang=lang,
                entity_class=entity_class,
                text=text,
                source_template="",
                llm_run_id=run_id,
            )
            rows.append(row)
            # Append-write cache from the WRAPPED UtteranceRow so id/lang/
            # entity_class/llm_run_id survive a resume. Earlier versions
            # cached the raw LLM dict (no metadata) — that produced the
            # "27 bad rows" bug observed 2026-04-29.
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"  [{lang}/{entity_class}] {len(rows)}/{n_target} after batch")
    return rows


def _row_from_dict(d: dict) -> UtteranceRow:
    tokens = [
        EntityToken(**t) if isinstance(t, dict) else t
        for t in d.get("entity_tokens", [])
    ]
    return UtteranceRow(
        id=d.get("id", ""),
        lang=d.get("lang", ""),
        entity_class=d.get("entity_class", ""),
        text=d.get("text", ""),
        entity_tokens=tokens,
        source_template=d.get("source_template", ""),
        llm_run_id=d.get("llm_run_id", ""),
    )


def _norm_text(text: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(text.split())).casefold()


# ---------------------------------------------------------------------------
# Step 3 — validators
# ---------------------------------------------------------------------------


def script_purity_ok(text: str, lang: str, allow_latin: bool = False) -> bool:
    """Reject rows whose Indic chars are not in the target script.

    Allow:
      - target script's Unicode block
      - ASCII Latin + digits + punct (always; Indian usage embeds them)
      - common punctuation, whitespace, ZWJ/ZWNJ
    Reject any *other* Indic script char (e.g. Kannada U+0C80-U+0CFF in a
    Telugu row).

    ``allow_latin`` is a no-op flag retained for API stability — Latin is
    always permitted since real Indic text routinely contains ASCII.
    """
    del allow_latin  # Latin is always permitted
    target = SCRIPT_RANGES.get(lang)
    if target is None:
        return True
    lo, hi = target
    for ch in text:
        cp = ord(ch)
        if lo <= cp <= hi:
            continue
        if cp < 0x80:  # ASCII
            continue
        if ch in ("‌", "‍", " "):  # ZWNJ, ZWJ, NBSP
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S") or cat.startswith("Z"):
            continue  # punctuation, symbol, separator
        # Reject any other Indic block (U+0900..U+0DFF excluding target)
        if 0x0900 <= cp <= 0x0DFF and not (lo <= cp <= hi):
            return False
        # Other non-target letters/marks: reject.
        if cat.startswith("L") or cat.startswith("M"):
            return False
    return True


def length_ok(text: str) -> bool:
    n = len(text.split())
    return MIN_TOKENS <= n <= MAX_TOKENS


def entity_presence_ok(
    text: str,
    entity_class: str,
    entities: list[dict] | None = None,
) -> bool:
    """Cheap structural check that *some* entity-like span is present."""
    if entity_class == "digits":
        # Either ASCII digits or Indic digit words. We can't enumerate every
        # word-form, so accept ASCII digit run OR any seed digit-word match.
        if re.search(r"\d{2,}", text):
            return True
        if entities:
            for e in entities:
                surf = e.get("surface", "") if isinstance(e, dict) else ""
                if surf and surf in text:
                    return True
            return False
        return True  # we trust Qwen if we have no seed dict
    if entity_class == "currency":
        # Look for INR markers, lakh/crore/rupees, or any digit run.
        if re.search(r"\d", text):
            return True
        for marker in ("रुपये", "రూపాయ", "ரூபாய", "lakh", "crore", "rupees", "₹"):
            if marker in text:
                return True
        return False
    if entity_class == "addresses":
        return bool(re.search(r"\d{3,}", text))  # PIN or plot number
    if entity_class == "brands":
        return _has_latin_word(text) or (
            entities is not None and any(
                isinstance(e, dict) and e.get("surface", "") in text for e in entities
            )
        )
    if entity_class == "codemix":
        return _has_latin_word(text)
    if entity_class == "proper_nouns":
        # Always pass (hard to validate cheaply); trust Qwen + downstream EHR.
        return True
    return True


def _has_latin_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text))


def validate_row(row: UtteranceRow, entities: list[dict] | None = None) -> tuple[bool, str]:
    """Return ``(ok, reason)``."""
    if not length_ok(row.text):
        return False, "length"
    if not script_purity_ok(row.text, row.lang):
        return False, "script_purity"
    if not entity_presence_ok(row.text, row.entity_class, entities):
        return False, "entity_missing"
    return True, "ok"


# ---------------------------------------------------------------------------
# Step 4 — entity-token tagger
# ---------------------------------------------------------------------------


def entity_token_tagger(text: str, entity_class: str) -> list[EntityToken]:
    """Tag spans of ``text`` that count as entity tokens for EHR.

    Heuristic per class. Used both at corpus-build time and at eval time;
    the same logic must produce the same spans for a given (text, class).
    """
    tokens: list[EntityToken] = []
    if entity_class == "digits":
        for m in re.finditer(r"\d+", text):
            tokens.append(EntityToken(m.group(), m.start(), m.end(), "digit_run"))
        # Indic digit-words handled at eval time via class-specific normalisation.
    elif entity_class == "currency":
        for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
            tokens.append(EntityToken(m.group(), m.start(), m.end(), "amount"))
    elif entity_class == "addresses":
        # PIN code is the most reliable structural anchor.
        for m in re.finditer(r"\b\d{6}\b", text):
            tokens.append(EntityToken(m.group(), m.start(), m.end(), "pincode"))
        for m in re.finditer(r"\b\d{1,4}[A-Za-z]?\b", text):
            if not any(t.start <= m.start() < t.end for t in tokens):
                tokens.append(EntityToken(m.group(), m.start(), m.end(), "house_or_plot"))
    elif entity_class == "brands":
        # Treat each capitalised Latin run as a brand candidate.
        for m in re.finditer(r"[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*", text):
            tokens.append(EntityToken(m.group(), m.start(), m.end(), "brand"))
    elif entity_class == "codemix":
        # Latin-script tokens are the EHR targets.
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9'-]*", text):
            tokens.append(EntityToken(m.group(), m.start(), m.end(), "english_loan"))
    elif entity_class == "proper_nouns":
        # Capitalised Latin runs OR (heuristic) standalone CJK-like Indic
        # tokens of length >=4. We lean on Latin-cap detection and accept
        # that pure-Indic proper nouns will be tagged at gold-set creation
        # time by the human curator.
        for m in re.finditer(r"[A-Z][A-Za-z]+", text):
            tokens.append(EntityToken(m.group(), m.start(), m.end(), "proper_noun"))
    return tokens


def dedupe_rows(rows: Iterable[UtteranceRow]) -> list[UtteranceRow]:
    """Hash dedup on case-folded normalised text."""
    seen: set[str] = set()
    out: list[UtteranceRow] = []
    for r in rows:
        key = _norm_text(r.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Step 5 — synthesis routing (gated)
# ---------------------------------------------------------------------------


def estimate_synth_cost_usd(n_clips: int, system: str) -> float:
    """Rough USD estimate per system. PLAN.md §4 numbers."""
    if system == "praxy_r6":
        # ~30 A10G-hr / 30k clips → ~0.001 USD/clip at $1.10/A10G-hr.
        return 0.0011 * n_clips
    if system == "elevenlabs":
        return 0.0  # free credits
    if system == "cartesia":
        return 0.0  # free credits
    return 0.0


def synthesise_audio(
    text: str,
    system: str,
    voice: str,
    language: str,
    out_path: Path,
    execute: bool = False,
) -> AudioClip | None:
    """Route a single (text, system) call to the right backend.

    ``--dry-run`` prints what would be called and writes nothing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not execute:
        print(
            f"    [dry-run] {system}({voice}, {language}): "
            f"text[{len(text)} chars] -> {out_path.name}"
        )
        return None

    # Real-execution paths import lazily so dry-run doesn't drag deps.
    if system == "elevenlabs":
        from serving.commercial_baselines import elevenlabs_synthesize
        wav_bytes, sr = elevenlabs_synthesize(text, voice="female" if voice == "female" else "male")
    elif system == "cartesia":
        from serving.commercial_baselines import cartesia_synthesize
        wav_bytes, sr = cartesia_synthesize(
            text, voice="female" if voice == "female" else "male", language=language,
        )
    elif system == "praxy_r6":
        # Praxy R6 runs on Modal — actual deploy is invoked separately via
        # `modal run serving/modal_app.py::run_baseline`. This function is
        # NOT the place to spin up Modal containers.
        raise RuntimeError(
            "Praxy R6 synthesis must be driven by `modal run "
            "serving/modal_app.py::run_baseline ...`; cannot be invoked "
            "from this CLI directly. Generate text JSONL here, then drive "
            "Modal separately."
        )
    else:
        raise ValueError(f"unknown system: {system}")

    out_path.write_bytes(wav_bytes)
    return AudioClip(
        clip_id=out_path.stem,
        utterance_id=out_path.stem.split("__")[0],
        lang=language,
        entity_class="",  # filled in by caller
        text=text,
        audio_path=str(out_path),
        system=system,
        voice=voice,
        duration_s=None,
        sample_rate=sr,
    )


# ---------------------------------------------------------------------------
# Step 6 — manifest writer
# ---------------------------------------------------------------------------


def write_manifest(rows: list[UtteranceRow], lang: str, entity_class: str) -> Path:
    """Atomic JSONL write to ``data/stt_flywheel/text/{lang}_{class}.jsonl``."""
    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_TEXT_DIR / f"{lang}_{entity_class}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return out_path


def manifest_writer(corpus_root: Path, out_jsonl: Path) -> int:
    """Walk an audio directory and emit a training manifest JSONL.

    Each manifest row:
        {audio_path, text, lang, entity_class, system, voice,
         entity_tokens, duration_s}

    Returns number of rows written.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for jp in sorted(corpus_root.rglob("*.json")):
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            wav = jp.with_suffix(".wav")
            if not wav.exists():
                continue
            row = {
                "audio_path": str(wav.resolve()),
                "text": meta.get("text", ""),
                "lang": meta.get("lang", ""),
                "entity_class": meta.get("entity_class", ""),
                "system": meta.get("system", ""),
                "voice": meta.get("voice", ""),
                "entity_tokens": meta.get("entity_tokens", []),
                "duration_s": meta.get("duration_s"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _load_plan_yaml(plan_path: Path) -> dict:
    """Load plan YAML or fall back to a default plan dict."""
    if plan_path is None or not plan_path.exists():
        return {
            "langs": list(LANGS),
            "classes": list(ENTITY_CLASSES),
            "rows_per_pair": ROWS_PER_LANG_CLASS,
            "synth_clips_per_lang": SYNTH_CLIPS_PER_LANG,
            "synth_mix": dict(SYNTH_MIX),
        }
    try:
        import yaml  # type: ignore
    except ImportError:
        # Hand-roll a tiny "key: value" parser sufficient for our plan files.
        plan: dict = {}
        for ln in plan_path.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#", 1)[0].strip()
            if not ln or ":" not in ln:
                continue
            k, v = ln.split(":", 1)
            plan[k.strip()] = v.strip()
        return plan
    return yaml.safe_load(plan_path.read_text(encoding="utf-8"))


def build_synth_corpus(plan_yaml: Path | None = None, dry_run: bool = True) -> dict:
    """Top-level orchestrator.

    Phase 1: text generation via Qwen (per (lang, class)).
    Phase 2: audio synthesis fan-out per source (Praxy R6 / 11labs / Cartesia).

    Honours OpenRouter and Modal budget caps. Returns a summary dict.
    """
    plan = _load_plan_yaml(plan_yaml) if plan_yaml else _load_plan_yaml(Path("/nonexistent"))
    langs = tuple(plan.get("langs", LANGS))
    classes = tuple(plan.get("classes", ENTITY_CLASSES))
    rows_per_pair = int(plan.get("rows_per_pair", ROWS_PER_LANG_CLASS))
    clips_per_lang = int(plan.get("synth_clips_per_lang", SYNTH_CLIPS_PER_LANG))
    synth_mix = plan.get("synth_mix", dict(SYNTH_MIX))

    summary: dict = {
        "dry_run": dry_run,
        "langs": list(langs),
        "classes": list(classes),
        "rows_per_pair": rows_per_pair,
        "anthropic_budget_usd": ANTHROPIC_BUDGET_USD,
        "phase1_text": [],
        "phase2_audio": [],
        "estimated_costs_usd": {"anthropic": 0.0, "modal_synth": 0.0},
    }

    # ---- Phase 1: text gen ----
    print(f"\n=== Phase 1: text gen (dry_run={dry_run}) ===")
    total_rows_target = 0
    for lang in langs:
        for cls in classes:
            total_rows_target += rows_per_pair
            entry = {
                "lang": lang,
                "class": cls,
                "rows_target": rows_per_pair,
                "manifest_path": str(OUT_TEXT_DIR / f"{lang}_{cls}.jsonl"),
            }
            summary["phase1_text"].append(entry)
            if dry_run:
                print(f"  would gen {rows_per_pair} rows for {lang}/{cls}")
            else:
                # Skip pairs whose text manifest is already populated to target size.
                # The previous Phase 1 attempt cleaned and verified those manifests;
                # re-running would burn Anthropic credits for no gain.
                existing_manifest = OUT_TEXT_DIR / f"{lang}_{cls}.jsonl"
                if existing_manifest.exists():
                    existing_rows = sum(1 for _ in existing_manifest.read_text(encoding="utf-8").splitlines() if _.strip())
                    # 70% threshold: LLMs hit a diversity ceiling around 1000-1100
                    # rows for a tight entity class, so a 73% yield is "done" in
                    # practice. Anything above this is a complete pair from a
                    # prior run; redoing it would just duplicate-burn credits.
                    if existing_rows >= rows_per_pair * 0.70:
                        print(f"  {lang}/{cls}: SKIP (manifest has {existing_rows} rows, target {rows_per_pair})")
                        entry["rows_kept"] = existing_rows
                        entry["rows_raw"] = existing_rows
                        entry["skipped"] = True
                        continue
                rows = call_llm_for_class(lang, cls, n_target=rows_per_pair, execute=True)
                rows = dedupe_rows(rows)
                # Tag entity spans + validate
                seed = load_entity_dict(lang, cls)
                kept: list[UtteranceRow] = []
                for r in rows:
                    ok, reason = validate_row(r, seed)
                    if not ok:
                        continue
                    r.entity_tokens = entity_token_tagger(r.text, cls)
                    kept.append(r)
                manifest = write_manifest(kept, lang, cls)
                entry["rows_kept"] = len(kept)
                entry["rows_raw"] = len(rows)
                print(f"  {lang}/{cls}: kept {len(kept)}/{len(rows)} -> {manifest}")
    # Rough Anthropic Haiku 4.5 cost estimate: ~50 utts per call × ~400 prompt tok + ~800 completion tok.
    # Haiku 4.5 pricing: $1/Mtok input, $5/Mtok output (per evaluation/anthropic_client.py MODEL_PRICES).
    n_calls = (total_rows_target + 49) // 50
    est_or = n_calls * (400 / 1e6 * 1.00 + 800 / 1e6 * 5.00)
    summary["estimated_costs_usd"]["anthropic"] = round(est_or, 4)

    # ---- Phase 2: audio synth ----
    print(f"\n=== Phase 2: synth audio (dry_run={dry_run}) ===")
    for lang in langs:
        for system, share in synth_mix.items():
            n = int(round(clips_per_lang * share))
            cost = estimate_synth_cost_usd(n, system)
            summary["phase2_audio"].append(
                {"lang": lang, "system": system, "clips": n, "est_cost_usd": round(cost, 2)}
            )
            summary["estimated_costs_usd"]["modal_synth"] += cost
            print(f"  {lang}/{system}: {n} clips, ~${cost:.2f}")
            if not dry_run and system in ("elevenlabs", "cartesia"):
                # We do NOT actually iterate 50k synth calls inside this CLI;
                # production runs invoke `serving.commercial_baselines.run_baseline`
                # in a separate driver to respect rate limits + retry logic.
                print(
                    f"  [{system}] real-run path not wired here — invoke "
                    f"`uv run python -m serving.commercial_baselines "
                    f"--provider {system} --test-set <generated_test_set>`"
                )
    summary["estimated_costs_usd"]["modal_synth"] = round(
        summary["estimated_costs_usd"]["modal_synth"], 2,
    )

    # ---- Sanity: total cost guard ----
    total = summary["estimated_costs_usd"]["anthropic"] + summary["estimated_costs_usd"]["modal_synth"]
    print(f"\nEstimated total cost: ${total:.2f} "
          f"(Anthropic Haiku 4.5 ${summary['estimated_costs_usd']['anthropic']:.2f} + "
          f"Modal ${summary['estimated_costs_usd']['modal_synth']:.2f})")
    if total > 100.0:
        print("WARNING: estimated total > $100; review PLAN.md budget before --execute")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="STT-flywheel entity-dense corpus pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build", help="run the corpus build pipeline")
    bp.add_argument("--plan", type=Path, default=None, help="YAML plan file")
    bp.add_argument("--execute", action="store_true",
                    help="ACTUALLY call APIs (default: dry-run)")

    tp = sub.add_parser("text", help="run only Phase 1 (text gen) for one (lang, class)")
    tp.add_argument("--lang", required=True, choices=list(LANGS))
    tp.add_argument("--class", dest="cls", required=True, choices=list(ENTITY_CLASSES))
    tp.add_argument("--n", type=int, default=ROWS_PER_LANG_CLASS)
    tp.add_argument("--execute", action="store_true")

    mp = sub.add_parser("manifest", help="walk an audio dir and write training manifest")
    mp.add_argument("--corpus-root", type=Path, required=True)
    mp.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()

    if args.cmd == "build":
        summary = build_synth_corpus(args.plan, dry_run=not args.execute)
        print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.cmd == "text":
        rows = call_llm_for_class(args.lang, args.cls, n_target=args.n, execute=args.execute)
        if args.execute:
            kept = []
            seed = load_entity_dict(args.lang, args.cls)
            for r in rows:
                ok, _ = validate_row(r, seed)
                if ok:
                    r.entity_tokens = entity_token_tagger(r.text, args.cls)
                    kept.append(r)
            path = write_manifest(kept, args.lang, args.cls)
            print(f"wrote {len(kept)} rows -> {path}")
        else:
            print(f"[dry-run] would generate {args.n} rows for {args.lang}/{args.cls}")
    elif args.cmd == "manifest":
        n = manifest_writer(args.corpus_root, args.out)
        print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
