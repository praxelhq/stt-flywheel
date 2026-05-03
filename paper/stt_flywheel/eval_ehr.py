"""Entity-Hit-Rate scorer — the headline metric for the STT-flywheel paper.

Implements the per-class normalisation rules pre-registered in
``paper/stt_flywheel/PLAN.md §7``:

  - digit_run     : NFKC-normalised exact match against any extracted
                    digit_run in hypothesis.
  - spelled_digit : ≥80% of spelled digits in the language's vocabulary
                    appear in correct order in the hypothesis.
  - currency_amount: numeric value (after parsing "5 lakh" /
                    "ఐదు లక్షల") within ±0.5% of GT.
  - pincode       : exact match (6-digit Indian PIN).
  - house_or_plot : NFKC + case-fold match.
  - brand         : case-folded match in either Latin or native script
                    via ``BRAND_ALIASES``.
  - proper_noun   : token-set match with ≥80% overlap (allows
                    transliteration variance).

The hypothesis-side parser uses the **same** ``entity_token_tagger`` that
the data pipeline used at training time, so both sides see consistent
spans.

Headline aggregation:
    EHR = sum(hits_per_row) / sum(total_entity_tokens_per_row)

CLI::

    uv run python -m paper.stt_flywheel.eval_ehr score \\
        --gt data/stt_flywheel/text/te_digits.jsonl \\
        --hyp /path/to/transcripts.jsonl \\
        [--per-class]

Tests::

    uv run python -m pytest paper/stt_flywheel/test_eval_ehr.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Importing entity_token_tagger here keeps the hypothesis-side parser
# identical to the GT-side parser used in data_pipeline.
from paper.stt_flywheel.data_pipeline import entity_token_tagger  # noqa: E402

# Spelled-digit dictionaries (used for digits-class scoring) — same source
# of truth as `clean_corpus.SPELLED_DIGITS`.
SPELLED_DIGITS = {
    "te": ("సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"),
    "ta": ("பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"),
    "hi": ("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"),
}

# Indic multipliers for currency parsing.
INDIC_MULTIPLIERS = {
    "lakh": 100_000, "lakhs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000,
    "thousand": 1_000, "hundred": 100,
    "లక్ష": 100_000, "లక్షల": 100_000, "లక్షలు": 100_000,
    "కోటి": 10_000_000, "కోట్ల": 10_000_000, "కోట్లు": 10_000_000,
    "வேயிரம்": 1_000, "லட்சம்": 100_000, "லட்சங்கள்": 100_000, "கோடி": 10_000_000,
    "हज़ार": 1_000, "हजार": 1_000, "लाख": 100_000, "करोड़": 10_000_000,
}

# Brand alias table — maps native-script brand names to a canonical Latin form.
# Extends as the corpus grows; for v1 covers the brands seeded in
# `stt/data/entities/brands/ta.jsonl` and the most common Hi/Te variants.
BRAND_ALIASES: dict[str, str] = {
    # Latin canonical forms (lowercase)
    "swiggy": "swiggy", "zomato": "zomato",
    "flipkart": "flipkart", "amazon": "amazon",
    "myntra": "myntra", "paytm": "paytm",
    "phonepe": "phonepe", "google pay": "google pay", "googlepay": "google pay",
    "gpay": "google pay", "bhim": "bhim", "bhim upi": "bhim",
    "hdfc": "hdfc", "icici": "icici", "sbi": "sbi", "axis": "axis", "kotak": "kotak",
    "tata": "tata", "maruti": "maruti", "mahindra": "mahindra",
    "ola": "ola", "uber": "uber", "rapido": "rapido",
    "whatsapp": "whatsapp", "instagram": "instagram", "facebook": "facebook",
    "youtube": "youtube", "netflix": "netflix", "hotstar": "hotstar",
    "amazon prime": "amazon prime", "prime video": "amazon prime",
    "jio": "jio", "airtel": "airtel", "bsnl": "bsnl",
    # Native-script (Hindi)
    "स्विगी": "swiggy", "ज़ोमैटो": "zomato", "जोमैटो": "zomato",
    "फ्लिपकार्ट": "flipkart", "अमेज़न": "amazon", "अमेज़ॉन": "amazon",
    "पेटीएम": "paytm", "फोनपे": "phonepe", "गूगल पे": "google pay",
    "व्हाट्सऐप": "whatsapp", "व्हाट्सएप": "whatsapp",
    "उबर": "uber", "ओला": "ola", "जिओ": "jio",
    # Native-script (Telugu)
    "స్విగ్గి": "swiggy", "జొమాటో": "zomato",
    "ఫ్లిప్‌కార్ట్": "flipkart", "అమెజాన్": "amazon",
    "పేటిఎం": "paytm", "ఫోన్‌పే": "phonepe",
    "వాట్సాప్": "whatsapp",
    "ఉబర్": "uber", "ఓలా": "ola",
    # Native-script (Tamil)
    "ஸ்விக்கி": "swiggy", "ஜொமேட்டோ": "zomato",
    "ஃபிளிப்கார்ட்": "flipkart", "அமேசான்": "amazon",
    "பேடிஎம்": "paytm", "ஃபோன்பே": "phonepe",
    "வாட்ஸ்அப்": "whatsapp",
    "உபர்": "uber", "ஓலா": "ola",
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    """NFKC + lowercase + collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).strip().casefold().split())


def _digits_only(text: str) -> str:
    """Extract a contiguous run of ASCII digits."""
    return re.sub(r"\D+", "", text)


# ---------------------------------------------------------------------------
# Per-class hit functions — return True/False per (gt_token, hyp_text)
# ---------------------------------------------------------------------------


def hit_digit_run(gt_surface: str, hyp_text: str) -> bool:
    target = _digits_only(gt_surface)
    if not target:
        return False
    # A hit if the digit string appears in the hypothesis (NFKC normalised).
    hyp_n = unicodedata.normalize("NFKC", hyp_text)
    return target in hyp_n.replace(" ", "")


def hit_pincode(gt_surface: str, hyp_text: str) -> bool:
    target = _digits_only(gt_surface)
    if len(target) != 6:
        return False
    hyp_n = unicodedata.normalize("NFKC", hyp_text).replace(" ", "")
    return target in hyp_n


def hit_house_or_plot(gt_surface: str, hyp_text: str) -> bool:
    target = _norm(gt_surface)
    return target in _norm(hyp_text)


def hit_brand(gt_surface: str, hyp_text: str) -> bool:
    """Brand hit: gt_surface (native or Latin) maps to canonical Latin via
    BRAND_ALIASES; check whether ANY surface form of that brand appears in
    the hypothesis."""
    canonical = BRAND_ALIASES.get(_norm(gt_surface)) or _norm(gt_surface)
    hyp_norm = _norm(hyp_text)
    # Check Latin canonical form
    if canonical in hyp_norm:
        return True
    # Check whether any alias (in any script) that maps to this canonical
    # appears in the hypothesis.
    for alias, can in BRAND_ALIASES.items():
        if can == canonical and alias in hyp_norm:
            return True
    return False


def hit_proper_noun(gt_surface: str, hyp_text: str) -> bool:
    """Token-set match with ≥80% overlap (allows transliteration variance)."""
    gt_tokens = set(_norm(gt_surface).split())
    if not gt_tokens:
        return False
    hyp_tokens = set(_norm(hyp_text).split())
    overlap = len(gt_tokens & hyp_tokens)
    return overlap / len(gt_tokens) >= 0.80


def hit_spelled_digit(gt_surface: str, hyp_text: str, lang: str) -> bool:
    """≥80% of spelled digits in the GT appear in correct order in the
    hypothesis. Order check uses sub-sequence matching."""
    spelled_words = SPELLED_DIGITS.get(lang, ())
    if not spelled_words:
        return False
    gt_seq = [w for w in gt_surface.split() if w in spelled_words]
    hyp_seq = [w for w in hyp_text.split() if w in spelled_words]
    if not gt_seq:
        return False
    # Check sub-sequence preservation
    j = 0
    matches = 0
    for w in hyp_seq:
        if j < len(gt_seq) and w == gt_seq[j]:
            matches += 1
            j += 1
    return matches / len(gt_seq) >= 0.80


def parse_currency_amount(text: str) -> float | None:
    """Parse 'ఐదు లక్షల', '5 lakh', '12 crore', etc. to a float numeric value.
    Returns None if no number can be extracted."""
    t = _norm(text)
    # Find a numeric component (Arabic digits or spelled digits)
    num_match = re.search(r"\d[\d,.]*", t)
    base: float | None = None
    if num_match:
        base = float(num_match.group(0).replace(",", ""))
    else:
        # Try spelled digits in any language
        for lang, spelled in SPELLED_DIGITS.items():
            for i, w in enumerate(spelled):
                if w in t:
                    base = float(i)
                    break
            if base is not None:
                break
    if base is None:
        return None
    # Apply multiplier if present
    for word, mult in INDIC_MULTIPLIERS.items():
        if word in t:
            return base * mult
    return base


def hit_currency_amount(gt_surface: str, hyp_text: str) -> bool:
    gt_v = parse_currency_amount(gt_surface)
    if gt_v is None:
        return False
    hyp_v = parse_currency_amount(hyp_text)
    if hyp_v is None:
        return False
    if gt_v == 0:
        return abs(hyp_v) < 0.5
    return abs(hyp_v - gt_v) / abs(gt_v) <= 0.005  # ±0.5%


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------


HIT_FNS = {
    "digit_run": hit_digit_run,
    "pincode": hit_pincode,
    "house_or_plot": hit_house_or_plot,
    "brand": hit_brand,
    "proper_noun": hit_proper_noun,
    "currency_amount": hit_currency_amount,
}


@dataclass
class RowScore:
    row_id: str
    lang: str
    entity_class: str
    n_tokens: int
    n_hits: int
    by_type: dict


def score_row(gt_row: dict, hyp_text: str) -> RowScore:
    tokens = gt_row.get("entity_tokens", []) or []
    by_type: dict = {}
    n_hits = 0
    for tok in tokens:
        ttype = tok.get("type", "")
        surface = tok.get("surface", "")
        if ttype == "spelled_digit":
            ok = hit_spelled_digit(surface, hyp_text, gt_row.get("lang", ""))
        elif ttype in HIT_FNS:
            ok = HIT_FNS[ttype](surface, hyp_text)
        else:
            # Unknown type — fall back to NFKC + casefold substring match
            ok = _norm(surface) in _norm(hyp_text)
        by_type.setdefault(ttype, {"hits": 0, "total": 0})
        by_type[ttype]["total"] += 1
        if ok:
            by_type[ttype]["hits"] += 1
            n_hits += 1
    return RowScore(
        row_id=gt_row.get("id", ""),
        lang=gt_row.get("lang", ""),
        entity_class=gt_row.get("entity_class", ""),
        n_tokens=len(tokens),
        n_hits=n_hits,
        by_type=by_type,
    )


def aggregate(scores: list[RowScore]) -> dict:
    total_tokens = sum(s.n_tokens for s in scores)
    total_hits = sum(s.n_hits for s in scores)
    ehr = total_hits / total_tokens if total_tokens else 0.0
    by_class: dict = {}
    by_lang: dict = {}
    by_type: dict = {}
    for s in scores:
        c = s.entity_class
        l = s.lang
        by_class.setdefault(c, {"hits": 0, "total": 0})
        by_class[c]["hits"] += s.n_hits
        by_class[c]["total"] += s.n_tokens
        by_lang.setdefault(l, {"hits": 0, "total": 0})
        by_lang[l]["hits"] += s.n_hits
        by_lang[l]["total"] += s.n_tokens
        for t, agg in s.by_type.items():
            by_type.setdefault(t, {"hits": 0, "total": 0})
            by_type[t]["hits"] += agg["hits"]
            by_type[t]["total"] += agg["total"]
    return {
        "ehr": round(ehr, 4),
        "n_rows": len(scores),
        "total_tokens": total_tokens,
        "total_hits": total_hits,
        "by_class": {c: {**v, "ehr": round(v["hits"] / v["total"], 4) if v["total"] else 0.0} for c, v in by_class.items()},
        "by_lang": {l: {**v, "ehr": round(v["hits"] / v["total"], 4) if v["total"] else 0.0} for l, v in by_lang.items()},
        "by_type": {t: {**v, "ehr": round(v["hits"] / v["total"], 4) if v["total"] else 0.0} for t, v in by_type.items()},
    }


# -----------------------------------------------------------------------------
# Script Fidelity Rate (SFR) — secondary primary metric per the *Script Collapse
# in Multilingual ASR* concurrent work (2026). Measures the fraction of output
# characters in the expected script over total non-whitespace characters.
# Vanilla Whisper-v3 fails this hard for Te (transcribes Te audio as Kannada
# script). Our per-language LoRA + decoder-prefix design fixes both EHR and SFR.
# -----------------------------------------------------------------------------

# Unicode block ranges (start, end inclusive) per Indic script.
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "te": [(0x0C00, 0x0C7F)],                       # Telugu
    "ta": [(0x0B80, 0x0BFF)],                       # Tamil
    "hi": [(0x0900, 0x097F)],                       # Devanagari
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],     # Basic Latin letters
}


def _in_script(ch: str, lang: str) -> bool:
    cp = ord(ch)
    for lo, hi in _SCRIPT_RANGES.get(lang, []):
        if lo <= cp <= hi:
            return True
    return False


def script_fidelity_rate(text: str, lang: str) -> float:
    """Fraction of letter characters in `text` that fall in `lang`'s expected
    script. Whitespace, digits, and punctuation are excluded from both
    numerator and denominator. Empty / no-letter text returns 1.0 (no script
    failure to count). Score is in [0, 1]; 1.0 = no script collapse.
    """
    if not text:
        return 1.0
    letters_total = 0
    letters_in_script = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters_total += 1
        if _in_script(ch, lang):
            letters_in_script += 1
    if letters_total == 0:
        return 1.0
    return letters_in_script / letters_total


def score_jsonl_pair(gt_path: Path, hyp_path: Path) -> dict:
    """Score a (ground-truth, hypothesis) JSONL pair.

    Both files are expected to be JSONL with rows containing at least an
    ``id`` field. GT rows must have ``entity_tokens``. Hyp rows have
    ``id`` + ``hypothesis`` (the STT-output text).
    """
    gt_rows = {json.loads(l)["id"]: json.loads(l)
               for l in gt_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    hyp_rows = {json.loads(l)["id"]: json.loads(l)["hypothesis"]
                for l in hyp_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    scores = []
    missing = 0
    sfr_per_lang: dict[str, list[float]] = {}
    for rid, gt in gt_rows.items():
        hyp_text = hyp_rows.get(rid)
        if hyp_text is None:
            missing += 1
            continue
        scores.append(score_row(gt, hyp_text))
        lang = gt.get("lang", "")
        if lang:
            sfr_per_lang.setdefault(lang, []).append(
                script_fidelity_rate(hyp_text, lang)
            )
    summary = aggregate(scores)
    summary["missing_hypotheses"] = missing
    summary["sfr_by_lang"] = {
        lang: round(sum(vals) / len(vals), 4) if vals else None
        for lang, vals in sfr_per_lang.items()
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score", help="Score a (gt, hyp) JSONL pair.")
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--hyp", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "score":
        s = score_jsonl_pair(args.gt, args.hyp)
        print(json.dumps(s, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
