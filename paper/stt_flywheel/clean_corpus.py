"""Drop the truly-bad rows flagged by audit_corpus.py.

Mutates the text/*.jsonl files in place; saves .bak copies.
Non-issues like ``no_entity_tokens_warn`` are NOT dropped (those are
spelled-out-number utterances with no numeric span — legitimate content).

Drops:
- Foreign Indic script bleed (Devanagari/Tamil chars in Telugu, etc.)
- Missing target script entirely
- Latin chars > 8 in non-codemix classes (English-Indic mash)
- digits class missing both numeric digits AND spelled-out number lemmas
- Apology / format leakage in the LLM output
- Empty / too-short / too-long texts
- Empty metadata (id/lang/entity_class)
- Within-file duplicates
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data" / "stt_flywheel" / "text"

SCRIPT_BLOCKS = {
    "te": (0x0C00, 0x0C7F),
    "ta": (0x0B80, 0x0BFF),
    "hi": (0x0900, 0x097F),
}
FOREIGN_INDIC_BLOCKS = {
    "te": [(0x0900, 0x097F), (0x0B80, 0x0BFF), (0x0C80, 0x0CFF)],
    "ta": [(0x0900, 0x097F), (0x0C00, 0x0C7F), (0x0D00, 0x0D7F)],
    "hi": [(0x0C00, 0x0C7F), (0x0B80, 0x0BFF)],
}
SPELLED_DIGITS = {
    "te": ("సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"),
    "ta": ("பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"),
    "hi": ("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"),
}

APOLOGY_RE = re.compile(r"(?i)\b(I'm sorry|I cannot|I can't|here are|here is|sure, here|of course|certainly)\b")
LEAKAGE_RE = re.compile(r"^\s*[\[\{\"`]|\boutput[: ]|^[0-9]+[.)] |^- ", re.IGNORECASE)


def _block_count(text: str, lo: int, hi: int) -> int:
    return sum(1 for c in text if lo <= ord(c) <= hi)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(text.split())).casefold()


def is_bad(row: dict) -> str | None:
    """Return the reason this row is bad, or None if clean."""
    text = row.get("text", "")
    lang = row.get("lang", "")
    cls = row.get("entity_class", "")

    if not row.get("id") or not lang or not cls:
        return "empty_metadata"
    n_tok = len(text.split())
    if n_tok == 0:
        return "empty_text"
    if n_tok < 3:
        return "too_short"
    if n_tok > 25:
        return "too_long"
    if APOLOGY_RE.search(text):
        return "apology_leakage"
    if LEAKAGE_RE.match(text):
        return "format_leakage"
    if lang not in SCRIPT_BLOCKS:
        return "unknown_lang"

    # Foreign Indic bleed
    for lo, hi in FOREIGN_INDIC_BLOCKS[lang]:
        if _block_count(text, lo, hi) > 0:
            return "foreign_indic_script"

    target_lo, target_hi = SCRIPT_BLOCKS[lang]
    has_target = _block_count(text, target_lo, target_hi) > 0

    # Latin chars in non-codemix. Brands class is allowed more Latin because
    # brand names ARE often English-script ("Amazon Prime मेंबरशिप ले लो").
    # The validator must reject pure-English brand utterances (no target
    # script) but accept Indic-with-English-brand-name.
    if cls != "codemix":
        latin = sum(1 for c in text if "A" <= c <= "Z" or "a" <= c <= "z")
        latin_cap = 30 if cls == "brands" else 8  # brands: tolerate brand names; others: strict
        if latin > latin_cap:
            return "excess_latin_in_indic"
        if not has_target:
            return "missing_target_script"

    # Codemix: must have Latin (English words). Either pure-Latin transliterated
    # ("Aaj ka meeting postpone kar denge kya?") or mixed-script
    # ("sir, oka minute, report ko download చేయాలి") is valid — both are
    # real Indian speech patterns. Earlier "must have both Indic and Latin"
    # rule rejected pure-Latin transliterated codemix, which is wrong.
    if cls == "codemix":
        has_latin = any("A" <= c <= "Z" or "a" <= c <= "z" for c in text)
        if not has_latin:
            return "codemix_no_latin"

    # digits class needs digits or spelled-out
    if cls == "digits":
        has_arabic = bool(re.search(r"\d", text))
        has_spelled = any(w in text for w in SPELLED_DIGITS.get(lang, ()))
        if not (has_arabic or has_spelled):
            return "digits_no_digits"

    return None


def clean_file(p: Path) -> dict:
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen: set[str] = set()
    good: list[dict] = []
    drop_reasons: Counter = Counter()
    dups = 0
    for r in rows:
        bad = is_bad(r)
        if bad:
            drop_reasons[bad] += 1
            continue
        n = _norm(r["text"])
        if n in seen:
            dups += 1
            drop_reasons["duplicate_within_file"] += 1
            continue
        seen.add(n)
        good.append(r)

    if drop_reasons:
        # Renumber ids deterministically post-drop
        if good:
            lang = good[0]["lang"]
            cls = good[0]["entity_class"]
            for i, r in enumerate(good):
                r["id"] = f"{lang}_{cls}_{i:06d}"
        bak = p.with_suffix(".jsonl.bak2")
        if not bak.exists():
            shutil.copy(p, bak)
        with open(p, "w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "file": p.name,
        "before": len(rows),
        "after": len(good),
        "dropped": dict(drop_reasons),
    }


def main() -> int:
    files = sorted(TEXT_DIR.glob("*.jsonl"))
    print(f"=== Cleaning {len(files)} files ===")
    total_before = 0
    total_after = 0
    aggregate: Counter = Counter()
    for p in files:
        r = clean_file(p)
        total_before += r["before"]
        total_after += r["after"]
        aggregate.update(r["dropped"])
        dropped = r["before"] - r["after"]
        if dropped:
            reasons = ", ".join(f"{k}={v}" for k, v in r["dropped"].items())
            print(f"  {r['file']}: {r['before']} -> {r['after']}  (-{dropped}: {reasons})")
        else:
            print(f"  {r['file']}: {r['before']} (clean)")
    print(f"\n=== Total: {total_before} -> {total_after} ({total_before - total_after} dropped) ===")
    if aggregate:
        print("Drop reasons (aggregate):")
        for k, v in aggregate.most_common():
            print(f"  {k:32s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
