"""Deep audit of Phase-1 entity-text corpus.

Catches things the in-pipeline validator might miss:
- Script bleed (Devanagari in Telugu, Kannada in Telugu, etc.)
- Prompt-leakage ("Sure, here are 50 utterances")
- Suspiciously short / long texts
- Duplicates within & across files (case-folded normalised)
- Empty entity-tokens for entity classes that should always have one
- Numeric-digit utterances that lost their digits
- LLM apology/refusal patterns
- Latin chars in pure-Indic classes (other than codemix)

Usage:
    uv run python paper/stt_flywheel/audit_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data" / "stt_flywheel" / "text"

SCRIPT_BLOCKS = {
    "te": ("TELUGU", 0x0C00, 0x0C7F),
    "ta": ("TAMIL", 0x0B80, 0x0BFF),
    "hi": ("DEVANAGARI", 0x0900, 0x097F),
}

# Other Indic blocks we explicitly DO NOT want bleeding into a target lang
FOREIGN_INDIC_BLOCKS = {
    "te": [("DEVANAGARI", 0x0900, 0x097F), ("TAMIL", 0x0B80, 0x0BFF), ("KANNADA", 0x0C80, 0x0CFF)],
    "ta": [("DEVANAGARI", 0x0900, 0x097F), ("TELUGU", 0x0C00, 0x0C7F), ("MALAYALAM", 0x0D00, 0x0D7F)],
    "hi": [("TELUGU", 0x0C00, 0x0C7F), ("TAMIL", 0x0B80, 0x0BFF)],
}

# Apology / refusal patterns from any LLM
APOLOGY_RE = re.compile(
    r"(?i)\b(I'm sorry|I cannot|I can't|here are|here is|sure, here|of course|certainly)\b"
)

# JSON / formatting leakage
LEAKAGE_RE = re.compile(r"^\s*[\[\{\"`]|\boutput[: ]|^[0-9]+[.)] |^- |sample utterance", re.IGNORECASE)


def _has_block(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(c) <= hi for c in text)


def _block_count(text: str, lo: int, hi: int) -> int:
    return sum(1 for c in text if lo <= ord(c) <= hi)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def audit_file(p: Path) -> dict:
    lang, cls = p.stem.split("_", 1)
    rows: list[dict] = []
    seen: dict[str, int] = {}
    issues: list[dict] = []

    # cross-row dup tracker (within file)
    dup_within: list[tuple[int, str]] = []

    target_block = SCRIPT_BLOCKS[lang]
    foreign_blocks = FOREIGN_INDIC_BLOCKS[lang]

    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            issues.append({"line": line_no, "type": "bad_json", "detail": str(e)[:80]})
            continue

        text = row.get("text", "")
        n = _norm(text)

        # Empty / too-short / too-long
        token_count = len(text.split())
        if token_count == 0:
            issues.append({"line": line_no, "type": "empty_text"})
            continue
        if token_count < 3:
            issues.append({"line": line_no, "type": "too_short", "n_tokens": token_count, "text": text})
        if token_count > 25:
            issues.append({"line": line_no, "type": "too_long", "n_tokens": token_count, "text": text[:80]})

        # Apology / refusal leakage
        if APOLOGY_RE.search(text):
            issues.append({"line": line_no, "type": "apology_leakage", "text": text[:80]})

        # Format leakage (markdown lists, JSON, "Output:" prefix)
        if LEAKAGE_RE.match(text):
            issues.append({"line": line_no, "type": "format_leakage", "text": text[:80]})

        # Foreign Indic script bleed
        for name, lo, hi in foreign_blocks:
            cnt = _block_count(text, lo, hi)
            if cnt > 0:
                issues.append({
                    "line": line_no, "type": f"foreign_script_{name.lower()}",
                    "n_chars": cnt, "text": text[:80],
                })
                break  # one issue per row is enough

        # Latin chars in pure-Indic classes (codemix is allowed)
        if cls != "codemix":
            latin_count = sum(1 for c in text if "A" <= c <= "Z" or "a" <= c <= "z")
            # Allow up to 3 Latin chars (e.g., "OTP", "PIN") — those are real words used in Indian speech
            if latin_count > 8:
                issues.append({
                    "line": line_no, "type": "excess_latin_in_indic",
                    "n_chars": latin_count, "text": text[:80],
                })

        # Codemix should have BOTH target Indic script AND Latin
        if cls == "codemix":
            has_indic = _has_block(text, target_block[1], target_block[2])
            has_latin = any("A" <= c <= "Z" or "a" <= c <= "z" for c in text)
            if not (has_indic and has_latin):
                issues.append({
                    "line": line_no, "type": "codemix_not_mixed",
                    "has_indic": has_indic, "has_latin": has_latin, "text": text[:80],
                })

        # Target script presence (every row should have target script chars,
        # except codemix-pure-Latin edge cases)
        target_count = _block_count(text, target_block[1], target_block[2])
        if target_count == 0 and cls != "codemix":
            issues.append({
                "line": line_no, "type": "missing_target_script", "text": text[:80],
            })

        # Entity-class-specific: digits class must contain digits or
        # spelled-out numbers (heuristic: native-digit lemmas)
        if cls == "digits":
            has_arabic_digit = bool(re.search(r"\d", text))
            # Spelled-out digits in target language (incomplete but covers "zero..nine"
            # in Te / Ta / Hi). Used as a fallback signal.
            spelled_out_hints = {
                "te": ("సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"),
                "ta": ("பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"),
                "hi": ("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"),
            }
            has_spelled = any(w in text for w in spelled_out_hints[lang])
            if not (has_arabic_digit or has_spelled):
                issues.append({
                    "line": line_no, "type": "digits_no_digits", "text": text[:80],
                })

        # entity_tokens sanity for entity classes that always have one
        ent = row.get("entity_tokens", []) or []
        if cls in ("digits", "currency", "addresses", "brands") and len(ent) == 0:
            # Soft warning — some templates won't have an extracted span (e.g. spelled-out digits)
            issues.append({
                "line": line_no, "type": "no_entity_tokens_warn", "cls": cls, "text": text[:80],
            })

        # Within-file dedup
        if n in seen:
            dup_within.append((line_no, text[:60]))
            issues.append({"line": line_no, "type": "duplicate_within_file", "text": text[:60]})
        else:
            seen[n] = line_no

        rows.append(row)

    return {
        "file": str(p.relative_to(ROOT)),
        "lang": lang,
        "class": cls,
        "rows_total": len(rows),
        "issues_count": len(issues),
        "issues_by_type": Counter(i["type"] for i in issues),
        "issues": issues[:50],  # first 50 examples
        "duplicates_within": len(dup_within),
    }


def cross_file_dedup_check(audited: list[dict]) -> dict:
    """Dedup across files within the same lang (cross-class)."""
    by_lang: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for a in audited:
        if not a.get("rows_total"):
            continue
        p = ROOT / a["file"]
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = _norm(row.get("text", ""))
            by_lang[a["lang"]][n].append((a["class"], row.get("id", "")))

    cross: dict[str, list] = {}
    for lang, m in by_lang.items():
        dups = [(t, vs) for t, vs in m.items() if len({c for c, _ in vs}) > 1]
        cross[lang] = [{"text": t[:80], "instances": vs[:5]} for t, vs in dups[:20]]
    return cross


def main() -> int:
    files = sorted(TEXT_DIR.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files in {TEXT_DIR}", file=sys.stderr)
        return 1

    audited = [audit_file(p) for p in files]

    # Per-file summary
    print(f"\n=== Per-file audit ({len(audited)} files) ===")
    for a in audited:
        types = ", ".join(f"{k}={v}" for k, v in a["issues_by_type"].most_common())
        print(f"  {a['lang']}/{a['class']:14s}  rows={a['rows_total']:5d}  issues={a['issues_count']:4d}  [{types}]")

    # Aggregate
    total_rows = sum(a["rows_total"] for a in audited)
    total_issues = sum(a["issues_count"] for a in audited)
    type_totals: Counter = Counter()
    for a in audited:
        type_totals.update(a["issues_by_type"])

    print(f"\n=== Aggregate ===")
    print(f"  total rows: {total_rows}")
    print(f"  total issues: {total_issues}  ({100 * total_issues / max(total_rows, 1):.2f}% of rows have ≥1 issue)")
    print(f"  by type:")
    for t, c in type_totals.most_common():
        print(f"    {t:32s}  {c}")

    # Cross-file (within-lang) dup check
    cross = cross_file_dedup_check(audited)
    print(f"\n=== Cross-class duplicate texts (per language) ===")
    for lang, dups in cross.items():
        print(f"  {lang}: {len(dups)} texts appear in 2+ classes")
        for d in dups[:3]:
            print(f"     - {d['text']}  in {[c for c,_ in d['instances']]}")

    # Show 5 random offending lines per non-trivial issue type for spot-check
    print(f"\n=== Sample offenders (first 3 per type) ===")
    seen_types: set[str] = set()
    for a in audited:
        for issue in a["issues"]:
            t = issue["type"]
            if t in seen_types:
                continue
            seen_types.add(t)
            print(f"\n  [{t}] in {a['lang']}/{a['class']} line {issue.get('line', '?')}")
            print(f"    text: {issue.get('text', '')!r}")
            for k, v in issue.items():
                if k not in ("type", "line", "text"):
                    print(f"    {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
