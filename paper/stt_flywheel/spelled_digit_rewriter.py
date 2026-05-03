"""Rewrite STT manifests to replace digit-string entities with spelled digits.

Problem (caught 2026-05-01): every TTS system reads "54235" as quantity form
("five lakh forty-two thousand..."). For digit_run, pincode, phone, etc., real
users pronounce digit-by-digit. Training Whisper on the quantity-form audio
teaches the wrong mapping.

Fix: replace digit-string surfaces in the TEXT with spelled-out digit-by-digit
form in the target script. Keep entity_tokens[].surface as the original digit
string for EHR matching at eval time (matched after Whisper output is
digit-recovered by an inverse map).

Rewrites apply to entity types:
  - pincode      (always)
  - phone        (always)
  - digit_run    (always — OTPs, account numbers)
  - spelled_digit (already spelled — skip)
  - house_or_plot (only if length >= 4 — "Plot 56" stays as "Plot fifty-six";
                   "Plot 5600" becomes "Plot ఐదు ఆరు సున్నా సున్నా")
  - currency_amount (skip — quantity reading is correct)

CLI::

    # dry-run: show 5 example rewrites per lang
    uv run python -m paper.stt_flywheel.spelled_digit_rewriter preview --lang te

    # rewrite the manifest in place + emit a list of rows that need re-synth
    uv run python -m paper.stt_flywheel.spelled_digit_rewriter rewrite --execute
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "data" / "stt_flywheel" / "manifests"
TEXT_DIR = ROOT / "data" / "stt_flywheel" / "text"

# Per-language digit → spelled-word lookup. ASCII keys '0'..'9' AND native
# script keys. Whitespace separator between digits in output.
DIGIT_SPELL: dict[str, dict[str, str]] = {
    "te": {
        "0": "సున్నా", "1": "ఒకటి", "2": "రెండు", "3": "మూడు", "4": "నాలుగు",
        "5": "ఐదు", "6": "ఆరు", "7": "ఏడు", "8": "ఎనిమిది", "9": "తొమ్మిది",
        "౦": "సున్నా", "౧": "ఒకటి", "౨": "రెండు", "౩": "మూడు", "౪": "నాలుగు",
        "౫": "ఐదు", "౬": "ఆరు", "౭": "ఏడు", "౮": "ఎనిమిది", "౯": "తొమ్మిది",
    },
    "hi": {
        "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
        "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
        "०": "शून्य", "१": "एक", "२": "दो", "३": "तीन", "४": "चार",
        "५": "पाँच", "६": "छह", "७": "सात", "८": "आठ", "९": "नौ",
    },
    "ta": {
        "0": "பூஜ்யம்", "1": "ஒன்று", "2": "இரண்டு", "3": "மூன்று", "4": "நான்கு",
        "5": "ஐந்து", "6": "ஆறு", "7": "ஏழு", "8": "எட்டு", "9": "ஒன்பது",
        "௦": "பூஜ்யம்", "௧": "ஒன்று", "௨": "இரண்டு", "௩": "மூன்று", "௪": "நான்கு",
        "௫": "ஐந்து", "௬": "ஆறு", "௭": "ஏழு", "௮": "எட்டு", "௯": "ஒன்பது",
    },
}

# Token types that get the digit-spelling treatment.
SPELL_TOKEN_TYPES = {"pincode", "phone", "digit_run"}
SHORT_OK_TYPES = {"house_or_plot"}  # rewrite ONLY if surface ≥ 4 chars
SHORT_OK_MIN_LEN = 4


def _spell_digit_string(s: str, lang: str) -> str:
    """Map each digit char in s to its spelled word in lang. Non-digit chars
    pass through unchanged (so e.g., 'AB1234' becomes 'AB ఒకటి రెండు మూడు
    నాలుగు' for te). Leading/trailing whitespace trimmed."""
    table = DIGIT_SPELL.get(lang, {})
    parts: list[str] = []
    for ch in s.strip():
        if ch in table:
            parts.append(table[ch])
        elif ch.isspace():
            continue  # drop intra-token whitespace
        else:
            parts.append(ch)
    return " ".join(parts).strip()


def _should_rewrite_token(tok: dict) -> bool:
    typ = tok.get("type", "")
    surf = (tok.get("surface") or "").strip()
    if not surf:
        return False
    if typ in SPELL_TOKEN_TYPES:
        return True
    if typ in SHORT_OK_TYPES and len(surf) >= SHORT_OK_MIN_LEN:
        return True
    return False


def rewrite_row(row: dict) -> tuple[dict, bool]:
    """Return (new_row, changed). Replaces relevant token surfaces in row['text']
    with their spelled-digit form. Keeps entity_tokens[].surface as-is (EHR
    matches against the original digit string at eval; the eval-time inverse
    map handles spelled→digit recovery)."""
    lang = row.get("lang", "")
    if lang not in DIGIT_SPELL:
        return row, False
    text = row.get("text", "")
    tokens = row.get("entity_tokens") or []
    candidates = [t for t in tokens if _should_rewrite_token(t)]
    if not candidates:
        return row, False
    # Replace longer surfaces first so substring overlap doesn't break offsets.
    candidates.sort(key=lambda t: -len(t.get("surface", "")))
    new_text = text
    changed = False
    for t in candidates:
        surf = t["surface"].strip()
        spelled = _spell_digit_string(surf, lang)
        if spelled and spelled != surf and surf in new_text:
            # Only replace ONE occurrence (the canonical entity surface).
            new_text = new_text.replace(surf, spelled, 1)
            changed = True
    if not changed:
        return row, False
    new_row = dict(row)
    new_row["text"] = new_text
    new_row["text_original"] = text  # keep audit trail
    return new_row, True


def _iter_manifest(p: Path) -> Iterable[dict]:
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            continue


def cmd_preview(args):
    for lang in ("te", "ta", "hi"):
        if args.lang and args.lang != lang:
            continue
        p = MANIFEST_DIR / f"{lang}_train.jsonl"
        print(f"\n=== {lang} ===")
        n_shown = 0
        for row in _iter_manifest(p):
            new_row, changed = rewrite_row(row)
            if not changed:
                continue
            print(f"  before: {row['text']}")
            print(f"  after : {new_row['text']}")
            print(f"  tokens: {[(t.get('surface'), t.get('type')) for t in row.get('entity_tokens', [])]}")
            print()
            n_shown += 1
            if n_shown >= 5:
                break


def cmd_rewrite(args):
    if not args.execute:
        print("DRY-RUN. Pass --execute to write rewrites + the resynth list.")
    total_rewritten = 0
    total_seen = 0
    resynth_paths: list[str] = []
    for lang in ("te", "ta", "hi"):
        p = MANIFEST_DIR / f"{lang}_train.jsonl"
        out_rows = []
        n_changed = 0
        for row in _iter_manifest(p):
            total_seen += 1
            new_row, changed = rewrite_row(row)
            if changed:
                n_changed += 1
                resynth_paths.append(new_row["audio_path"])
            out_rows.append(new_row)
        total_rewritten += n_changed
        print(f"  {lang}: {n_changed} of {len(out_rows)} rows rewritten")
        if args.execute:
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    print(f"\nTOTAL: {total_rewritten} of {total_seen} rows rewritten")
    if args.execute:
        resynth_path = ROOT / "data" / "stt_flywheel" / "resynth_list.txt"
        resynth_path.write_text("\n".join(resynth_paths) + "\n")
        print(f"Wrote {len(resynth_paths)} paths to {resynth_path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("preview")
    sp.add_argument("--lang", choices=("te", "ta", "hi"), default=None)
    sp = sub.add_parser("rewrite")
    sp.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.cmd == "preview":
        cmd_preview(args)
    elif args.cmd == "rewrite":
        cmd_rewrite(args)


if __name__ == "__main__":
    main()
