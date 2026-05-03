"""Unit tests for pure-logic helpers in data_pipeline + modal_stt_train.

No paid-API calls. Run:
    uv run python -m paper.stt_flywheel.test_data_pipeline
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from paper.stt_flywheel.data_pipeline import (
    EntityToken,
    UtteranceRow,
    _norm_text,
    dedupe_rows,
    entity_presence_ok,
    entity_token_tagger,
    length_ok,
    manifest_writer,
    script_purity_ok,
    validate_row,
)
from paper.stt_flywheel.modal_stt_train import entity_hit_rate, fleurs_regression_delta

PASS = "PASS"
FAIL = "FAIL"


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{PASS if ok else FAIL}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def test_script_purity() -> bool:
    ok = True
    # Valid Telugu — passes
    ok &= _check(
        "telugu pure",
        script_purity_ok("నా మొబైల్ నంబర్ 9876543210", "te"),
    )
    # Telugu with Hindi/Devanagari — fails
    ok &= _check(
        "telugu rejects devanagari",
        not script_purity_ok("నా name है రాము", "te"),
    )
    # Telugu with Kannada — fails (R1 Tamil-bleed analogue)
    ok &= _check(
        "telugu rejects kannada",
        not script_purity_ok("నా ಕನ್ನಡ word", "te"),
    )
    # Hindi pure
    ok &= _check(
        "hindi pure",
        script_purity_ok("मेरा नंबर है 9876543210", "hi"),
    )
    # Tamil pure
    ok &= _check(
        "tamil pure",
        script_purity_ok("என் பெயர் ராஜ்", "ta"),
    )
    # Codemix Latin allowed always
    ok &= _check(
        "telugu allows latin",
        script_purity_ok("మా CEO meeting ఇవాళ", "te"),
    )
    return ok


def test_length() -> bool:
    ok = True
    ok &= _check("length ok 5 tokens", length_ok("one two three four five"))
    ok &= _check("length too short", not length_ok("hi"))
    ok &= _check("length too long", not length_ok(" ".join(["x"] * 30)))
    return ok


def test_entity_presence() -> bool:
    ok = True
    ok &= _check(
        "digits via numeric run",
        entity_presence_ok("call 9876543210 now please ok", "digits"),
    )
    ok &= _check(
        "currency via lakh marker",
        entity_presence_ok("salary five lakh per annum yes", "currency"),
    )
    ok &= _check(
        "addresses via 6-digit pin",
        entity_presence_ok("flat 4 banjara hills hyderabad 500034", "addresses"),
    )
    ok &= _check(
        "brands via latin word",
        entity_presence_ok("Swiggy లో biryani order చేశాను", "brands"),
    )
    ok &= _check(
        "codemix needs latin",
        entity_presence_ok("మా CEO meeting ఇవాళ jaye", "codemix"),
    )
    ok &= _check(
        "codemix rejects pure indic",
        not entity_presence_ok("నేను ఇంటికి వెళ్తున్నాను ఇవాళ", "codemix"),
    )
    return ok


def test_entity_token_tagger() -> bool:
    ok = True
    # digits class — finds digit run
    toks = entity_token_tagger("call 9876543210 today", "digits")
    ok &= _check("digit tagger finds run", len(toks) == 1 and toks[0].surface == "9876543210")

    # codemix — finds Latin tokens
    toks = entity_token_tagger("మా CEO meeting ఇవాళ", "codemix")
    surfaces = [t.surface for t in toks]
    ok &= _check("codemix finds CEO + meeting", "CEO" in surfaces and "meeting" in surfaces)

    # addresses — pin code wins
    toks = entity_token_tagger("flat 4 banjara hyderabad 500034", "addresses")
    has_pin = any(t.type == "pincode" and t.surface == "500034" for t in toks)
    ok &= _check("addresses tags pincode", has_pin)

    # brands — capitalised Latin
    toks = entity_token_tagger("I love Flipkart and Amazon", "brands")
    surfaces = [t.surface for t in toks]
    ok &= _check("brands finds Flipkart", any(s.startswith("Flipkart") for s in surfaces))
    return ok


def test_dedupe() -> bool:
    rows = [
        UtteranceRow(id="a", lang="te", entity_class="digits", text="hello world"),
        UtteranceRow(id="b", lang="te", entity_class="digits", text="HELLO WORLD"),
        UtteranceRow(id="c", lang="te", entity_class="digits", text="something else"),
    ]
    deduped = dedupe_rows(rows)
    return _check("dedup case-insensitive", len(deduped) == 2)


def test_validate_row() -> bool:
    ok = True
    good = UtteranceRow(
        id="t1", lang="te", entity_class="digits",
        text="call 9876543210 now please ok bye",
    )
    is_ok, _ = validate_row(good)
    ok &= _check("validate good row", is_ok)

    bad_script = UtteranceRow(
        id="t2", lang="te", entity_class="digits",
        text="मेरा 9876543210 नंबर है यहाँ देखो",
    )
    is_ok, reason = validate_row(bad_script)
    ok &= _check("validate rejects script mix", not is_ok and reason == "script_purity")

    too_short = UtteranceRow(id="t3", lang="te", entity_class="digits", text="ఏమి")
    is_ok, reason = validate_row(too_short)
    ok &= _check("validate rejects short", not is_ok and reason == "length")
    return ok


def test_manifest_writer() -> bool:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "audio"
        root.mkdir()
        # synth a tiny structure: clip01.json + clip01.wav
        meta = {
            "text": "hello", "lang": "te", "entity_class": "digits",
            "system": "praxy_r6", "voice": "female", "entity_tokens": [],
            "duration_s": 1.2,
        }
        (root / "clip01.json").write_text(json.dumps(meta), encoding="utf-8")
        (root / "clip01.wav").write_bytes(b"RIFF0000WAVE")
        out = Path(td) / "manifest.jsonl"
        n = manifest_writer(root, out)
        rows = [json.loads(ln) for ln in out.read_text().splitlines()]
        return _check(
            "manifest writes clip",
            n == 1 and len(rows) == 1 and rows[0]["text"] == "hello",
        )


def test_entity_hit_rate() -> bool:
    ok = True
    tokens = [{"surface": "9876543210", "type": "digit_run"}]
    ehr = entity_hit_rate(tokens, "my number is 9876543210 ok")
    ok &= _check("EHR full hit", abs(ehr - 1.0) < 1e-9)
    ehr = entity_hit_rate(tokens, "my number is unknown")
    ok &= _check("EHR miss", ehr == 0.0)
    tokens = [
        {"surface": "Flipkart", "type": "brand"},
        {"surface": "Amazon", "type": "brand"},
    ]
    ehr = entity_hit_rate(tokens, "i love flipkart but not the other")
    ok &= _check("EHR partial 50%", abs(ehr - 0.5) < 1e-9)
    return ok


def test_fleurs_delta() -> bool:
    return _check(
        "fleurs delta arithmetic",
        abs(fleurs_regression_delta(0.10, 0.115) - 0.015) < 1e-9,
    )


def test_norm_text() -> bool:
    return _check(
        "norm text dedup key",
        _norm_text("Hello   World") == _norm_text("hello world"),
    )


def main() -> int:
    print("=== STT-flywheel pure-logic tests ===\n")
    suites = [
        ("script_purity", test_script_purity),
        ("length", test_length),
        ("entity_presence", test_entity_presence),
        ("entity_token_tagger", test_entity_token_tagger),
        ("dedupe", test_dedupe),
        ("validate_row", test_validate_row),
        ("manifest_writer", test_manifest_writer),
        ("entity_hit_rate", test_entity_hit_rate),
        ("fleurs_delta", test_fleurs_delta),
        ("norm_text", test_norm_text),
    ]
    all_ok = True
    for name, fn in suites:
        print(f"[{name}]")
        if not fn():
            all_ok = False
        print()
    print("=" * 40)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
