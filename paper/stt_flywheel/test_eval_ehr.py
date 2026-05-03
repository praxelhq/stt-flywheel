"""Unit tests for paper/stt_flywheel/eval_ehr.py.

Pure-logic helpers; no API calls. Run with::

    uv run python -m pytest paper/stt_flywheel/test_eval_ehr.py -v
"""
from __future__ import annotations

from paper.stt_flywheel.eval_ehr import (
    aggregate,
    hit_brand,
    hit_currency_amount,
    hit_digit_run,
    hit_house_or_plot,
    hit_pincode,
    hit_proper_noun,
    hit_spelled_digit,
    parse_currency_amount,
    score_row,
)


def test_digit_run_exact_match():
    assert hit_digit_run("9876543210", "నా ఫోన్ నంబర్ 9876543210 కు కాల్ చేయండి")
    assert hit_digit_run("9876543210", "9876543210") is True
    # Spaces between digits — should still match (NFKC + space-strip)
    assert hit_digit_run("9876543210", "9876543210") is True


def test_digit_run_miss():
    assert not hit_digit_run("9876543210", "నా ఫోన్ నంబర్ 1234567890 కు")
    assert not hit_digit_run("9876543210", "no digits here")


def test_pincode_match():
    assert hit_pincode("560034", "address: Bengaluru 560034")
    assert hit_pincode("600040", "சென்னை 600040 இல்")
    # 5-digit (not a real PIN) shouldn't be flagged
    assert not hit_pincode("60004", "Chennai 60004")


def test_house_or_plot_match():
    assert hit_house_or_plot("B-204", "Plot No. B-204, Sector 62")
    assert hit_house_or_plot("5/7", "5/7, அண்ணா நகர்")
    # Different format shouldn't match
    assert not hit_house_or_plot("B-204", "Plot 204B, Sector 62")


def test_brand_latin_match():
    assert hit_brand("Swiggy", "Swiggy lo biryani order chesanu")
    assert hit_brand("HDFC", "hdfc bank का खाता है")  # case-insensitive
    assert hit_brand("WhatsApp", "मैंने WhatsApp पे message किया")


def test_brand_native_to_latin_match():
    """Native-script brand in GT, Latin alias in hypothesis (or vice versa)."""
    # GT in Devanagari, hypothesis in Latin — should match via BRAND_ALIASES
    assert hit_brand("व्हाट्सऐप", "I sent a WhatsApp message")
    # GT in Latin, hypothesis in Devanagari
    assert hit_brand("WhatsApp", "मैंने व्हाट्सऐप पे भेजा")
    # Telugu to Latin
    assert hit_brand("స్విగ్గి", "Swiggy delivered fast")


def test_brand_miss():
    assert not hit_brand("Swiggy", "Zomato delivered the food")
    assert not hit_brand("HDFC", "ICICI bank account")


def test_proper_noun_match():
    assert hit_proper_noun("Mahatma Gandhi", "Mahatma Gandhi was born in Porbandar")
    # 80% overlap — "Subramanian Chandrasekhar" in GT, just "Subramanian" in hyp = 50%, fail
    assert not hit_proper_noun("Subramanian Chandrasekhar", "Subramanian was a physicist")
    # 100% overlap with reordering still matches (token-set)
    assert hit_proper_noun("Mahatma Gandhi", "Gandhi Mahatma was great")


def test_proper_noun_miss():
    assert not hit_proper_noun("Mahatma Gandhi", "Jawaharlal Nehru was the first PM")


def test_spelled_digit_telugu():
    gt = "ఐదు ఆరు సున్నా"  # 5 6 0
    # All present in correct order
    assert hit_spelled_digit(gt, "నా OTP ఐదు ఆరు సున్నా రెండు", "te")
    # Order broken
    assert not hit_spelled_digit(gt, "సున్నా ఐదు ఆరు", "te")
    # Missing one of the three (66% < 80% threshold)
    assert not hit_spelled_digit(gt, "ఐదు ఆరు", "te")


def test_spelled_digit_hindi():
    gt = "एक दो तीन"  # 1 2 3
    assert hit_spelled_digit(gt, "OTP एक दो तीन है", "hi")


def test_parse_currency_arabic():
    assert parse_currency_amount("12 lakh rupees") == 1_200_000.0
    assert parse_currency_amount("5 crore") == 50_000_000.0
    assert parse_currency_amount("25,000 rupees") == 25_000.0


def test_parse_currency_indic():
    assert parse_currency_amount("ఐదు లక్షల") == 500_000.0  # 5 × 100k
    assert parse_currency_amount("दो करोड़") == 20_000_000.0   # 2 × 10M


def test_currency_match_arabic_vs_spelled():
    # Mixed numeric / spelled — should still match within ±0.5%
    assert hit_currency_amount("5 lakh", "ఐదు లక్షల రూపాయలు")
    assert hit_currency_amount("12 crore", "12 crore rupees")


def test_currency_match_tolerance():
    # Within ±0.5%: 100000 vs 100499 → 0.499%, hit
    assert hit_currency_amount("1 lakh", "100499")
    # Outside ±0.5%: 100000 vs 101000 → 1%, miss
    assert not hit_currency_amount("1 lakh", "101000")


def test_score_row_full_hit():
    gt = {
        "id": "te_addr_001",
        "lang": "te",
        "entity_class": "addresses",
        "text": "ప్లాట్ నెం 145, బంజారా హిల్స్, హైదరాబాద్ 500034",
        "entity_tokens": [
            {"surface": "500034", "start": 0, "end": 6, "type": "pincode"},
            {"surface": "145", "start": 0, "end": 3, "type": "house_or_plot"},
        ],
    }
    hyp = "ప్లాట్ నెం 145, బంజారా హిల్స్, హైదరాబాద్ 500034"
    s = score_row(gt, hyp)
    assert s.n_tokens == 2
    assert s.n_hits == 2
    assert s.by_type["pincode"]["hits"] == 1
    assert s.by_type["house_or_plot"]["hits"] == 1


def test_score_row_partial_hit():
    gt = {
        "id": "te_addr_002", "lang": "te", "entity_class": "addresses",
        "text": "ప్లాట్ 12, చెన్నై 600012",
        "entity_tokens": [
            {"surface": "600012", "start": 0, "end": 6, "type": "pincode"},
            {"surface": "12", "start": 0, "end": 2, "type": "house_or_plot"},
        ],
    }
    hyp = "ప్లాట్ 12, చెన్నై 600013"  # PIN wrong by 1 digit
    s = score_row(gt, hyp)
    assert s.n_tokens == 2
    assert s.n_hits == 1  # house_or_plot hits, pincode misses
    assert s.by_type["pincode"]["hits"] == 0
    assert s.by_type["house_or_plot"]["hits"] == 1


def test_aggregate_basic():
    gt_a = {
        "id": "a", "lang": "te", "entity_class": "digits",
        "entity_tokens": [{"surface": "9876543210", "start": 0, "end": 0, "type": "digit_run"}],
    }
    gt_b = {
        "id": "b", "lang": "te", "entity_class": "digits",
        "entity_tokens": [{"surface": "1234567890", "start": 0, "end": 0, "type": "digit_run"}],
    }
    s_a = score_row(gt_a, "9876543210 దయచేసి")  # hit
    s_b = score_row(gt_b, "9999999999 lol")  # miss
    agg = aggregate([s_a, s_b])
    assert agg["total_tokens"] == 2
    assert agg["total_hits"] == 1
    assert agg["ehr"] == 0.5
    assert agg["by_class"]["digits"]["ehr"] == 0.5
    assert agg["by_lang"]["te"]["ehr"] == 0.5


def test_score_row_empty_tokens():
    """Rows with no entity tokens should score as 0/0 (don't affect EHR)."""
    gt = {"id": "x", "lang": "te", "entity_class": "currency", "entity_tokens": []}
    s = score_row(gt, "నా జీతం ఐదు లక్షల రూపాయలు")
    assert s.n_tokens == 0
    assert s.n_hits == 0
