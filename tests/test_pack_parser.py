from __future__ import annotations

from imdb_app.pack_parser import parse_pack_text


def test_parse_multipack_weight_syntax():
    parsed = parse_pack_text("TAPOK PREMIUM BLACK TEA 2GX100PCS BOX")

    assert parsed.normalized_weight == "2G"
    assert parsed.pack_count == 100
    assert parsed.addons == "100PCS 2G"


def test_parse_promotional_pack_syntax_derives_each_weight():
    parsed = parse_pack_text("ZESTA STRAWBERRY 25+7 FREE TEABAG 57.6G BOX ENVELOPE")

    assert parsed.normalized_weight == "1.8G"
    assert parsed.pack_count == 32
    assert parsed.promotion == "7 FREE"
    assert parsed.addons == "ENVELOPE"


def test_parse_piece_count_weight_syntax():
    parsed = parse_pack_text("1PCS 2G")

    assert parsed.normalized_weight == "2G"
    assert parsed.pack_count == 1
    assert parsed.addons == "1PCS 2G"
