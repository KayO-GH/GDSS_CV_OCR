from __future__ import annotations

from pathlib import Path

from app import available_sample_groups, load_sample_group


def test_available_sample_groups_reads_hackathon_style_names(tmp_path: Path):
    (tmp_path / "S1_1.jpg").write_bytes(b"1")
    (tmp_path / "S1_2.jpg").write_bytes(b"2")
    (tmp_path / "S2_1.jpg").write_bytes(b"3")
    (tmp_path / "ignore.txt").write_bytes(b"x")

    assert available_sample_groups(tmp_path) == ["S1", "S2"]


def test_load_sample_group_returns_sorted_payloads(tmp_path: Path):
    (tmp_path / "S1_2.jpg").write_bytes(b"2")
    (tmp_path / "S1_1.jpg").write_bytes(b"1")

    payloads = load_sample_group("S1", tmp_path)

    assert [payload.filename for payload in payloads] == ["S1_1.jpg", "S1_2.jpg"]
    assert [payload.image_bytes for payload in payloads] == [b"1", b"2"]
