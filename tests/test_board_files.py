from vf import cli


def _board():
    points = [{"name": "htdemucs", "s": 1.0, "q": 11.0}, {"name": "htdemucs-lite-8k", "s": 4.0, "q": 10.1}]
    return {"points": points, "floor_q": 10.0, "pot": 1000.0, "area": {"maintainers": 2.0}}


def test_payout_rows_price_a_gap_and_skip_entries_that_are_gone():
    rows = cli.payout_rows(_board())
    assert rows.startswith("| new entry | units | paid |")
    assert "lite-8k, twice as fast" in rows
    assert "16×, 10.5 dB" in rows
    assert "WASM" not in rows


def test_a_rewrite_only_touches_what_is_between_the_markers(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("before\n<!-- vf board writes this table -->\nold\n<!-- end of table -->\nafter\n")
    monkeypatch.setattr(cli.M, "ROOT", str(tmp_path))
    cli._write_readme_table(_board())
    text = (tmp_path / "README.md").read_text()
    assert "old" not in text and text.startswith("before\n") and text.endswith("<!-- end of table -->\nafter\n")
