import pytest

from pcbad.report import inject_section

DOC = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n"


def test_inject_replaces_block_and_keeps_rest():
    out = inject_section(DOC, "RESULTS", "| a | b |")
    assert "old" not in out and "| a | b |" in out
    assert out.startswith("intro\n") and out.endswith("outro\n")


def test_inject_is_idempotent():
    once = inject_section(DOC, "RESULTS", "X")
    assert inject_section(once, "RESULTS", "X") == once
    assert inject_section(once, "RESULTS", "Y").count("Y") == 1


def test_missing_or_misordered_markers_raise():
    with pytest.raises(ValueError):
        inject_section("no markers here", "RESULTS", "X")
    with pytest.raises(ValueError):
        inject_section("<!-- RESULTS:END -->\n<!-- RESULTS:START -->", "RESULTS", "X")
