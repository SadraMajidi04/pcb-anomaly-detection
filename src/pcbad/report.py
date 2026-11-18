"""Turn evaluation results into a Markdown table + JSON file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _fmt(v, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def results_to_markdown(results: Dict[str, dict]) -> str:
    head = ("| Model | Image AUROC | Pixel AUROC | Precision | Recall | F1 | Latency (ms/img) |\n"
            "|---|---|---|---|---|---|---|")
    rows = []
    for name, m in results.items():
        lat = (m.get("latency") or {}).get("mean_ms")
        rows.append(
            f"| {name} | {_fmt(m.get('image_auroc'))} | {_fmt(m.get('pixel_auroc'))} | "
            f"{_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | {_fmt(m.get('f1'))} | "
            f"{_fmt(lat, 1)} |"
        )
    return "\n".join([head, *rows]) + "\n"


def save_results(results: Dict[str, dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "results.md").write_text(results_to_markdown(results), encoding="utf-8")


def inject_section(text: str, name: str, content: str) -> str:
    """Replace the block between `<!-- NAME:START -->` and `<!-- NAME:END -->`.

    Idempotent: running it again with new content just swaps the block.
    """
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    i, j = text.find(start), text.find(end)
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"markers {start} ... {end} not found (in this order)")
    return text[: i + len(start)] + "\n" + content.strip() + "\n" + text[j:]
