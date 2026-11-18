"""Copy real numbers from outputs/ into README.md (between the marker comments).

    python scripts/update_readme.py
"""
import argparse
from pathlib import Path

from pcbad.report import inject_section


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--readme", default="README.md")
    p.add_argument("--outputs", default="outputs")
    a = p.parse_args()

    out = Path(a.outputs)
    readme = Path(a.readme)
    text = readme.read_text(encoding="utf-8")
    changed = False

    results = out / "results.md"
    if results.exists():
        text = inject_section(text, "RESULTS", results.read_text(encoding="utf-8"))
        changed = True

    blocks = []
    for model in ("patchcore", "autoencoder"):
        f = out / f"robustness_{model}.md"
        if f.exists():
            blocks.append(f"**{model}**\n\n{f.read_text(encoding='utf-8')}")
    if blocks:
        text = inject_section(text, "ROBUSTNESS", "\n".join(blocks))
        changed = True

    if not changed:
        raise SystemExit(f"Nothing found in {out}/. Run scripts/evaluate.py and robustness.py first.")
    readme.write_text(text, encoding="utf-8")
    print(f"Updated {readme}")


if __name__ == "__main__":
    main()
