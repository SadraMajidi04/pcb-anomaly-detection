"""Static checks that catch typos, missing imports and stale docs WITHOUT running torch.

These run everywhere (even where PyTorch/Gradio are not installed) and cover the
modules whose runtime tests need those libraries.
"""
import ast
import builtins
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pcbad"
PY_FILES = sorted([*SRC.glob("*.py"), *(ROOT / "scripts").glob("*.py"),
                   ROOT / "app.py", *(ROOT / "tests").glob("*.py")])


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def test_all_python_files_parse():
    assert PY_FILES, "no python files found"
    for p in PY_FILES:
        _tree(p)


def _internal_imports(path):
    """Yield (module_name, [imported names]) for imports of pcbad modules."""
    for n in ast.walk(_tree(path)):
        if isinstance(n, ast.ImportFrom):
            if n.level == 1 and path.parent == SRC and n.module:
                yield n.module, [a.name for a in n.names]
            elif n.level == 0 and n.module and n.module.startswith("pcbad."):
                yield n.module.split(".", 1)[1], [a.name for a in n.names]


def test_internal_imports_resolve_to_defined_names():
    for p in PY_FILES:
        for mod, names in _internal_imports(p):
            target = SRC / f"{mod}.py"
            assert target.exists(), f"{p.name}: imports missing module pcbad.{mod}"
            defined = _top_level_names(_tree(target))
            for name in names:
                assert name in defined, f"{p.name}: '{name}' is not defined in pcbad.{mod}"


def _undefined_names(tree):
    bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                if arg is not None:
                    bound.add(arg.arg)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        if isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return sorted({n.id for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                   and n.id not in bound})


def test_no_undefined_names_anywhere():
    for p in PY_FILES:
        missing = _undefined_names(_tree(p))
        assert not missing, f"{p.relative_to(ROOT)}: undefined names {missing}"


def test_scripts_have_main_and_entry_guard():
    scripts = [*(ROOT / "scripts").glob("*.py"), ROOT / "app.py"]
    assert len(scripts) >= 5
    for p in scripts:
        tree = _tree(p)
        assert "main" in _top_level_names(tree), f"{p.name} has no main()"
        assert "__main__" in p.read_text(encoding="utf-8"), f"{p.name} lacks __main__ guard"


def _cli_flags(script: Path):
    flags = set()
    for n in ast.walk(_tree(script)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            flags |= {a.value for a in n.args
                      if isinstance(a, ast.Constant) and str(a.value).startswith("--")}
    return flags


def test_readme_commands_use_existing_scripts_and_flags():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    found = 0
    for m in re.finditer(r"python (scripts/\w+\.py|app\.py)([^\n#`]*)", readme):
        script = ROOT / m.group(1)
        assert script.exists(), f"README mentions missing {m.group(1)}"
        for flag in re.findall(r"--[a-z][a-z0-9-]*", m.group(2)):
            assert flag in _cli_flags(script), f"README: {m.group(1)} has no flag {flag}"
        found += 1
    assert found >= 5
    for rel in re.findall(r"(?:bash )(scripts/\w+\.sh)", readme):
        assert (ROOT / rel).exists()


def test_run_all_script_calls_only_existing_scripts():
    text = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    called = re.findall(r"python (scripts/\w+\.py)", text)
    assert called
    for rel in called:
        assert (ROOT / rel).exists(), rel


def test_third_party_imports_are_declared_in_requirements():
    pkg = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
           "PIL": "pillow", "sklearn": "scikit-learn", "gradio": "gradio"}
    req = (ROOT / "requirements.txt").read_text().lower() + \
          (ROOT / "requirements-demo.txt").read_text().lower()
    stdlib = getattr(sys, "stdlib_module_names", None)
    used = set()
    for p in [*SRC.glob("*.py"), *(ROOT / "scripts").glob("*.py"), ROOT / "app.py"]:
        for n in ast.walk(_tree(p)):
            if isinstance(n, ast.Import):
                used |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                used.add(n.module.split(".")[0])
    used -= {"pcbad"}
    for mod in used:
        if mod in pkg:
            assert pkg[mod] in req, f"{mod} is imported but '{pkg[mod]}' is not in requirements"
        elif stdlib is not None:
            assert mod in stdlib, f"unknown third-party import {mod!r}: add it to requirements"


def test_pyproject_and_package_layout():
    text = (ROOT / "pyproject.toml").read_text()
    assert 'name = "pcbad"' in text and 'where = ["src"]' in text
    assert (SRC / "__init__.py").exists()


def test_ci_workflow_runs_pytest():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    steps = [s for job in cfg["jobs"].values() for s in job["steps"]]
    assert any("pytest" in s.get("run", "") for s in steps)
    assert any("requirements.txt" in s.get("run", "") for s in steps)


def test_readme_has_result_markers_and_license_exists():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("RESULTS", "ROBUSTNESS"):
        assert f"<!-- {name}:START -->" in readme and f"<!-- {name}:END -->" in readme
    assert (ROOT / "LICENSE").exists() and (ROOT / ".gitignore").exists()


def test_gitignore_excludes_data_and_checkpoints():
    text = (ROOT / ".gitignore").read_text()
    for pattern in ("data/", "outputs/", "*.pt", "__pycache__/"):
        assert pattern in text
