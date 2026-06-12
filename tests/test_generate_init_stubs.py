"""Regression tests for the ``GSEGUtils.generate_init_stubs`` AST stub generator.

Covers TEST-06 (Plan 06-02 D-09): four fixture-driven round-trip tests over
the supported ``_lazy_map`` schemas:

1. ``test_generator_string_only_lazy_map`` — bare ``_lazy_map = {...}``
   assignment with string-valued entries (``ast.Assign`` branch).
2. ``test_generator_tuple_form_lazy_map`` — PEP 526 annotated
   ``_lazy_map: dict[str, str | tuple[str, str]] = {...}`` mixing tuple-form
   rename entries and string passthroughs (``ast.AnnAssign`` branch).
3. ``test_generator_empty_eager_init`` — no ``_lazy_map`` at all; pure eager
   submodule imports (verifies the generator does not silently no-op on
   empty maps — RESEARCH Pitfall 5).
4. ``test_generator_pep526_annotated_consistency`` — PEP 526 annotated map
   mixing both inner schemas; exercises the ``ast.AnnAssign`` code path.

Each test writes a fixture ``__init__.py`` (plus a small backing submodule
so the generator can resolve real names) to a per-test ``tmp_path``,
invokes ``python -m GSEGUtils.generate_init_stubs <fixture_dir>`` via
``subprocess.run`` for full isolation, then asserts on the emitted
``__init__.pyi`` content.
"""

import subprocess
import sys
from pathlib import Path


def _run_generator(fixture_dir: Path) -> subprocess.CompletedProcess[str]:
    """Invoke ``python -m GSEGUtils.generate_init_stubs <fixture_dir>``.

    Uses ``check=True`` so a non-zero exit propagates as ``CalledProcessError``;
    captures stdout/stderr for diagnostic inclusion in test assertion messages.
    """
    return subprocess.run(
        [sys.executable, "-m", "GSEGUtils.generate_init_stubs", str(fixture_dir), "--overwrite"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_generator_string_only_lazy_map(tmp_path: Path) -> None:
    """TEST-06 / D-09 schema #1: bare ``_lazy_map = {...}`` with string values.

    Exercises the ``ast.Assign`` branch of ``parse_ast`` (the
    ``isinstance(node, ast.Assign)`` block at generate_init_stubs.py:115).
    Asserts the generator emits a ``from .submod import FooClass, bar_fn``
    line (lexicographic order of public names per ``sorted(set(simples))``).
    """
    fixture_dir = tmp_path / "fix"
    fixture_dir.mkdir()
    (fixture_dir / "__init__.py").write_text(
        '"""Fixture package (string-only _lazy_map)."""\n'
        '__all__ = ["sub"]\n'
        '_lazy_map = {"FooClass": "submod", "bar_fn": "submod"}\n'
        "__all__ = __all__ + list(_lazy_map)\n",
        encoding="utf-8",
    )
    (fixture_dir / "submod.py").write_text(
        "class FooClass:\n    pass\n\n\ndef bar_fn() -> None:\n    pass\n",
        encoding="utf-8",
    )
    (fixture_dir / "sub.py").write_text("", encoding="utf-8")

    result = _run_generator(fixture_dir)
    pyi_path = fixture_dir / "__init__.pyi"
    assert pyi_path.exists() and pyi_path.stat().st_size > 0, (
        f"Generator silently no-op'd on string-only fixture. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    emitted = pyi_path.read_text(encoding="utf-8")
    # The string-only branch emits a single grouped import from the backing module.
    assert "from .submod import FooClass, bar_fn" in emitted, (
        f"Generator did not emit the grouped string-form import for FooClass + bar_fn.\nEmitted:\n{emitted}"
    )
    # Both names land in __all__.
    assert "'FooClass'" in emitted and "'bar_fn'" in emitted, (
        f"Generator omitted FooClass or bar_fn from __all__. Emitted:\n{emitted}"
    )
    # Submodule from __all__ is also emitted.
    assert "from . import sub as sub" in emitted, (
        f"Generator omitted the eager submodule import for 'sub'. Emitted:\n{emitted}"
    )
    # The __getattr__ NoReturn sentinel is present (mypy-strict drift guard).
    assert "def __getattr__(name: str) -> NoReturn:" in emitted


def test_generator_tuple_form_lazy_map(tmp_path: Path) -> None:
    """TEST-06 / D-09 schema #2: PEP 526 annotated ``_lazy_map`` with mixed string/tuple values.

    Exercises the ``ast.AnnAssign`` branch + the tuple-vs-string
    discrimination inside ``_update_lazy_map_from_dict``. Asserts the
    generator emits both:
      * ``from .csv import CsvHandler as Csv`` (tuple-form rename), and
      * ``from .core import find_pcd`` (string-form passthrough).
    """
    fixture_dir = tmp_path / "fix"
    fixture_dir.mkdir()
    (fixture_dir / "__init__.py").write_text(
        '"""Fixture package (tuple-form _lazy_map)."""\n'
        "__all__ = []\n"
        "_lazy_map: dict[str, str | tuple[str, str]] = {\n"
        '    "Csv": ("csv", "CsvHandler"),\n'
        '    "find_pcd": "core",\n'
        "}\n"
        "__all__ = __all__ + list(_lazy_map)\n",
        encoding="utf-8",
    )
    (fixture_dir / "csv.py").write_text("class CsvHandler:\n    pass\n", encoding="utf-8")
    (fixture_dir / "core.py").write_text("def find_pcd() -> None:\n    pass\n", encoding="utf-8")

    result = _run_generator(fixture_dir)
    pyi_path = fixture_dir / "__init__.pyi"
    assert pyi_path.exists() and pyi_path.stat().st_size > 0, (
        f"Generator silently no-op'd on tuple-form fixture. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    emitted = pyi_path.read_text(encoding="utf-8")
    assert "from .csv import CsvHandler as Csv" in emitted, (
        f"Generator did not emit the tuple-form alias 'CsvHandler as Csv'.\nEmitted:\n{emitted}"
    )
    assert "from .core import find_pcd" in emitted, (
        f"Generator did not emit the string-form passthrough 'find_pcd'.\nEmitted:\n{emitted}"
    )
    # Both public names land in __all__.
    assert "'Csv'" in emitted and "'find_pcd'" in emitted


def test_generator_empty_eager_init(tmp_path: Path) -> None:
    """TEST-06 / D-09 schema #3: no ``_lazy_map``, only eager submodule imports.

    Mirrors the ``30_GSEGUtils/src/GSEGUtils/__init__.py`` shape (no
    ``_lazy_map``; ``__all__`` lists submodules; ``from . import sub``
    eager imports). Asserts the generator emits a valid stub WITHOUT
    crashing AND the file is non-empty (RESEARCH Pitfall 5 guard:
    the generator silently no-ops on missing paths / empty maps, which
    would otherwise pass vacuously).
    """
    fixture_dir = tmp_path / "fix"
    fixture_dir.mkdir()
    (fixture_dir / "__init__.py").write_text(
        '"""Fixture package (empty-eager)."""\n__all__ = ["sub"]\nfrom . import sub\n',
        encoding="utf-8",
    )
    (fixture_dir / "sub.py").write_text("", encoding="utf-8")

    result = _run_generator(fixture_dir)
    pyi_path = fixture_dir / "__init__.pyi"

    # Pitfall 5 guard: assert the .pyi was actually emitted (not a silent no-op).
    assert pyi_path.exists(), (
        f"Generator silently no-op'd on empty-eager fixture; no __init__.pyi emitted.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert pyi_path.stat().st_size > 0, (
        f"Generator emitted an empty __init__.pyi for empty-eager fixture.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    emitted = pyi_path.read_text(encoding="utf-8")
    assert "from . import sub as sub" in emitted, (
        f"Generator did not emit the eager submodule import for 'sub'. Emitted:\n{emitted}"
    )
    assert "'sub'" in emitted, f"Generator omitted 'sub' from __all__. Emitted:\n{emitted}"
    assert "def __getattr__(name: str) -> NoReturn:" in emitted


def test_generator_pep526_annotated_consistency(tmp_path: Path) -> None:
    """TEST-06 / D-09 schema #4: PEP 526 annotated map mixing both inner schemas.

    Same shape as schema #2 but renames the entries to lock in the
    ``ast.AnnAssign`` code path explicitly under a single backing module.
    Verifies both inner schemas (string passthrough and tuple rename) are
    handled together when the annotation is present.
    """
    fixture_dir = tmp_path / "fix"
    fixture_dir.mkdir()
    (fixture_dir / "__init__.py").write_text(
        '"""Fixture package (PEP 526 annotated, mixed schemas)."""\n'
        "__all__ = []\n"
        "_lazy_map: dict[str, str | tuple[str, str]] = {\n"
        '    "A": "mod",\n'
        '    "B": ("mod", "RealB"),\n'
        "}\n"
        "__all__ = __all__ + list(_lazy_map)\n",
        encoding="utf-8",
    )
    (fixture_dir / "mod.py").write_text(
        "class A:\n    pass\n\n\nclass RealB:\n    pass\n",
        encoding="utf-8",
    )

    result = _run_generator(fixture_dir)
    pyi_path = fixture_dir / "__init__.pyi"
    assert pyi_path.exists() and pyi_path.stat().st_size > 0, (
        f"Generator silently no-op'd on PEP 526 mixed fixture. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    emitted = pyi_path.read_text(encoding="utf-8")
    # The annotated map is parsed via ast.AnnAssign; both inner schemas resolve.
    assert "from .mod import A" in emitted, (
        f"Generator did not emit the string-form passthrough for 'A'. Emitted:\n{emitted}"
    )
    assert "from .mod import RealB as B" in emitted, (
        f"Generator did not emit the tuple-form rename 'RealB as B'. Emitted:\n{emitted}"
    )
    assert "'A'" in emitted and "'B'" in emitted, f"Generator omitted 'A' or 'B' from __all__. Emitted:\n{emitted}"
