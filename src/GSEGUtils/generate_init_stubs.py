#!/usr/bin/env python3
"""
generate_init_stubs.py  —  Pure AST stub generator for lazy __init__.py

- Parses __all__ and _lazy_map from __init__.py without importing.
- Supports:
  * _lazy_map values as "module" or ("module", "RealName")
  * __all__ as list/tuple literals, += list(...), extend/append
  * Annotated assignments (PEP 526): e.g. _lazy_map: dict[str, str | tuple[str,str]] = {...}
- Options:
  * --walk : traverse given roots and process all package dirs with __init__.py
  * --submodule-stubs {eager,any} : default eager (real imports in .pyi) or Any-typed submodules
  * --create-py-typed : create py.typed per package if missing
  * --overwrite : overwrite existing __init__.pyi

New in this version:
- Stub includes a concrete __all__: Final[list[str]] = [...]
- Stub adds __getattr__ typing:
  * eager: def __getattr__(name: str) -> NoReturn  (everything should be declared)
  * any  : overloads for submodule names -> ModuleType, fallback -> NoReturn
"""
from __future__ import annotations
import argparse, ast, os, sys
from pathlib import Path
from typing import Any, Iterable

# ----------------- AST utilities -----------------

def _const_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

def _const_tuple2(node: ast.AST) -> tuple[str, str] | None:
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        a = _const_str(node.elts[0]); b = _const_str(node.elts[1])
        if a is not None and b is not None:
            return a, b
    return None

def _update_lazy_map_from_dict(dst: dict[str, str | tuple[str, str]], dict_node: ast.Dict) -> None:
    for k, v in zip(dict_node.keys, dict_node.values):
        ks = _const_str(k)
        if ks is None:
            continue
        vs = _const_str(v)
        if vs is not None:
            dst[ks] = vs
            continue
        vt = _const_tuple2(v)
        if vt is not None:
            dst[ks] = vt

def _extend_exports_from_seq(dst: list[str], seq: ast.AST) -> None:
    if isinstance(seq, (ast.List, ast.Tuple)):
        for elt in seq.elts:
            s = _const_str(elt)
            if s:
                dst.append(s)

# ----------------- Parsing -----------------

def parse_ast(init_py: Path) -> tuple[dict[str, str | tuple[str, str]], list[str], dict[str, bool]]:
    src = init_py.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(init_py))

    lazy_map: dict[str, str | tuple[str, str]] = {}
    exports: list[str] = []
    dunders = {"__author__": False, "__email__": False}

    for node in tree.body:
        # Handle regular assignments: __all__ = [...], _lazy_map = {...}, __author__ = "..."
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]

            if "__all__" in targets:
                _extend_exports_from_seq(exports, node.value)
                # __all__ = __all__ + [...]
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    _extend_exports_from_seq(exports, node.value.right)

            if "_lazy_map" in targets and isinstance(node.value, ast.Dict):
                _update_lazy_map_from_dict(lazy_map, node.value)

            for dn in dunders:
                if dn in targets and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    dunders[dn] = True

        # Handle annotated assignments (PEP 526): e.g., _lazy_map: dict[...] = {...}
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value
            if name == "__all__" and value is not None:
                _extend_exports_from_seq(exports, value)
            if name == "_lazy_map" and isinstance(value, ast.Dict):
                _update_lazy_map_from_dict(lazy_map, value)
            if name in dunders and isinstance(value, ast.Constant) and isinstance(value.value, str):
                dunders[name] = True

        # __all__.extend([...]) / __all__.append("x")
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "__all__"
            ):
                if call.func.attr == "extend" and call.args:
                    _extend_exports_from_seq(exports, call.args[0])
                if call.func.attr == "append" and call.args:
                    s = _const_str(call.args[0])
                    if s:
                        exports.append(s)

    # Ensure lazy_map keys appear as exported names
    for k in lazy_map:
        if k not in exports:
            exports.append(k)

    # De-dup while preserving order
    seen = set()
    exports = [x for x in exports if not (x in seen or seen.add(x))]
    return lazy_map, exports, dunders

# ----------------- Stub generation -----------------

def build_stub_text(
    lazy_map: dict[str, str | tuple[str, str]],
    exports: Iterable[str],
    dunders: dict[str, bool],
    submodule_mode: str,  # "eager" | "any"
) -> str:
    exports = list(exports)
    dunder_names = {"__author__", "__email__", "__all__"}
    # Treat exported names that aren't lazy-mapped as submodules
    submodules = [n for n in exports if n not in lazy_map and n.isidentifier() and n not in dunder_names]

    needs_moduletype_overloads = (submodule_mode == "any" and len(submodules) > 0)

    lines: list[str] = [
        "# Auto-generated stub for lazy exports",
        "from typing import Any, Final, NoReturn",
    ]
    if needs_moduletype_overloads:
        lines.append("from typing import overload, Literal")
        lines.append("from types import ModuleType")
    lines.append("")

    # Group lazy_map entries by backing module for compact imports
    by_mod: dict[str, list[tuple[str, str | None]]] = {}
    for public, target in lazy_map.items():
        if isinstance(target, str):
            by_mod.setdefault(target, []).append((public, None))
        else:
            mod, real = target
            by_mod.setdefault(mod, []).append((public, real))

    # Import concrete symbols (classes/functions/constants) so they exist at stub top-level
    for mod in sorted(by_mod):
        simples = [pub for pub, real in by_mod[mod] if real is None]
        aliases = [(pub, real) for pub, real in by_mod[mod] if real is not None]
        if simples:
            names = ", ".join(sorted(set(simples)))
            lines.append(f"from .{mod} import {names}")
        for pub, real in sorted(set(aliases), key=lambda x: (x[1], x[0])):
            lines.append(f"from .{mod} import {real} as {pub}")
    if by_mod:
        lines.append("")

    # Submodules from __all__
    if submodule_mode == "eager":
        for sub in sorted(set(submodules)):
            lines.append(f"from . import {sub} as {sub}")
        if submodules:
            lines.append("")
    else:  # "any" — don't import them; they'll be available via __getattr__ overloads
        # Optionally, give them a very weak declaration to aid completion (not required)
        # for sub in sorted(set(submodules)):
        #     lines.append(f"{sub}: Any")
        # if submodules:
        #     lines.append("")
        pass

    # Dunders (declare if present)
    if dunders.get("__author__"):
        lines.append("__author__: str")
    if dunders.get("__email__"):
        lines.append("__email__: str")

    # Always emit a concrete __all__ list in the stub
    # Keep only identifier-like exports and strip dunders (except __all__)
    all_exports = [e for e in exports if e != "__all__"]
    if all_exports:
        quoted = ", ".join(repr(e) for e in all_exports)
        lines.append(f"__all__: Final[list[str]] = [{quoted}]")
    else:
        lines.append("__all__: Final[list[str]] = []")
    lines.append("")

    # __getattr__ typing:
    # - eager: everything that exists is declared above; any other name is an error
    # - any: allow listed submodule names (-> ModuleType), everything else is an error
    if submodule_mode == "eager":
        lines.append("def __getattr__(name: str) -> NoReturn: ...")
    else:
        if needs_moduletype_overloads:
            # Overload for allowed submodule names
            names = ", ".join(repr(s) for s in sorted(set(submodules)))
            lines.append("@overload")
            lines.append(f"def __getattr__(name: Literal[{names}]) -> ModuleType: ...")
        # Fallback: everything else is an error for type checkers
        lines.append("def __getattr__(name: str) -> NoReturn: ...")

    lines.append("")
    return "\n".join(lines)

# ----------------- Discovery / IO -----------------

DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site", "venv", ".venv", "__pycache__", "env", ".env",
}

def is_package_dir(p: Path) -> bool:
    return p.is_dir() and (p / "__init__.py").exists()

def find_package_dirs(roots: list[Path], walk: bool) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not walk:
            if is_package_dir(root):
                found.append(root)
            else:
                print(f"[skip] Not a package dir (no __init__.py): {root}", file=sys.stderr)
            continue

        if root.is_file():
            if root.name == "__init__.py" and is_package_dir(root.parent):
                found.append(root.parent)
            else:
                print(f"[skip] File is not __init__.py: {root}", file=sys.stderr)
            continue

        for dirpath, dirnames, _files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDES and not d.startswith(".")]
            dpath = Path(dirpath)
            if is_package_dir(dpath):
                found.append(dpath)

    # Dedup preserve order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique

def write_stub(init_dir: Path, text: str, overwrite: bool) -> None:
    out = init_dir / "__init__.pyi"
    if out.exists() and not overwrite:
        print(f"[skip] {out} exists (use --overwrite)", file=sys.stderr)
        return
    out.write_text(text, encoding="utf-8")
    print(f"[write] {out}")

# ----------------- CLI -----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate __init__.pyi stubs for lazy-loading packages.")
    ap.add_argument("paths", type=Path, nargs="+", help="Package dirs or roots to process.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing __init__.pyi files.")
    ap.add_argument("--create-py-typed", action="store_true", help="Create py.typed in each processed package dir if missing.")
    ap.add_argument(
        "--submodule-stubs",
        choices=["eager", "any"],
        default="eager",
        help="How to represent submodules listed in __all__: "
             "'eager' = real imports in the .pyi (stronger checking), "
             "'any' = only allow via __getattr__ overloads (ModuleType). Default: eager.",
    )
    ap.add_argument(
        "--walk",
        action="store_true",
        help="Greedily traverse the given paths and process ALL package dirs containing __init__.py",
    )
    args = ap.parse_args()

    pkgs = find_package_dirs([p.resolve() for p in args.paths], walk=args.walk)
    if not pkgs:
        print("[info] No package directories found to process.", file=sys.stderr)
        return

    for pkg_dir in pkgs:
        init_py = pkg_dir / "__init__.py"
        try:
            lazy_map, exports, dunders = parse_ast(init_py)
            stub = build_stub_text(lazy_map, exports, dunders, args.submodule_stubs)
            write_stub(pkg_dir, stub, args.overwrite)
            if args.create_py_typed:
                pt = pkg_dir / "py.typed"
                if not pt.exists():
                    pt.write_text("", encoding="utf-8")
                    print(f"[write] {pt}")
        except Exception as e:
            print(f"[error] Failed on {init_py}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()