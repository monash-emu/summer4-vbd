"""Look up the installed summer4's API: names, signatures, docstrings, source.

summer4 ships no docs with the package, and this project has no copy of its repository, so
this reads the installed package itself. It always describes the version pinned in
``pixi.toml``.

    pixi run api                        # every public name, by module, with a one-line summary
    pixi run api ForceOfInfection       # signature, docstring, public methods, source location
    pixi run api Output.rolling         # one method
    pixi run api --source Output.at     # its source code
    pixi run api --search mixing        # names and docstrings that mention a word
    pixi run api --where                # installed version, commit and path
"""

from __future__ import annotations

import argparse
import enum
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from typing import Any

MODULES = (
    "summer4",
    "summer4.epi",
    "summer4.epi.calibration",
    "summer4.epi.calibration.workflow",
    "summer4.solvers",
    "summer4.data",
    "summer4.timevarying",
)


def _public(module: Any) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return list(names)


def _first_line(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _location(obj: Any) -> str:
    target = inspect.unwrap(obj)
    if isinstance(target, property):
        target = target.fget
    try:
        path = inspect.getsourcefile(target)
        _, line = inspect.getsourcelines(target)
    except (TypeError, OSError):
        return ""
    return f"{path}:{line}" if path else ""


def _modules() -> list[Any]:
    loaded = []
    for name in MODULES:
        try:
            loaded.append(importlib.import_module(name))
        except ImportError:
            continue
    return loaded


def _resolve(query: str) -> list[tuple[str, Any]]:
    """Find ``query`` (``Name``, ``Name.attr`` or a dotted module path) in the public API."""
    if query.startswith("summer4"):
        parts = query.split(".")
        for cut in range(len(parts), 0, -1):
            try:
                obj: Any = importlib.import_module(".".join(parts[:cut]))
            except ImportError:
                continue
            for attr in parts[cut:]:
                obj = getattr(obj, attr)
            return [(query, obj)]
        return []
    head, _, rest = query.partition(".")
    found: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for module in _modules():
        if head not in _public(module) or not hasattr(module, head):
            continue
        obj = getattr(module, head)
        for attr in rest.split(".") if rest else []:
            obj = getattr(obj, attr)
        if id(obj) not in seen:
            seen.add(id(obj))
            found.append((f"{module.__name__}.{query}", obj))
    return found


def _show(qualname: str, obj: Any) -> None:
    kind = (
        "module"
        if inspect.ismodule(obj)
        else "class" if inspect.isclass(obj) else "function" if callable(obj) else "value"
    )
    print(f"{qualname}  ({kind})")
    if callable(obj) and not inspect.ismodule(obj):
        sig = _signature(obj)
        if sig:
            print(f"    {qualname.rsplit('.', 1)[-1]}{sig}")
    where = _location(obj)
    if where:
        print(f"    defined at {where}")
    doc = inspect.getdoc(obj)
    if doc:
        print()
        print("\n".join("    " + line for line in doc.splitlines()))
    if kind == "value":
        print(f"\n    value: {obj!r}")
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        print("\n  values:")
        print("\n".join(f"    {m.name} = {m.value!r}" for m in obj))
    elif inspect.isclass(obj):
        members = []
        for name in _own_members(obj):
            member = getattr(obj, name)
            if isinstance(inspect.getattr_static(obj, name), property):
                members.append(f"    .{name}  (property)  {_first_line(member)}")
            elif callable(member):
                members.append(f"    .{name}{_signature(member)}  {_first_line(member)}")
        if members:
            print("\n  public members:")
            print("\n".join(members))
    print()


def _own_members(cls: type) -> list[str]:
    """Public attributes defined by summer4 classes in ``cls``'s MRO (not by ``str``, ``object``)."""
    names: dict[str, None] = {}
    for base in cls.__mro__:
        if not base.__module__.startswith("summer4"):
            continue
        for name in vars(base):
            if not name.startswith("_"):
                names[name] = None
    return sorted(names)


def _list() -> None:
    for module in _modules():
        names = _public(module)
        print(f"## {module.__name__}  ({len(names)} names)")
        for name in names:
            obj = getattr(module, name, None)
            print(f"  {name:<28} {_first_line(obj)[:90]}")
        print()


def _search(word: str) -> None:
    word_l = word.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for module in _modules():
        for name in _public(module):
            obj = getattr(module, name, None)
            key = f"{module.__name__}.{name}"
            if name in seen:
                continue
            doc = inspect.getdoc(obj) or ""
            if word_l in name.lower() or word_l in doc.lower():
                seen.add(name)
                hits.append(f"  {key:<45} {_first_line(obj)[:80]}")
            if inspect.isclass(obj) and not issubclass(obj, enum.Enum):
                for mname in _own_members(obj):
                    member = getattr(obj, mname)
                    mdoc = inspect.getdoc(member) or ""
                    if not mdoc or mdoc == doc:  # fields and enum members repeat the class doc
                        if word_l not in mname.lower():
                            continue
                    mkey = f"{name}.{mname}"
                    if mkey not in seen and (word_l in mname.lower() or word_l in mdoc.lower()):
                        seen.add(mkey)
                        hits.append(f"  {module.__name__}.{mkey:<37} {_first_line(member)[:80]}")
    print("\n".join(hits) if hits else f"No public name or docstring mentions {word!r}.")


def _where() -> None:
    import summer4

    dist = importlib.metadata.distribution("summer4")
    print(f"summer4 {dist.version}")
    print(f"package: {Path(summer4.__file__).parent}")
    direct = dist.read_text("direct_url.json")
    if direct:
        info = json.loads(direct)
        vcs = info.get("vcs_info", {})
        print(f"source:  {info.get('url')} @ {vcs.get('commit_id', '?')}")
        print(
            "         browse the docs and examples for this exact commit at "
            f"https://github.com/monash-emu/summer4/tree/{vcs.get('commit_id', 'main')}"
        )
    print("docs:    https://monash-emu.github.io/summer4/ (latest main; may be ahead of this pin)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="public names, e.g. ForceOfInfection")
    parser.add_argument("--source", action="store_true", help="print source code")
    parser.add_argument("--search", metavar="WORD", help="search names and docstrings")
    parser.add_argument("--where", action="store_true", help="installed version and path")
    args = parser.parse_args(argv)
    if args.where:
        _where()
        return 0
    if args.search:
        _search(args.search)
        return 0
    if not args.names:
        _list()
        return 0
    status = 0
    for query in args.names:
        matches = _resolve(query)
        if not matches:
            print(f"{query}: not found in the public API. Try `pixi run api --search {query}`.")
            status = 1
            continue
        for qualname, obj in matches:
            if args.source:
                print(f"# {qualname}  ({_location(obj)})")
                print(inspect.getsource(inspect.unwrap(obj)))
            else:
                _show(qualname, obj)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
