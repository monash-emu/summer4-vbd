# Agent instructions

This is a compartmental-modelling project built on
[summer4](https://github.com/monash-emu/summer4), a JAX library for epidemic models. It
started from `monash-emu/summer4template`. These instructions are for any coding agent
(Claude Code, Codex, Cursor, Copilot, Gemini, ...) and for people new to summer4.

## Before you write any summer4 code

You probably do not know summer4: it is new, and it is **not** summer2 or summer3 with a
new name. Its API is different, and guessing from those, or from summer4's own `main`
branch, will produce code that does not run against the pinned version. This project has
no copy of summer4's repository; everything you need is here or in the installed package.
summer4's documentation site, <https://monash-emu.github.io/summer4/>, is useful with web
access, but it documents summer4's latest `main`, which may be ahead of the pin.

1. Read [`docs/summer4-guide.md`](docs/summer4-guide.md). It covers the whole workflow
   (compartments, flows, infection, running, results, calibration) for exactly the pinned
   version, every snippet in it is tested, and section 10 lists the mistakes people make.
2. Read [`notebooks/01-Example.ipynb`](notebooks/01-Example.ipynb): a complete age-stratified
   SIR model, run and calibrated.
3. Look up anything else in the installed package rather than guessing:

   ```bash
   pixi run api                      # every public name, with a one-line summary
   pixi run api ForceOfInfection     # signature, docstring, methods, source location
   pixi run api Output.rolling       # one method
   pixi run api --search mixing      # search names and docstrings
   pixi run api --source Output.at   # read the implementation
   pixi run api --where              # installed version, commit, path
   ```

## Environment

Everything runs through [pixi](https://pixi.sh). Do not use `pip install`, a system
Python, or a separate virtualenv.

| Task | Command |
|---|---|
| Install or refresh the environment | `pixi install` |
| Run a script | `pixi run python path/to/script.py` |
| Check everything (tests, then every notebook headless) | `pixi run check` |
| Tests only (fast) | `pixi run check --quick` (or `pixi run python -m pytest tests`) |
| Clear notebook outputs before committing | `pixi run clear-notebooks` |
| Look up summer4 | `pixi run api ...` |

Supported platforms: macOS on Apple silicon (`osx-arm64`) and Linux (`linux-64`).
Notebooks use the environment's kernel, `.pixi/envs/default/bin/python`.

## Layout

| Path | What goes there |
|---|---|
| `notebooks/` | Analyses and examples. Committed **without outputs**. |
| `src/vbd_model/vbd_model/` | Reusable code (model builders, parameters, plotting), installed in editable mode: `import vbd_model` or `from vbd_model.model import build_model`. The outer `src/vbd_model/` holds its `pyproject.toml`. Rename to suit. |
| `tests/` | pytest tests, run by `pixi run check`. `test_summer4_guide.py` keeps the guide honest. |
| `docs/summer4-guide.md` | The summer4 reference for the pinned version. |
| `tools/` | `summer4_api.py` (`pixi run api`), `check.py`, `clear_notebooks.py`. |

## Working conventions

- **Verify by running.** After a change, run `pixi run check`; it must pass before you say
  the work is done. For a quick look at a model, write a short script and
  `pixi run python` it rather than reasoning about what summer4 will do.
- **Keep model code importable.** When a model grows past a notebook cell or two, move
  its construction into a function in `src/vbd_model/vbd_model/` (for example
  `build_model() -> FlowModel` in `model.py`) and add a test in `tests/` that compiles
  and runs it.
- **Check results, not just that code runs.** Code that runs can still be wrong:
  - list every flow's edges (`compiled.edges(name)`) and check each source and
    destination; a selector that names a property on only one side fans out silently
    (guide pitfall 1), and population conservation will not catch it;
  - check the first saved value of a flow equals rate × source stock;
  - check population is conserved, or changes by exactly births minus deaths;
  - check units (model time is usually days);
  - calibration: R-hat below 1.05 while developing, ≤ 1.01 for results you report.
- **Write claims after measuring.** Captions, "what to check" text and summaries quote
  numbers from the run, written after it, not expectations written before.
- **Do not change the summer4 pin** in `pixi.toml` unless asked. If you are asked:
  change `tag` (or `rev`), run `pixi update summer4`, update the version named at the top of
  `docs/summer4-guide.md` and anything the new release changed (its notes are at
  https://github.com/monash-emu/summer4/releases), then `pixi run check`.
- Clear notebook outputs (`pixi run clear-notebooks`) before committing.

## Troubleshooting

- **`ModuleNotFoundError` for `vbd_model` only in Jupyter**: the notebook kernel is
  not this project's environment (it can happen after moving or copying `.pixi/`). Select
  the kernel at `.pixi/envs/default/bin/python`, or re-register it with
  `pixi run python -m ipykernel install --prefix .pixi/envs/default --name python3`.
  `pixi run check` always uses this project's interpreter.
- **A summer4 call fails with an unexpected argument**: you are probably using a newer or
  older API. `pixi run api NAME` shows the installed signature.

## Task-specific help

[`.claude/skills/summer4-model/SKILL.md`](.claude/skills/summer4-model/SKILL.md) is a
step-by-step checklist for building or changing a model. Claude Code loads it as a skill;
other agents can read it directly.
