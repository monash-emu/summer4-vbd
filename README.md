# vbd-summer4

A basic synthetic vector-borne disease model in [summer4](https://github.com/monash-emu/summer4),
started from [monash-emu/summer4template](https://github.com/monash-emu/summer4template).
No real data; all parameters are illustrative.

- **Humans**: SEIRS in four age bands (0–4, 5–14, 15–39, 40+), with ageing, and births
  replacing deaths.
- **Mosquitoes**: SEI (with an extrinsic incubation period). Emergence is seasonal (a cosine
  over the year), so mosquito density and transmission cycle annually.
- **Transmission**: Ross–Macdonald biting. Each age band has a relative bite rate
  (`rel_bite_<age>`, which shares out mosquito bites) and a relative susceptibility
  (`rel_susc_<age>`). An optional intervention cuts bites per mosquito by `bite_reduction`
  from `intervention_start`.

| Path | What it is |
|---|---|
| `src/vbd_model/vbd_model/model.py` | `build_model()`, `DEFAULT_PARAMS`, and `r0()` (next-generation matrix) |
| `tests/test_model.py` | edges, populations, seasonal mosquito dynamics, hand-checked infection rates, R0 = 1 threshold |
| `notebooks/02-Vector-model.ipynb` | seasonal baseline, incidence and immunity by age, 10/20/30% bite-reduction scenarios |
| `notebooks/01-Example.ipynb` | the template's SIR example with calibration, kept for reference |

```bash
pixi install
```

Then open the notebook with the `default` environment's kernel
(`.pixi/envs/default/bin/python`). Works on macOS (Apple silicon) and Linux.

| Command | What it does |
|---|---|
| `pixi run check` | run the tests, then every notebook headless |
| `pixi run api NAME` | show the installed summer4's documentation for `NAME` (`pixi run api` lists everything) |
| `pixi run clear-notebooks` | clear notebook outputs before committing |

[`docs/summer4-guide.md`](docs/summer4-guide.md) is a tested, version-specific guide to
summer4: compartments, flows, infection, results and calibration, plus the common
pitfalls and a summer2/summer3 translation table.

summer4 is pinned to a commit in `pixi.toml` (currently `v0.3.0a1`). To move to a
newer summer4, change `tag` there, run `pixi update summer4`, update the guide, and run
`pixi run check`.

Model code lives in `src/vbd_model` (installed in editable mode).

## Coding agents

The project is set up for coding agents that have never seen summer4.
[`AGENTS.md`](AGENTS.md) is the entry point (read natively by Codex, Cursor, Copilot,
Jules and others; `CLAUDE.md`, `GEMINI.md` and `.github/copilot-instructions.md` point to
it). It sends agents to the guide, to `pixi run api` for anything else, and to
`pixi run check` to verify their work. Claude Code also gets a `summer4-model` skill
(`.claude/skills/`) with a step-by-step checklist for model work.
