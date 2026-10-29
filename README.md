# summer4template

A small pixi template for [summer4](https://github.com/monash-emu/summer4) projects.

`notebooks/01-Example.ipynb` builds an age-stratified SIR model, runs it, makes
synthetic weekly observations from it, and calibrates the two parameters with NUTS.

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

Put your own code in `src/vbd_model` (installed in editable mode), and
rename it to suit.

## Coding agents

The project is set up for coding agents that have never seen summer4.
[`AGENTS.md`](AGENTS.md) is the entry point (read natively by Codex, Cursor, Copilot,
Jules and others; `CLAUDE.md`, `GEMINI.md` and `.github/copilot-instructions.md` point to
it). It sends agents to the guide, to `pixi run api` for anything else, and to
`pixi run check` to verify their work. Claude Code also gets a `summer4-model` skill
(`.claude/skills/`) with a step-by-step checklist for model work.
