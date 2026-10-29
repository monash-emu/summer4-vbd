# Copilot instructions

Follow [AGENTS.md](../AGENTS.md) at the repository root. In short:

- This project uses summer4, pinned in `pixi.toml`. Its API is not summer2's or summer3's;
  do not guess it. Read `docs/summer4-guide.md` and look names up with `pixi run api NAME`.
- Run everything through pixi (`pixi run python ...`); verify changes with `pixi run check`.
- Commit notebooks without outputs (`pixi run clear-notebooks`).
