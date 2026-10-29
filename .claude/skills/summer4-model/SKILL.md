---
name: summer4-model
description: Build, change, run or calibrate a summer4 compartmental model in this project (compartments, flows, force of infection, outputs, targets, BayesianModel). Use for any task that writes or edits summer4 code, before writing it.
---

# Building or changing a summer4 model

summer4 is probably unfamiliar, and it is not summer2 or summer3. Do not write summer4 code
from memory. This project has no copy of summer4's repository; the references are
`docs/summer4-guide.md` and the installed package, via `pixi run api`.

## 1. Orient (once per task)

- Read `docs/summer4-guide.md`: at least section 1 (concepts), the sections for what the
  task touches, and section 10 (pitfalls).
- If the task resembles the example, read `notebooks/01-Example.ipynb`.
- `pixi run api --where` shows the installed version. The guide describes that version only.

## 2. Look up before you use

For every summer4 name you are not certain of, run `pixi run api NAME` (signature,
docstring, methods) or `pixi run api --search WORD`. Check:

- argument names and which are keyword-only;
- whether a rate is per capita or absolute (`EntryFlow` is people per day);
- whether a flow output needs `side=` (`sum_by(age, side="source")`);
- whether a `MixingMatrix` should be row-normalised (the default) or literal
  (`normalize=MixingNormalize.NONE`);
- likelihood `aggregate` (pass `"sum"`).

## 3. Build in small, runnable steps

1. Properties and `PropertyMap`; print `pmap.labels()` and confirm the compartments.
2. `FlowModel(pmap)` and flows, one at a time. Name the force of infection differently
   from every flow and parameter (for example `"foi"`). For each flow, name every
   property you restrict on **both** sides (`S & adult` → `V & adult`), or the flow fans
   out; after compiling, read `compiled.edges(name)` and confirm each source → dest.
3. Initial population; `np.asarray(compiled.initial_state(params).data).sum()` should be
   the intended total.
4. `compile()`, then `run(params, t0=..., t1=..., dt=..., epoch=...)`; check
   `result.solver.ok`.

Run each step with `pixi run python` (a scratch script is fine) rather than assuming.

## 4. Sanity-check the numbers

- Every flow's edges are the ones you meant (`compiled.edges(name)`); conservation alone
  does not catch a fan-out.
- The first saved value of each flow is rate × source stock.
- Population conserved, or changing by exactly births minus deaths.
- Flow outputs are rates per unit time; integrate (`integrate_intervals()`,
  `integrate()`) before comparing with counts.
- Units: model time is whatever `Epoch(unit=...)` and your rates say (days by default).
- If accuracy matters, compare `solver="tsit5"` with the default Euler.

## 5. Calibration

- Targets refer to `OutputSet` names; build the observed series with a `DatetimeIndex`
  and pass the run's `epoch`.
- Start with `bm.find_map()` to check the target and priors make sense, then
  `bm.sample(num_warmup=..., num_samples=..., num_chains=4)`.
- Report R-hat and ESS from `arviz.summary(mcmc.idata)`: below 1.05 while developing,
  ≤ 1.01 for final results.
- Bad R-hat or tiny ESS: plot the fitted output at a few prior values first. An epidemic
  that ends before the observations do (all-zero tail) cannot be fitted by more draws.

## 6. Finish

- Write captions and summaries after the run, quoting its numbers.
- Move reusable model construction into `src/vbd_model/` with a test in `tests/`.
- `pixi run clear-notebooks`, then `pixi run check`; both must pass.
- Do not change the summer4 pin unless asked.
