# summer4 guide for this project

This is a working reference for **summer4 0.3.0a1**, the version pinned in `pixi.toml`.
It is written for people and coding agents who have not used summer4 before and have no
copy of its repository: only what `pixi install` puts in `.pixi/`.

Every ```` ```python ```` block below runs, top to bottom, in `tests/test_summer4_guide.py`,
so the guide cannot drift from the installed version without `pixi run check` failing.
Model time is in **days** throughout.

**Contents:** [1 Concepts](#1-concepts) · [2 Compartments](#2-compartments) ·
[3 Flows and rates](#3-flows-and-rates) · [4 Infection](#4-infection) ·
[5 Initial population](#5-initial-population) · [6 Running](#6-running) ·
[7 Results](#7-results) · [8 Named outputs](#8-named-outputs-outputset) ·
[9 Calibration](#9-calibration) · [10 Pitfalls](#10-pitfalls) ·
[11 Finding more](#11-finding-more) · [12 Coming from summer2/summer3](#12-coming-from-summer2-or-summer3)

## 1. Concepts

```text
Property ──► PropertyMap ──► FlowModel ──add_flow──► compile() ──► CompiledModel
(a named      (the compartments:            (flows,                    │ .run(params, t0, t1, dt, ...)
 set of        every combination of          initial                   ▼
 traits)       traits, optionally ragged)    population)            Result ──► Output (one saved series)
                                                                       │
                                     OutputSet (named derived outputs) ┘──► Target ──► BayesianModel
```

- A **compartment** is a combination of traits, one per property (`state=S_age=0`). There
  is no compartment object and no string naming: you pick compartments with **selectors**.
- A **flow** moves people between compartments chosen by selectors, at a **rate** that is
  an expression (`Param("x")`, numbers, arithmetic, `Time()`, a force of infection). The
  expression is evaluated at run time, so changing parameter values never recompiles.
- `compile()` turns the model into a JAX program. Compile once, then `run()` as often as
  you like with different parameters.
- A run returns a **Result**: a mapping from names to **Output** objects (a time series,
  possibly one column per compartment or per flow edge), which you reduce and combine.

## 2. Compartments

```python
import numpy as np
import pandas as pd

from summer4 import Property, PropertyMap

state = Property("state", ["S", "I", "R"])
age = Property("age", ["0", "15", "65"])

pmap = PropertyMap.from_property(state).stratify(age)
assert pmap.labels()[:3] == ("state=S_age=0", "state=S_age=15", "state=S_age=65")
assert pmap.size == 9
```

**Selectors** pick compartments. `prop["trait"]` is the basic one; combine with `&`, `|`
and `~`. `prop.present()` matches every compartment that has the property.
`pmap.select(selector)` returns the indices.

```python
assert list(pmap.select(state["I"] & age["0"])) == [3]
assert list(pmap.select(~state["S"])) == [3, 4, 5, 6, 7, 8]
assert len(pmap.select(age.present())) == 9
```

Stratification can be **ragged**: `stratify(prop, where=selector)` splits only the
compartments the selector matches (here only infectious people carry a strain).

```python
strain = Property("strain", ["a", "b"])
ragged = PropertyMap.from_property(state).stratify(strain, where=state["I"])
assert ragged.labels() == ("state=S", "state=I_strain=a", "state=I_strain=b", "state=R")
```

## 3. Flows and rates

```python
from summer4 import (
    EntryFlow,
    ExitFlow,
    FlowModel,
    Multiply,
    Param,
    Time,
    TraitChain,
    TransitionFlow,
)

model = FlowModel(pmap)

model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 1.0 / Param("recovery_time")))
```

- `TransitionFlow(name, source, dest, rate)`: `source` and `dest` are selectors. A property
  named on **neither** side is kept, so `TransitionFlow("recovery", state["I"], state["R"], ...)`
  moves each age group within its own age.
- A property named on **only one** side is free on the other, and the flow **fans out**: with
  `source=state["S"] & age["65"]` and `dest=state["R"]`, people aged 65 move to `R` in
  *every* age group, each edge at the full rate. Name the property on both sides
  (`dest=state["R"] & age["65"]`) or on neither. Check with `compiled.edges(name)`
  (example in [section 6](#6-running)).
- The rate is **per capita** (per person in the source, per day) unless `absolute=True`.
- Rates are expressions: numbers, `Param("name")`, `+ - * / **`, NumPy ufuncs such as
  `np.exp(Param("log_k"))` and `np.cos(...)`, and `Time()` (the current model time).
  Time-varying helpers are in `summer4.timevarying` (`linear`, `step`, `sigmoidal`).
- `Reduce(where=selector)` is the current number of people in the matching compartments,
  and `Reduce()` the whole population, so mass action needs no epidemiology helper:
  `Param("beta") * Reduce(where=state["I"])` (see [section 4](#4-infection) for the usual
  force of infection).

```python
from summer4 import Reduce

mass_action = Param("beta") * Reduce(where=state["I"])  # a rate expression, per susceptible
```
- Names must be unique across flows, and also across forces of infection and parameters
  (see [Pitfalls](#10-pitfalls)).

**Stratum-specific rates** use `adjust=`: `Multiply(value, where=selector)` scales the rate
for the matching source compartments, `Overwrite(value, where=...)` replaces it.

**Ageing** is one flow with a `TraitChain` pairing (`rates` are per-band multipliers; the
last band has no exit). `TraitChain.from_breakpoints(age, unit=365.0)` builds the same
from numeric trait names when every band should leave at `1 / width`.

```python
model.add_flow(
    TransitionFlow(
        "ageing",
        age.present(),
        age.present(),
        1.0 / 365.0,
        pairing=TraitChain(age, (("0", "15"), ("15", "65")), rates=(1 / 15, 1 / 50)),
    )
)
```

**Leaving and entering the system**: `ExitFlow(name, source, rate)` (per capita) and
`EntryFlow(name, dest, rate)`, whose rate is **people per day**. `add_flow` returns a
`FlowRef`, so a rate can refer to another flow, here births replacing deaths exactly:

```python
death = model.add_flow(ExitFlow("death", age["65"], 1.0 / (15 * 365.0)))
model.add_flow(EntryFlow("birth", state["S"] & age["0"], death.sum_by()))
```

## 4. Infection

`ForceOfInfection` (in `summer4.epi`) is a rate expression. Use it as the rate of the
infection flow.

```python
from summer4.epi import ForceOfInfection

seasonal = 1 + 0.2 * np.cos(2 * np.pi * Time() / 365.0)
foi = ForceOfInfection("foi", infectious=state["I"], contact_rate=Param("contact_rate") * seasonal)

model.add_flow(
    TransitionFlow(
        "infection",
        state["S"],
        state["I"],
        foi,
        adjust=[Multiply(Param("child_susceptibility"), where=age["0"])],
    )
)
```

- With no `group_by` (above), mixing is **homogeneous over the whole population**:
  `contact_rate * I_total / N_total`, however the compartments are stratified. This is
  summer2's default.
- `group_by=age` **without** a mixing matrix means each age group catches infection only
  from its own group. For age-structured mixing, pass a `MixingMatrix` (below).
- `kind=` chooses frequency (default) or density dependence; `infectiousness=` weights who
  transmits. See `pixi run api ForceOfInfection`.

Age-structured mixing: `matrix[i][j]` is contact of age group `i` (being infected) with
group `j`, in `age.traits` order. **By default each row is normalised to sum to 1**, so the
matrix sets the *pattern* and `contact_rate` the overall level. To use the numbers as
absolute contact rates, pass `normalize=MixingNormalize.NONE`, and scale `contact_rate`
down to match.

```python
from summer4.epi import MixingMatrix, MixingNormalize

contacts = np.array([[4.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 2.0]])
structured = ForceOfInfection(
    "foi_by_age",
    infectious=state["I"],
    contact_rate=Param("contact_rate"),
    group_by=age,
    mixing=MixingMatrix(age, contacts),  # rows normalised; MixingNormalize.NONE keeps them
)
```

## 5. Initial population

Give the size of each group, then split it along other properties. Keys are selectors;
`Split` weights are fractions.

```python
from summer4 import Split

model.set_initial_population(
    {age["0"]: 2e5, age["15"]: 6e5, age["65"]: 2e5},
    [Split(state, {"S": 0.999, "I": 0.001, "R": 0.0})],
)
```

You can also pass a `y0` array of length `pmap.size` (in `pmap.labels()` order) to `run`.

## 6. Running

```python
from summer4 import Epoch

compiled = model.compile()
params = {"contact_rate": 0.25, "recovery_time": 10.0, "child_susceptibility": 0.5}
epoch = Epoch(pd.Timestamp("2024-01-01"))  # model time 0 is this date; 1 unit = 1 day

result = compiled.run(params, t0=0.0, t1=365.0, dt=1.0, epoch=epoch)
assert result.solver.ok

initial = compiled.initial_state(params)  # a PropertyData: the numbers are .data
assert abs(float(np.asarray(initial.data).sum()) - 1e6) < 1
```

- Every `Param` name used by the model must be in `params`.
- `dt` is the save spacing **and** the step of the default solver, which is fixed-step
  **Euler** (first order). With `dt=1` on a typical epidemic it puts the peak a few
  percent and about a day off. Pass `solver="tsit5"` (adaptive) when peaks, timings or
  totals matter, or use a smaller `dt`.
- `epoch=` makes results date-indexed and enables calendar operations (`resample`,
  `Target.from_series`). Without it, time is just numbers.
- `save=` limits what is kept; by default everything is saved (next section).

**Check what each flow does** before trusting a run: `compiled.edges(name)` lists its
edges, and the first saved flow value is the per-capita rate times the source stock.

```python
labels = compiled.pmap.labels()
recovery_edges = compiled.edges("recovery")
assert [
    (labels[a], labels[b]) for a, b in zip(recovery_edges.src_idx, recovery_edges.dest_idx)
] == [
    ("state=I_age=0", "state=R_age=0"),
    ("state=I_age=15", "state=R_age=15"),
    ("state=I_age=65", "state=R_age=65"),
]
```

**Scenarios**: compile once and run with different parameters; only `params` changes.

```python
faster = compiled.run({**params, "recovery_time": 5.0}, t0=0.0, t1=365.0, dt=1.0, epoch=epoch)
```

## 7. Results

`result.keys()` lists what a default run saved: `"compartments"`, every **flow** by name,
every **force of infection** by name, and every **parameter** by name, in one namespace.

```python
assert {"compartments", "infection", "recovery", "foi", "contact_rate"} <= set(result.keys())

comps = result["compartments"]
assert comps.dims == ("time", "compartment")

population = comps.sum_by()                                  # one value per time
infectious_by_age = comps.select(state["I"]).sum_by(age)     # one column per age
assert population.dims == ("time",)
assert abs(float(np.asarray(population.values)[-1]) - 1e6) < 10  # births replace deaths
```

A **flow output is a rate**: people per day at each saved time, one column per edge.

```python
incidence = result["infection"].sum_by()                         # infections per day
incidence_by_age = result["infection"].sum_by(age, side="source")  # flows need side=

daily_counts = incidence.integrate_intervals()        # infections in each day (count)
cumulative = incidence.cumulative_integral()          # infections so far, at each save
weekly = daily_counts.resample("W", how="sum")        # calendar weeks (needs an epoch)
whole_year = incidence.integrate().scalar()           # one number (a 0-d array)

assert abs(float(np.asarray(cumulative.values)[-1]) - float(whole_year)) < 1e-6 * whole_year
```

An output reduced over time (`integrate()`, `at(t)`) has no `time` dim and combines with
a series by repeating its value at every time:

```python
share = incidence / incidence.integrate()      # share of the year's infections, per day
change = population - population.at(0.0)       # change since day 0
assert share.dims == ("time",)
assert abs(float(np.asarray(share.integrate().values)) - 1.0) < 1e-9
```

Useful `Output` methods (`pixi run api Output` lists them all):

| Method | What it does |
|---|---|
| `select(sel)` | keep the compartments or edges matching a selector |
| `sum_by()` / `sum_by(age)` | sum across strata, keeping time (and one column per `age` trait) |
| `between(t0, t1)`, `at(t)`, `at_times(ts)` | slice, take, or interpolate time |
| `rolling(n, how="sum")` | window of `n` **saved points** (a week only if saves are daily); the first `n - 1` values are NaN |
| `resample("W", how="sum")` | calendar bins (needs `epoch=`) |
| `integrate_intervals()`, `integrate()` | trapezoid integral per save interval, or over the whole run |
| `cumulative_integral()` | running integral of a **rate**: cumulative people at each save |
| `cumsum()` | running **sum** of amounts per interval (for example `integrate_intervals()` output) |
| `array()`, `scalar()` | the values as a plain JAX array (`scalar()` requires a single value) |
| `to_pandas()`, `values`, `times` | export; a time-only output has one column, `"value"` |

Outputs on the same time axis combine with `+ - * /` and NumPy ufuncs, matching columns
by property (`infectious_by_age / population` is prevalence by age):

```python
prevalence_by_age = infectious_by_age / population
frame = prevalence_by_age.to_pandas()
assert list(frame.columns) == ["age=0", "age=15", "age=65"]
```

## 8. Named outputs (`OutputSet`)

An `OutputSet` declares derived outputs once, saves only what they need, and is what
calibration targets refer to. Leaves are `Compartments(...)` and `FlowMass(flow_name)`,
with the same methods as `Output`; `outputs.ref(name)` builds on another output.

```python
from summer4 import Compartments, FlowMass, OutputSet

outputs = OutputSet()
outputs["incidence"] = FlowMass("infection").sum_by()
outputs["weekly_incidence"] = outputs.ref("incidence").rolling(7, how="sum")
outputs["prevalence"] = Compartments(where=state["I"]).sum_by()
outputs["prevalence_by_age"] = Compartments(where=state["I"]).sum_by(age)

named = outputs.evaluate(
    compiled.run(params, save=outputs.plan(), t0=0.0, t1=365.0, dt=1.0, epoch=epoch), params
)
assert named["weekly_incidence"].dims == ("time",)
```

## 9. Calibration

Calibration is Bayesian (numpyro). A `Target` compares observations with a named output;
priors say which parameters are free; everything else stays fixed.

```python
from summer4 import Target, TargetSet
from summer4.epi import BayesianModel, Poisson, Uniform

observed = named["weekly_incidence"].to_pandas()["value"].iloc[13:120:7]  # weekly, dated

target = Target.from_series("weekly_incidence", observed, epoch, likelihood=Poisson("sum"))
priors = [Uniform("contact_rate", 0.05, 0.6), Uniform("recovery_time", 5.0, 20.0)]
fixed = {"child_susceptibility": 0.5}
run_kwargs = dict(t0=0.0, t1=365.0, dt=1.0, epoch=epoch)

bm = BayesianModel(compiled, fixed, priors, TargetSet((target,)), outputs=outputs, run_kwargs=run_kwargs)

best = bm.find_map(steps=300).best_params     # quick point estimate
mcmc = bm.sample(num_warmup=50, num_samples=50, num_chains=1, progress_bar=False)
draws = mcmc.idata.posterior["contact_rate"]  # arviz InferenceData
assert abs(float(draws.mean()) - 0.25) < 0.05
```

- `Target.from_series(key, series, epoch, likelihood=...)`: `key` is an `OutputSet` name;
  the series needs a `DatetimeIndex`. Observations are read off the output at those dates.
- Likelihoods: `Poisson`, `NegativeBinomial(dispersion)`, `NormalLikelihood(sd)`; their
  first argument is how per-point log-densities combine. **Pass `"sum"`**; the default,
  `"mean"`, averages them and makes the posterior far too wide.
- Priors: `Uniform`, `NormalPrior`, `TruncatedNormal`, `LogNormal`, `Gamma`, `Beta`.
- `bm.sample(...)` takes `num_warmup`, `num_samples`, `num_chains`, `seed`, `progress_bar`;
  the default kernel is NUTS. `mcmc.idata` is an `arviz.InferenceData` (`az.summary`,
  `az.plot_trace_dist`). R-hat below 1.05 is fine while developing; final results want
  R-hat ≤ 1.01, which usually needs several hundred warm-up and sample draws and 4 chains.
- **If R-hat is bad or ESS tiny**, plot the output being fitted at a few parameter values
  from the prior first. An epidemic that is over before the observations end (an
  all-zero tail), or observations the model cannot reach, make the fit ill-posed; more
  draws will not help.
- `bm.posterior_runs(mcmc.idata, n=200)` reruns the model for posterior draws, for
  uncertainty bands on outputs.

## 10. Pitfalls

1. **Selectors that fan out.** A property named on only one side of a flow is free on the
   other side: `TransitionFlow("vaccination", state["S"] & age["15"], state["R"], rate)`
   sends people aged 15 into `R` in *every* age group, and population is still conserved,
   so nothing looks wrong. Name the property on both sides or neither, and check
   `compiled.edges(name)`.
2. **One namespace for saved names.** Flows, forces of infection and parameters are all
   saved under their own names. Name the force of infection `"foi"`, not the same as the
   infection flow, or one silently replaces the other in `result`.
3. **Default solver is Euler at `dt`.** With `dt=1` it is off by a few percent for fast
   epidemics. Use `solver="tsit5"` when accuracy matters.
4. **Flow outputs are rates, not counts.** Counts per interval: `integrate_intervals()`.
   Totals: `integrate()`. Cumulative counts: `cumulative_integral()`. `cumsum()` adds up
   saved values, so on a rate it is a count only when saves are one day apart.
   (`cumulative()` was removed in 0.3.0a1.)
5. **`rolling(n)` counts saved points**, not days.
6. **Flow outputs need `side=`** to keep a property: `sum_by(age, side="source")`.
7. **Two series combine only on the same time axis** (same saves); otherwise they raise
   *Output time axes differ*. Use `at_times(ts)` to bring one onto the other's times. An
   output reduced over time (`integrate()`, `at(t)`) combines with any series.
8. **`group_by` without `mixing` is not homogeneous mixing** (each group mixes only with
   itself). Omit `group_by` for homogeneous mixing.
9. **Likelihood aggregation defaults to `"mean"`**: use `Poisson("sum")` and friends.
10. **Precision is float64**: importing summer4 turns on JAX's 64-bit mode for the whole
   process. To trade accuracy for speed or memory (or on a GPU without float64), set
   `JAX_ENABLE_X64=0` before starting Python, or call
   `jax.config.update("jax_enable_x64", False)` after `import summer4` and before
   compiling. float32 keeps about 7 significant figures (a compartment of 100 million moves
   in steps of 8 people); compare float32 results with `rtol=1e-4`, float64 with `1e-9`.
11. **Values are JAX arrays, sometimes wrapped.** A time-only output's `.values` is a JAX
    array (`np.asarray(output.values)`). An output that keeps compartments or columns, and
    `compiled.initial_state(params)`, return a `PropertyData`: the array is `.data`.
    `output.to_pandas()` works for both.
12. **Changing the model's structure needs `compile()` again**; changing `params` does not.
13. **`rolling(n)` starts with `n - 1` NaN values**; drop them (`.iloc[n - 1:]`) before
    summing or comparing.
14. **`Split` weights must list every trait and sum to 1**; it raises otherwise (or pass
    `normalize=True`).

## 11. Finding more

This project has no copy of summer4's repository. In order of usefulness:

1. `pixi run api NAME` prints the signature, docstring and methods of any public name
   (`pixi run api Output.rolling`, `pixi run api --search mixing`, `pixi run api` for the
   full list). Docstrings are the authoritative reference for the installed version.
2. `pixi run api --source NAME` prints source; `pixi run api --where` prints the installed
   path and exact commit. The package is plain Python under `.pixi/envs/default/lib/`.
3. With web access, summer4's documentation is published at
   <https://monash-emu.github.io/summer4/>: a guide in the summer2 style (basic model,
   flow types, rates, derived outputs, stratification, mixing, initial population,
   calibration), worked textbook chapters, and release notes with a migration table.
   **The site documents summer4's latest `main`**, which may be ahead of the pinned
   version. When they disagree, `pixi run api` is right for this project.
4. The repository at the pinned commit has the same pages as notebooks (`docs/guide/`,
   `docs/textbook/`) plus acceptance notebooks (`examples/notebooks/`):
   `https://github.com/monash-emu/summer4/tree/<commit from pixi run api --where>`.

## 12. Coming from summer2 or summer3

| summer2 / summer3 | summer4 |
|---|---|
| `CompartmentalModel(compartments=[...])` | `PropertyMap.from_property(Property(...))` + `FlowModel(pmap)` |
| `Stratification(...)`, `model.stratify_with(...)` | `pmap.stratify(prop)`, or `FlowModel.stratify(prop)` after declaring flows |
| `add_transition_flow`, `add_death_flow`, `add_crude_birth_flow` | `TransitionFlow`, `ExitFlow`, `EntryFlow` |
| `add_infection_frequency_flow` | `TransitionFlow(..., ForceOfInfection(...))` |
| `set_mixing_matrix` | `ForceOfInfection(group_by=age, mixing=MixingMatrix(age, matrix))` |
| `set_population_split`, stratification `adjustments` | `Split(...)`; `adjust=[Multiply(...), Overwrite(...)]` on the flow |
| `request_output_for_flow`, `request_output_for_compartments` | `FlowMass(...)`, `Compartments(...)` in an `OutputSet` |
| `request_cumulative_output` | `.cumulative_integral()` (of a flow) |
| `Parameter("x")` / `computegraph` functions | `Param("x")`, plain arithmetic, NumPy ufuncs, `defer(f)` |
| estival `BayesianCompartmentalModel` | `BayesianModel(compiled, fixed, priors, TargetSet(...), outputs=...)` |
