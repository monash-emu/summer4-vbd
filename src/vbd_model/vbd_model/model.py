"""A basic synthetic vector-borne disease model: humans and mosquitoes.

Humans are SEIRS and age-stratified; mosquitoes are SEI and unstratified. The two
populations are coupled by Ross-Macdonald biting:

- Each mosquito bites ``bite_rate * (1 - bite_reduction)`` times per day, where
  ``bite_reduction`` (an intervention such as bednets) applies from ``intervention_start``. A bite lands on age group ``j`` with
  probability ``w_j N_j / W``, where ``w_j`` is that group's relative attractiveness
  (``rel_bite_<j>``) and ``W = sum_j w_j N_j`` is the bite-weighted human population.
- Force of infection on susceptible humans in age ``j``:
  ``bite_rate * p_mh * w_j * s_j * I_m / W``, with ``s_j`` the relative susceptibility
  (``rel_susc_<j>``).
- Force of infection on susceptible mosquitoes:
  ``bite_rate * p_hm * sum_j w_j I_j / W``.

Human births replace deaths, so the human population is constant. Mosquito emergence is
seasonal: ``M0 / mosquito_lifespan * (1 + seasonal_amplitude * cos(2 pi (t - peak_emergence_day) / 365))``,
so the mosquito population cycles each year around a mean of ``M0`` (``mosquitoes_per_human``
per human). Time is in days from 1 January.
"""

from __future__ import annotations

import numpy as np

from summer4 import (
    Time,
    EntryFlow,
    ExitFlow,
    FlowModel,
    Multiply,
    Param,
    Property,
    PropertyMap,
    Reduce,
    Split,
    TraitChain,
    TransitionFlow,
)
from summer4.timevarying import step

species = Property("species", ["human", "mosquito"])
hstate = Property("hstate", ["S", "E", "I", "R"])
mstate = Property("mstate", ["S", "E", "I"])
age = Property("age", ["0", "5", "15", "40"])  # lower bounds of the age bands, in years

HUMANS = 100_000.0
MOSQUITOES_PER_HUMAN = 5.0  # mean mosquito density over the seasonal cycle
OLDEST_BAND_YEARS = 30.0  # mean time spent in the 40+ band before death
# Band widths in years; the initial age split is proportional to them, which is the
# equilibrium of the ageing chain, so the age structure does not drift during a run.
BAND_YEARS = {"0": 5.0, "5": 10.0, "15": 25.0, "40": OLDEST_BAND_YEARS}

DEFAULT_PARAMS = {
    # Vector
    "bite_rate": 0.4,  # bites per mosquito per day
    "p_mh": 0.5,  # probability an infectious bite infects a human
    "p_hm": 0.4,  # probability a bite on an infectious human infects the mosquito
    "mosquito_lifespan": 10.0,  # days
    "eip": 10.0,  # extrinsic incubation period, days
    "seasonal_amplitude": 0.8,  # relative swing in emergence (0 = no seasonality, < 1)
    "peak_emergence_day": 45.0,  # day of year when emergence peaks
    # Intervention: proportional reduction in bites per mosquito, from intervention_start
    "bite_reduction": 0.0,
    "intervention_start": 5 * 365.0,  # days
    # Human natural history
    "human_latent": 5.0,  # days
    "human_infectious": 7.0,  # days
    "immunity_duration": 365.0,  # days
    # Age heterogeneity: relative attractiveness to mosquitoes ...
    "rel_bite_0": 0.5,
    "rel_bite_5": 0.8,
    "rel_bite_15": 1.0,
    "rel_bite_40": 1.1,
    # ... and relative susceptibility per infectious bite (acquired partial protection)
    "rel_susc_0": 1.0,
    "rel_susc_5": 0.9,
    "rel_susc_15": 0.7,
    "rel_susc_40": 0.6,
}


def build_pmap() -> PropertyMap:
    """Ragged compartments: humans by state and age, mosquitoes by state only."""
    return (
        PropertyMap.from_property(species)
        .stratify(hstate, where=species["human"])
        .stratify(age, where=species["human"])
        .stratify(mstate, where=species["mosquito"])
    )


def _weighted_humans(where=None):
    """sum_j rel_bite_j * N_j over age groups (restricted to ``where`` if given)."""
    base = species["human"] if where is None else where
    return sum(Param(f"rel_bite_{a}") * Reduce(where=base & age[a]) for a in age.traits)


def build_model(seed_infectious: float = 0.001, mosquitoes_per_human: float = MOSQUITOES_PER_HUMAN) -> FlowModel:
    """Build the uncompiled model.

    ``seed_infectious`` is the fraction of humans infectious at time 0 (mosquitoes start
    uninfected). ``mosquitoes_per_human`` sets the initial and mean mosquito population;
    it is structural, so changing it needs a rebuild.
    """
    model = FlowModel(build_pmap())
    mean_mosquitoes = HUMANS * mosquitoes_per_human

    # Bites per mosquito per day, reduced by the intervention once it starts
    bites = Param("bite_rate") * (
        1.0 - step(Time(), [Param("intervention_start")], [0.0, Param("bite_reduction")])
    )

    bite_weighted = _weighted_humans()
    infectious_mosquitoes = Reduce(where=mstate["I"])

    # Humans: infection, with per-age bite and susceptibility multipliers
    human_foi = bites * Param("p_mh") * infectious_mosquitoes / bite_weighted
    model.add_flow(
        TransitionFlow(
            "human_infection",
            hstate["S"],
            hstate["E"],
            human_foi,
            adjust=[
                Multiply(Param(f"rel_bite_{a}") * Param(f"rel_susc_{a}"), where=age[a])
                for a in age.traits
            ],
        )
    )
    model.add_flow(TransitionFlow("human_onset", hstate["E"], hstate["I"], 1.0 / Param("human_latent")))
    model.add_flow(TransitionFlow("human_recovery", hstate["I"], hstate["R"], 1.0 / Param("human_infectious")))
    model.add_flow(TransitionFlow("waning", hstate["R"], hstate["S"], 1.0 / Param("immunity_duration")))

    # Human demography: ageing through the bands, death from the oldest, births = deaths
    model.add_flow(
        TransitionFlow(
            "ageing",
            age.present(),
            age.present(),
            1.0 / 365.0,
            pairing=TraitChain(
                age,
                (("0", "5"), ("5", "15"), ("15", "40")),
                rates=tuple(1.0 / BAND_YEARS[a] for a in ("0", "5", "15")),
            ),
        )
    )
    human_death = model.add_flow(ExitFlow("human_death", age["40"], 1.0 / (OLDEST_BAND_YEARS * 365.0)))
    model.add_flow(EntryFlow("human_birth", hstate["S"] & age["0"], human_death.sum_by()))

    # Mosquitoes: infection from biting infectious humans, extrinsic incubation
    mosquito_foi = bites * Param("p_hm") * _weighted_humans(hstate["I"]) / bite_weighted
    model.add_flow(TransitionFlow("mosquito_infection", mstate["S"], mstate["E"], mosquito_foi))
    model.add_flow(TransitionFlow("mosquito_incubation", mstate["E"], mstate["I"], 1.0 / Param("eip")))

    # Mosquito demography: seasonal emergence (people per day), averaging mean_mosquitoes / lifespan
    model.add_flow(ExitFlow("mosquito_death", species["mosquito"], 1.0 / Param("mosquito_lifespan")))
    season = 1.0 + Param("seasonal_amplitude") * np.cos(
        2 * np.pi * (Time() - Param("peak_emergence_day")) / 365.0
    )
    model.add_flow(
        EntryFlow("mosquito_emergence", mstate["S"], mean_mosquitoes / Param("mosquito_lifespan") * season)
    )

    total_years = sum(BAND_YEARS.values())
    base = {age[a]: HUMANS * BAND_YEARS[a] / total_years for a in age.traits}
    base[mstate["S"]] = mean_mosquitoes
    model.set_initial_population(
        base,
        [
            Split(
                hstate,
                {"S": 1.0 - seed_infectious, "E": 0.0, "I": seed_infectious, "R": 0.0},
                where=species["human"],
            )
        ],
    )
    return model


def r0(params: dict, mosquitoes_per_human: float = MOSQUITOES_PER_HUMAN) -> float:
    """Basic reproduction number from the two-type next-generation matrix.

    Ignores human mortality during infection (negligible). Uses the equilibrium age
    distribution set by ``build_model``, the mean mosquito density (so with seasonality
    this is the season-averaged R0; at peak density it is up to ``sqrt(1 + amplitude)``
    times higher), and the bite rate with ``bite_reduction`` applied.
    """
    p = {**DEFAULT_PARAMS, **params}
    m = mosquitoes_per_human
    total_years = sum(BAND_YEARS.values())
    n = np.array([HUMANS * BAND_YEARS[a] / total_years for a in age.traits])
    w = np.array([p[f"rel_bite_{a}"] for a in age.traits])
    s = np.array([p[f"rel_susc_{a}"] for a in age.traits])
    weighted = (w * n).sum()
    mu = 1.0 / p["mosquito_lifespan"]
    survive_eip = (1.0 / p["eip"]) / (1.0 / p["eip"] + mu)
    a = p["bite_rate"] * (1.0 - p["bite_reduction"])
    # Humans in age j infected by one mosquito over its infectious life
    m_to_h = a / mu * p["p_mh"] * w * s * n / weighted
    # Mosquitoes infected (and surviving the EIP) by one infectious human in age j
    h_to_m = p["human_infectious"] * a * p["p_hm"] * m * HUMANS * w / weighted * survive_eip
    return float(np.sqrt((m_to_h * h_to_m).sum()))
