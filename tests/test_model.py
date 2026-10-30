"""Checks on the vector-borne model: structure, populations, rates, seasonality and R0."""

from __future__ import annotations

import numpy as np
import pytest

from vbd_model.model import (
    DEFAULT_PARAMS,
    HUMANS,
    MOSQUITOES_PER_HUMAN,
    age,
    build_model,
    hstate,
    mstate,
    r0,
    species,
)


@pytest.fixture(scope="module")
def compiled():
    return build_model().compile()


@pytest.fixture(scope="module")
def result(compiled):
    res = compiled.run(DEFAULT_PARAMS, t0=0.0, t1=2 * 365.0, dt=1.0, solver="tsit5")
    assert res.solver.ok
    return res


def edge_labels(compiled, name):
    labels = compiled.pmap.labels()
    edges = compiled.edges(name)
    side = lambda idx: [labels[i] for i in idx] if idx is not None else None  # noqa: E731
    return side(edges.src_idx), side(edges.dest_idx)


def test_transitions_stay_within_species_and_age(compiled):
    for name in ["human_infection", "human_onset", "human_recovery", "waning"]:
        src, dest = edge_labels(compiled, name)
        assert len(src) == len(age.traits)
        for s, d in zip(src, dest):
            assert s.split("_hstate=")[1][1:] == d.split("_hstate=")[1][1:]  # same age
    src, dest = edge_labels(compiled, "mosquito_infection")
    assert (src, dest) == (["species=mosquito_mstate=S"], ["species=mosquito_mstate=E"])
    _, dest = edge_labels(compiled, "human_birth")
    assert dest == ["species=human_hstate=S_age=0"]


def test_human_population_constant(result):
    comps = result["compartments"]
    humans = np.asarray(comps.select(species["human"]).sum_by().values)
    np.testing.assert_allclose(humans, HUMANS, rtol=1e-9)
    by_age = np.asarray(comps.select(species["human"]).sum_by(age).to_pandas())
    np.testing.assert_allclose(by_age[-1], by_age[0], rtol=1e-9)


def test_mosquito_population_seasonal(result):
    comps = result["compartments"]
    mosquitoes = np.asarray(comps.select(species["mosquito"]).sum_by().values)
    # Change in mosquitoes = emergence - deaths, integrated over the run
    emerged = float(result["mosquito_emergence"].sum_by().integrate().scalar())
    died = float(result["mosquito_death"].sum_by().integrate().scalar())
    assert mosquitoes[-1] - mosquitoes[0] == pytest.approx(emerged - died, rel=1e-3)
    # Over the second year (past the start-up transient) the mean is the nominal density
    # and the population swings with the seasonal amplitude
    second_year = mosquitoes[365:730]
    mean = HUMANS * MOSQUITOES_PER_HUMAN
    assert second_year.mean() == pytest.approx(mean, rel=0.01)
    assert second_year.max() > 1.5 * mean and second_year.min() < 0.5 * mean
    # Peak density lags the emergence peak by about a mosquito lifespan or less
    peak_day = int(np.argmax(second_year))
    assert 0 < peak_day - DEFAULT_PARAMS["peak_emergence_day"] < DEFAULT_PARAMS["mosquito_lifespan"] * 2


def test_no_seasonality_keeps_mosquitoes_constant(compiled):
    params = {**DEFAULT_PARAMS, "seasonal_amplitude": 0.0}
    res = compiled.run(params, t0=0.0, t1=365.0, dt=1.0, solver="tsit5")
    mosquitoes = np.asarray(res["compartments"].select(species["mosquito"]).sum_by().values)
    np.testing.assert_allclose(mosquitoes, HUMANS * MOSQUITOES_PER_HUMAN, rtol=1e-9)


@pytest.mark.parametrize("bite_reduction, intervention_start", [(0.0, 365.0), (0.25, 20.0), (0.25, 40.0)])
def test_infection_rates_match_hand_calculation(compiled, bite_reduction, intervention_start):
    """Flow values at day 30 equal the Ross-Macdonald rate times the source stock.

    The intervention cuts bites per mosquito only once it has started.
    """
    p = {**DEFAULT_PARAMS, "bite_reduction": bite_reduction, "intervention_start": intervention_start}
    result = compiled.run(p, t0=0.0, t1=60.0, dt=1.0, solver="tsit5")
    k = 30
    bites = p["bite_rate"] * (1 - (bite_reduction if k >= intervention_start else 0.0))
    comps = result["compartments"]
    stock = lambda sel: np.asarray(comps.select(sel).sum_by(age).to_pandas())[k]  # noqa: E731
    n, i_h, s_h = stock(species["human"]), stock(hstate["I"]), stock(hstate["S"])
    w = np.array([p[f"rel_bite_{a}"] for a in age.traits])
    s = np.array([p[f"rel_susc_{a}"] for a in age.traits])
    weighted = (w * n).sum()
    mosq = np.asarray(comps.select(mstate.present()).to_pandas())[k]  # S, E, I

    human_expected = bites * p["p_mh"] * mosq[2] / weighted * w * s * s_h
    human_actual = np.asarray(result["human_infection"].to_pandas())[k]
    np.testing.assert_allclose(human_actual, human_expected, rtol=1e-9)

    mosq_expected = bites * p["p_hm"] * (w * i_h).sum() / weighted * mosq[0]
    mosq_actual = float(np.asarray(result["mosquito_infection"].sum_by().values)[k])
    assert mosq_actual == pytest.approx(mosq_expected, rel=1e-9)


def test_r0_includes_bite_reduction():
    reduced = {**DEFAULT_PARAMS, "bite_reduction": 0.2}
    assert r0(reduced) == pytest.approx(0.8 * r0(DEFAULT_PARAMS))


@pytest.mark.parametrize("target_r0, grows", [(0.9, False), (1.1, True)])
def test_r0_is_the_threshold(compiled, target_r0, grows):
    # R0 is proportional to bite_rate, so rescale it to hit the target exactly. Without
    # seasonality, so the mosquito density is constant at the mean R0 assumes.
    base = {**DEFAULT_PARAMS, "seasonal_amplitude": 0.0}
    bite_rate = base["bite_rate"] * target_r0 / r0(base)
    params = {**base, "bite_rate": bite_rate}
    assert r0(params) == pytest.approx(target_r0)
    res = compiled.run(params, t0=0.0, t1=720.0, dt=1.0, solver="tsit5")
    incidence = np.asarray(res["human_infection"].sum_by().values)
    # Compare after the initial transient, while depletion of susceptibles is negligible
    assert (incidence[720] > incidence[360]) == grows
