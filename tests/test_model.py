"""Checks on the vector-borne model: structure, conservation, rates and the R0 threshold."""

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


def test_populations_constant(result):
    comps = result["compartments"]
    humans = np.asarray(comps.select(species["human"]).sum_by().values)
    mosquitoes = np.asarray(comps.select(species["mosquito"]).sum_by().values)
    np.testing.assert_allclose(humans, HUMANS, rtol=1e-9)
    np.testing.assert_allclose(mosquitoes, HUMANS * MOSQUITOES_PER_HUMAN, rtol=1e-9)
    by_age = np.asarray(comps.select(species["human"]).sum_by(age).to_pandas())
    np.testing.assert_allclose(by_age[-1], by_age[0], rtol=1e-9)


def test_infection_rates_match_hand_calculation(result):
    """Flow values at day 30 equal the Ross-Macdonald rate times the source stock."""
    p = DEFAULT_PARAMS
    k = 30
    comps = result["compartments"]
    stock = lambda sel: np.asarray(comps.select(sel).sum_by(age).to_pandas())[k]  # noqa: E731
    n, i_h, s_h = stock(species["human"]), stock(hstate["I"]), stock(hstate["S"])
    w = np.array([p[f"rel_bite_{a}"] for a in age.traits])
    s = np.array([p[f"rel_susc_{a}"] for a in age.traits])
    weighted = (w * n).sum()
    mosq = np.asarray(comps.select(mstate.present()).to_pandas())[k]  # S, E, I

    human_expected = p["bite_rate"] * p["p_mh"] * mosq[2] / weighted * w * s * s_h
    human_actual = np.asarray(result["human_infection"].to_pandas())[k]
    np.testing.assert_allclose(human_actual, human_expected, rtol=1e-9)

    mosq_expected = p["bite_rate"] * p["p_hm"] * (w * i_h).sum() / weighted * mosq[0]
    mosq_actual = float(np.asarray(result["mosquito_infection"].sum_by().values)[k])
    assert mosq_actual == pytest.approx(mosq_expected, rel=1e-9)


@pytest.mark.parametrize("target_r0, grows", [(0.9, False), (1.1, True)])
def test_r0_is_the_threshold(compiled, target_r0, grows):
    # R0 is proportional to bite_rate, so rescale it to hit the target exactly
    bite_rate = DEFAULT_PARAMS["bite_rate"] * target_r0 / r0(DEFAULT_PARAMS)
    params = {**DEFAULT_PARAMS, "bite_rate": bite_rate}
    assert r0(params) == pytest.approx(target_r0)
    res = compiled.run(params, t0=0.0, t1=720.0, dt=1.0, solver="tsit5")
    incidence = np.asarray(res["human_infection"].sum_by().values)
    # Compare after the initial transient, while depletion of susceptibles is negligible
    assert (incidence[720] > incidence[360]) == grows
