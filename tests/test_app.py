"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und Quelle, Pivot-Slider, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import xov_constants as C
import xov_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    state.setdefault("xov_step", step)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_vertex_result():
    at = _run()
    _ok(at)
    assert {"Quelle", "Aufräum-Pivots", "Simplex von Null", "Ergebnis"} <= {m.label for m in at.metric}
    assert any("Ecke gefunden" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run(kind_select="centre")
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["density_select"], ss["source_select"], ss["eps_select"], ss["scale_select"], ss["change_select"], ss["xov_step"]) == (p["kind"], p["density"], p["source"], p["eps"], p["scale"], p["change"], p["step"])
    if "pivot_k" in p and "pivot_k" in ss:
        assert ss["pivot_k"] == p["pivot_k"]


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind_and_source(step, kind):
    for source in C.SOURCES:
        at = _run(step=step, kind_select=kind, m_slider=6, n_slider=6, source_select=source)
        _ok(at)
        assert at.session_state["xov_step"] == step


def test_pivot_slider_on_two_variables_and_every_position():
    at = _run(step=1, kind_select="degenerate")
    _ok(at)
    assert at.session_state["pivot_k"] == 1
    for k in (0, 1):
        at.session_state["pivot_k"] = k
        at.run()
        _ok(at)
    zero = _run(step=1, kind_select="centre")
    _ok(zero)
    assert any("kein Bild" in md.value for md in zero.markdown)
    none = _run(step=1, kind_select="textbook", eps_select=2)
    _ok(none)
    assert any("0 Pivots" in md.value for md in none.markdown) and "pivot_k" not in none.session_state


def test_status_messages_reflect_the_run():
    inf = _run(step=1, kind_select="infeasible")
    _ok(inf)
    assert any("liefert hier keine Näherung" in i.value for i in inf.info)
    unb = _run(step=3, kind_select="unbounded")
    _ok(unb)
    assert any("keine Näherung" in i.value for i in unb.info)
    plateau = _run(step=1, kind_select="plateau", m_slider=8, n_slider=10)
    _ok(plateau)
    assert any("Ecke gefunden" in s.value for s in plateau.success)


def test_on_demand_experiments():
    two = _run(step=2)
    _click(two, "eps_start")
    _ok(two)
    assert len(two.get("plotly_chart")) >= 2 and any("schon optimal / primal / dual / beides" in d.value.columns for d in two.dataframe)
    three = _run(step=3)
    _click(three, "err_start")
    _ok(three)
    four = _run(step=4, eps_select=2)
    for key in ("size_start", "cube_start", "warm_start"):
        _click(four, key)
        _ok(four)
    assert len(four.get("plotly_chart")) >= 3
    five = _run(step=5)
    _click(five, "deg_start")
    _ok(five)
    assert any("Instanzart" in d.value.columns for d in five.dataframe)


@pytest.mark.parametrize("kw", [dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX, density_select=len(C.DENSITIES) - 1),
                                dict(kind_select="mixed", m_slider=C.M_MIN, n_slider=C.N_MAX, density_select=0), dict(kind_select="plateau", m_slider=C.M_MAX, n_slider=C.N_MIN), dict(kind_select="klee_minty", n_slider=S.CUBE_MAX)])
def test_extreme_sizes_run_on_every_step(kw):
    for step in (1, 2, 3, 4, 5):
        _ok(_run(step=step, **kw))


@pytest.mark.parametrize("kw", [dict(eps_select=0), dict(eps_select=len(C.EPS_EXPS) - 1), dict(scale_select=len(C.SCALE_EXPS) - 1), dict(change_select=0), dict(change_select=len(C.CHANGES) - 1)])
def test_extreme_controls_on_every_kind(kw):
    for kind in ("textbook", "random", "plateau", "klee_minty", "infeasible"):
        for source in C.SOURCES:
            _ok(_run(step=4, kind_select=kind, m_slider=6, n_slider=6, source_select=source, **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(m="999", n="1", step="9", kind="nope", dens="99", source="gpu", eps="-1", scale="99", change="-4", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["xov_step"], ss["kind_select"], ss["density_select"], ss["source_select"], ss["eps_select"], ss["scale_select"], ss["change_select"], ss["seed_input"]) == (
        C.M_MAX, C.N_MIN, 1, "random", len(C.DENSITIES) - 1, "pdlp", 0, len(C.SCALE_EXPS) - 1, 0, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(kind="plateau", m="10", n="9", seed="7", dens="1", source="ipm", eps="3", scale="2", change="2", step="4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["seed_input"], ss["density_select"], ss["source_select"], ss["eps_select"], ss["scale_select"], ss["change_select"], ss["xov_step"]) == (
        "plateau", 10, 9, 7, 1, "ipm", 3, 2, 2, 4)
    assert at.query_params["source"] == ["ipm"] and at.query_params["change"] == ["2"]


def test_sidebar_shows_the_controls_that_belong_to_the_instance_and_the_step():
    fixed = _run(kind_select="centre")
    assert not any(w.key in ("m_widget", "n_widget") for w in fixed.slider) and not any(s.key == "density_widget" for s in fixed.select_slider)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "n_widget" for w in rnd.slider) and any(s.key == "density_widget" for s in rnd.select_slider)
    assert not any(s.key == "change_widget" for s in rnd.select_slider)
    warm = _run(step=4)
    assert any(s.key == "change_widget" for s in warm.select_slider)
    cube = _run(kind_select="klee_minty")
    assert any(w.key == "n_widget" for w in cube.slider) and not any(w.key == "m_widget" for w in cube.slider)


def test_changing_kind_and_step_on_later_steps_does_not_crash():
    for step in (1, 2, 3, 4, 5):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=8, n_slider=8), dict(kind_select="infeasible"), dict(kind_select="unbounded"), dict(kind_select="textbook"), dict(kind_select="plateau", m_slider=6, n_slider=8),
                   dict(kind_select="klee_minty", n_slider=8), dict(kind_select="random", m_slider=4, n_slider=3, source_select="ipm")):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Megiddo" in m.value and "Bixby" in m.value for e in at.expander for m in e.markdown)
