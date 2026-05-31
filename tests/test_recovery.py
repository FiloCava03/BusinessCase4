"""Recovery composite + hysteresis-with-override invariants."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from sentinel_alpha.strategy.recovery import recovery_composite, hysteresis_with_override


def _toy_panel(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=n, freq="W")
    return pd.DataFrame(
        {
            "ENG_credit_excess": rng.normal(0.0, 0.005, size=n),
            "ENG_vix_d": rng.normal(0.0, 1.5, size=n),
            "ENG_eq_dm": rng.normal(0.001, 0.02, size=n),
        },
        index=idx,
    )


def test_recovery_composite_no_nan_output():
    F = _toy_panel()
    rec = recovery_composite(F)
    assert not rec.isna().any()


def test_recovery_composite_requires_inputs():
    F = _toy_panel().drop(columns=["ENG_eq_dm"])
    with pytest.raises(KeyError):
        recovery_composite(F)


def test_hysteresis_override_strong_forces_exit():
    """In state 1, a single rec[t] > thr must force exit even before dwell."""
    signal = np.array([0.9] * 5 + [0.2] * 2, dtype=float)
    p = np.full_like(signal, 0.5)
    # rec is huge only at week 5 (well after we've entered state 1).
    rec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0])
    states = hysteresis_with_override(
        signal, p, rec, enter=0.5, exit_=0.3, dwell=3,
        override_kind="strong", thr=1.5,
    )
    # By week 3 we're in state 1 (dwell=3 super-threshold weeks).
    # At week 5 the override should force-exit even though signal is 0.2 only briefly.
    assert states[3] == 1
    assert states[5] == 0, f"override should have forced exit at t=5; got {states.tolist()}"


def test_hysteresis_override_none_matches_plain_dwell():
    """override_kind='none' must produce identical output to plain hysteresis dwell."""
    from sentinel_alpha.strategy.state_machine import hysteresis
    rng = np.random.default_rng(2)
    signal = rng.uniform(0.0, 1.0, size=80)
    p = rng.uniform(0.0, 1.0, size=80)
    rec = rng.normal(0.0, 1.0, size=80)
    a = hysteresis_with_override(signal, p, rec, enter=0.5, exit_=0.3, dwell=2,
                                 override_kind="none")
    b = hysteresis(signal, enter=0.5, exit_=0.3, dwell=2)
    np.testing.assert_array_equal(a, b)


def test_hysteresis_override_joint_requires_both_conditions():
    signal = np.array([0.9] * 4 + [0.2] * 4, dtype=float)
    # rec high at t=4 but p_raw also high (so joint condition fails).
    p_high = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5])
    rec_high = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    states = hysteresis_with_override(
        signal, p_high, rec_high, enter=0.5, exit_=0.3, dwell=3,
        override_kind="joint", thr=1.5,
    )
    # Joint condition fails (p=0.9 >= 0.40), so override should NOT force exit at t=4.
    # State should remain 1 because dwell-3 exit hasn't completed yet.
    assert states[4] == 1
