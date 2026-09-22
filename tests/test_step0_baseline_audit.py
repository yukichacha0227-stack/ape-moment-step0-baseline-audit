"""Small synthetic checks for the independent Step 0 spectral audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "step0_baseline_audit.py"
SPEC = importlib.util.spec_from_file_location("step0_baseline_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_constant_spectrum_trapezoid_and_legacy_differ_at_endpoints():
    moments = audit.spectral_moments(np.ones(701))
    assert np.isclose(moments["S0_trapezoid_W_m2"], 0.700)
    assert np.isclose(moments["S0_legacy_W_m2"], 0.701)
    assert np.isclose(moments["S1_trapezoid_W_m2_nm"], 490.0)
    assert np.isclose(moments["S1_legacy_W_m2_nm"], 490.7)
    assert np.isclose(moments["ape_trapezoid_eV"], moments["ape_legacy_eV"])


def test_hourly_a_and_b_are_not_assumed_equal():
    spectra = []
    for minute, first, second in ((0, 1.0, 1000.0), (30, 10.0, 100.0)):
        data = np.zeros(701)
        data[0], data[-1] = first, second
        moments = audit.spectral_moments(data)
        spectra.append({
            "datetime": pd.Timestamp(f"2026-01-01 09:{minute:02d}", tz="Asia/Tokyo"),
            "SiteNum": 301,
            "surface": "HSR",
            "ape_target_eV": moments["ape_legacy_eV"],
            **moments,
        })
    hourly = audit.hourly_moments(pd.DataFrame(spectra))
    assert len(hourly) == 1
    assert hourly.loc[0, "n_spectra"] == 2
    assert not np.isclose(hourly.loc[0, "ape_trapezoid_A_eV"],
                          hourly.loc[0, "ape_trapezoid_B_eV"])
    assert hourly.loc[0, "datetime"] == pd.Timestamp("2026-01-01 00:30Z")


def test_jst_source_midnight_rollover():
    assert audit.parse_source_datetime("2026/01/01", "24:00") == pd.Timestamp("2026-01-02")

