# tests/test_signals.py
# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox

import numpy as np
import pandas as pd
import pytest

from qblox_sim.signals import ScheduleSignalProvider, extract_amplitude


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_val, default_val, expected",
    [
        ({"amp": 0.5}, 0.0, 0.5),
        ({"amplitude": 0.75}, 0.0, 0.75),
        ({"missing": 1.0}, 0.1, 0.1),
        (pd.Series({"amplitude": 0.8}), 0.0, 0.8),
        ("invalid_input", 0.2, 0.2),
    ],
)
def test_extract_amplitude(input_val, default_val, expected):
    """Verify amplitude extraction handles dicts, pandas Series, and invalid types safely."""
    assert extract_amplitude(input_val, default=default_val) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "wf_func, amp, has_imaginary_component",
    [
        ("square", 0.5, False),
        ("gauss", 1.0, False),
        ("drag", 0.5, True),  # DRAG pulses apply derivative on the Q (imaginary) axis
    ],
)
def test_pulse_envelope_generation(wf_func, amp, has_imaginary_component):
    """Verify that different waveform functions generate correct arrays without off-by-one errors."""
    t_list = np.linspace(0, 100e-9, 100)
    pulses = [
        {
            "port": "q0:mw",
            "abs_time": 10e-9,
            "duration": 40e-9,
            "amp": amp,
            "phase": 0.0,
            "wf_func": wf_func,
            # WHY: Scale beta to match the unnormalized SI time derivative
            "beta": 1e-10 if wf_func == "drag" else 0.0,
            "sigma": 10e-9,  # Used by gauss and drag
        }
    ]

    provider = ScheduleSignalProvider(pulses)
    drives = provider.get_drives(t_list)

    assert "q0:mw" in drives
    q_drive = drives["q0:mw"]

    # Assert grid dimensions precisely match the solver input requirements
    assert len(q_drive) == len(t_list)

    # Verify the peak amplitude roughly matches the requested amplitude
    assert np.isclose(np.max(np.abs(q_drive)), amp, atol=0.05)

    if has_imaginary_component:
        assert np.any(np.imag(q_drive) != 0.0)
    else:
        assert np.all(np.imag(q_drive) == 0.0)


@pytest.mark.unit
def test_virtual_z_phase_tracking():
    """Verify Virtual-Z gates rotate the complex microwave envelope into the correct quadrature."""
    t_list = np.linspace(0, 100e-9, 100)
    pulses_list = [
        {
            "port": "q0:mw",
            "abs_time": 0.0,
            "duration": 20e-9,
            "amp": 1.0,
            "phase": 0.0,  # Pure I-quadrature
        },
        {
            "port": "q0:mw",
            "abs_time": 50e-9,
            "duration": 20e-9,
            "amp": 1.0,
            "phase": 90.0,  # Pure Q-quadrature after 90 deg Virtual Z
        },
    ]

    provider = ScheduleSignalProvider(pulses_list)
    drives = provider.get_drives(t_list)
    mw_drive = drives["q0:mw"]

    # Pulse 1 (0 to 20ns): Expected strictly Real
    pulse_1_slice = mw_drive[5:15]
    np.testing.assert_allclose(np.real(pulse_1_slice), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.imag(pulse_1_slice), 0.0, atol=1e-12)

    # Pulse 2 (50 to 70ns): Expected strictly Imaginary
    pulse_2_slice = mw_drive[55:65]
    np.testing.assert_allclose(np.real(pulse_2_slice), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.imag(pulse_2_slice), 1.0, atol=1e-12)


@pytest.mark.edge_case
def test_signal_provider_grid_boundaries():
    """Verify out-of-bounds pulses and empty time grids resolve safely without crashing."""
    t_list = np.linspace(0, 100e-9, 100)

    # 1. Out-of-bounds pulse
    p_out = ScheduleSignalProvider(
        [{"port": "q0:mw", "abs_time": 500e-9, "duration": 10e-9, "amp": 0.5}]
    )
    assert np.all(p_out.get_drives(t_list).get("q0:mw", 0) == 0)

    # 2. Degenerate 1-point time grid
    p_grid = ScheduleSignalProvider(
        [{"port": "q0:mw", "abs_time": 0.0, "duration": 10e-9, "amp": 0.5}]
    )
    assert p_grid.get_drives(np.array([0.0])) == {}
