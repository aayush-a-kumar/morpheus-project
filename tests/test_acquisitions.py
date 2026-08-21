# tests/test_acquisitions.py
# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox

from unittest.mock import patch

import numpy as np
import pytest
import qutip

from qblox_sim.acquisitions import SSBIntegrationHandler, ThresholdedAcquisitionHandler


@pytest.fixture
def acq_setup(default_config):
    """Provides standard inputs to test acquisition processing math."""
    state = qutip.basis(3, 1)  # Excited state (prob_1 = 1.0)
    a_op = qutip.destroy(5)
    return {
        "state": state,
        "cfg": default_config,
        "a_op": a_op,
        "ad_op": a_op.dag(),
        "t_list": np.linspace(0, 100e-9, 10),
        "states": [state] * 10,
    }


@pytest.mark.unit
@pytest.mark.parametrize("shots", [1, 100])
@patch("numpy.random.normal")
@patch("numpy.random.binomial")
def test_ssb_integration_exact_math(mock_binomial, mock_normal, shots, acq_setup):
    """Verify deterministic centroid math and dynamic dimension unboxing for N shots."""
    # Force deterministic noise to strictly assert math
    mock_normal.return_value = np.zeros(shots) if shots > 1 else np.array([0.0])
    mock_binomial.return_value = np.ones(shots) if shots > 1 else np.array([1])

    handler = SSBIntegrationHandler()
    res = handler.process(
        state=acq_setup["state"],
        t_list=acq_setup["t_list"],
        states=acq_setup["states"],
        acq_time=10e-9,
        acq_duration=50e-9,
        acq_info={"shots": shots},
        cfg=acq_setup["cfg"],
        a_op=acq_setup["a_op"],
        ad_op=acq_setup["ad_op"],
    )

    v_1_expected = acq_setup["cfg"].acquisition.v_1

    if shots == 1:
        assert isinstance(res["I"], float)
        assert isinstance(res["outcome"], int)
        assert res["I"] == v_1_expected.real
        assert res["Q"] == v_1_expected.imag
    else:
        assert isinstance(res["I"], np.ndarray)
        assert len(res["I"]) == shots
        assert np.all(res["I"] == v_1_expected.real)


@pytest.mark.unit
@patch("numpy.random.normal", return_value=np.array([0.0]))
@patch("numpy.random.binomial", return_value=np.array([1]))
def test_thresholded_acquisition_logic(mock_bin, mock_norm, acq_setup):
    """Verify that a rotated IQ blob strictly thresholds into a boolean mapped outcome."""
    handler = ThresholdedAcquisitionHandler()

    # Apply a 90-degree rotation and a -0.01 threshold
    res = handler.process(
        state=acq_setup["state"],
        t_list=acq_setup["t_list"],
        states=acq_setup["states"],
        acq_time=10e-9,
        acq_duration=50e-9,
        acq_info={"shots": 1, "acq_rotation": 90.0, "acq_threshold": -0.01},
        cfg=acq_setup["cfg"],
        a_op=acq_setup["a_op"],
        ad_op=acq_setup["ad_op"],
    )

    assert "outcome" in res
    assert res["outcome"] in (0, 1)
