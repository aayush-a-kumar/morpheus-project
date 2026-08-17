# SPDX-License-Identifier: LicenseRef-Proprietary
import pytest

@pytest.fixture
def default_qubit_params():
    """Provides default multi-qubit topology parameters for single-qubit baseline tests."""
    return {
        "qubits": {
            "q0": {
                "f_q": 5.0e9,
                "f_d": 5.0e9,
                "alpha": -300.0e6,
                "N_q": 3,
                "rabi_freq_per_volt": 10.0e6,
                "T1": float("inf"),
                "T2": float("inf"),
            }
        },
        "resonators": {
            "q0": {
                "f_res": 6.0e9,
                "f_d_res": 6.0e9,
                "N_res": 5,
                "chi": 1.0e6,
                "kappa": 1.0e6,
                "rabi_freq_res_per_volt": 10.0e6,
            }
        },
        "couplings": [],
        "acquisition": {
            "noise_sigma": 0.02,
            "cable_delay": 120e-9,
            "v_0": complex(0.05, 0.05),
            "v_1": complex(-0.05, -0.05),
        },
        "dt": 1.0e-9,
    }