# SPDX-License-Identifier: LicenseRef-Proprietary
import pytest
from qblox_sim.config import SimulationConfig, QubitConfig, ResonatorConfig, CouplingConfig


def test_config_defaults():
    cfg = SimulationConfig()
    assert cfg.qubit.f_q == 5.0e9
    assert cfg.qubit.N_q == 3
    assert cfg.resonator.N_res == 5


def test_multi_qubit_config_parsing():
    multi_params = {
        "qubits": {
            "q0": QubitConfig(f_q=5.0e9),
            "q1": QubitConfig(f_q=5.2e9)
        },
        "resonators": {
            "q0": ResonatorConfig(f_res=6.0e9),
            "q1": ResonatorConfig(f_res=6.2e9)
        },
        "couplings": [CouplingConfig(q1="q0", q2="q1", J=10e6)]
    }
    cfg = SimulationConfig.from_dict(multi_params)
    
    assert len(cfg.qubits) == 2
    assert cfg.qubits["q1"].f_q == 5.2e9
    assert len(cfg.couplings) == 1
    assert cfg.couplings[0].J == 10e6