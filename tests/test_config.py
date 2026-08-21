# tests/test_config.py
# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox

import pytest

from qblox_sim.config import (
    CouplingConfig,
    QubitConfig,
    ResonatorConfig,
    SimulationConfig,
)


@pytest.mark.unit
def test_config_dataclass_defaults():
    """Verify that pure dataclass instantiation yields expected default physical values."""
    cfg = SimulationConfig()

    assert cfg.qubit.f_q == 5.0e9
    assert cfg.qubit.N_q == 3
    assert cfg.resonator.N_res == 5
    assert cfg.acquisition.noise_sigma == 0.02


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw_dict, expected_q_count, expected_r_count, expected_c_count",
    [
        # Standard multi-qubit parsing
        (
            {
                "qubits": {"q0": QubitConfig(f_q=5.0e9), "q1": QubitConfig(f_q=5.2e9)},
                "resonators": {"q0": ResonatorConfig(), "q1": ResonatorConfig()},
                "couplings": [CouplingConfig(q1="q0", q2="q1", J=10e6)],
            },
            2,
            2,
            1,
        ),
        # Parsing with missing sections (graceful fallback)
        (
            {"qubits": {"q0": QubitConfig()}},
            1,
            0,
            0,
        ),
    ],
)
def test_simulation_config_parsing(
    raw_dict, expected_q_count, expected_r_count, expected_c_count
):
    """Verify that dictionaries are correctly parsed into strongly-typed configurations."""
    cfg = SimulationConfig.from_dict(raw_dict)

    assert len(cfg.qubits) == expected_q_count
    assert len(cfg.resonators) == expected_r_count
    assert len(cfg.couplings) == expected_c_count


@pytest.mark.edge_case
def test_primary_component_property_fallbacks():
    """Verify that cfg.qubit and cfg.resonator properties fallback safely if 'q0' is missing."""
    cfg = SimulationConfig.from_dict(
        {"qubits": {"q_alt": {"f_q": 7.0e9}}, "resonators": {"r_alt": {"f_res": 8.0e9}}}
    )

    # Should fallback to the first available component instead of crashing
    assert cfg.qubit.f_q == 7.0e9
    assert cfg.resonator.f_res == 8.0e9

    # Should safely return an empty default dataclass if completely empty
    empty_cfg = SimulationConfig.from_dict({"qubits": {}, "resonators": {}})
    assert isinstance(empty_cfg.qubit, QubitConfig)
    assert empty_cfg.qubit.f_q == 5.0e9
