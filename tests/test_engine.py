# tests/test_engine.py
# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox

import numpy as np
import pytest
import qutip

from qblox_sim.engine import QuTiPEngine
from qblox_sim.physics import QuantumSystem


@pytest.fixture
def engine_setup():
    from qblox_sim.config import SimulationConfig

    # WHY: Define finite relaxation times so the Trotterized decay logic actually
    # has a non-zero gamma matrix to apply during idle gaps.
    config_dict = {
        "dt": 1e-9,
        "qubits": {
            "q0": {
                "T1": 10e-6,  # 10 microseconds
                "T2": 10e-6,  # 10 microseconds
            }
        },
        "resonators": {"q0": {}},
    }

    cfg = SimulationConfig.from_dict(config_dict)
    system = QuantumSystem(cfg)
    engine = QuTiPEngine()

    t_list = np.linspace(0, 100e-9, 101)  # 101 points total
    drives = {
        "q0:mw": np.zeros_like(t_list, dtype=complex),
        "q0:res": np.zeros_like(t_list, dtype=complex),
    }

    return system, engine, t_list, drives


@pytest.mark.edge_case
def test_engine_zero_length_chunk_safety(engine_setup):
    """Verify the engine integrates boundaries gracefully without crashing on empty arrays."""
    system, engine, t_list, drives = engine_setup
    initial_state = system.get_default_initial_state()

    # Deliberately feed a zero-length gap where pulse 1 ends and pulse 2 exactly begins
    chunks = [
        ("active", 0, 50),
        ("idle", 50, 50),  # The dangerous zero-length slice
        ("active", 50, 100),
    ]

    res = engine.run(system, drives, t_list, initial_state, chunks=chunks)
    assert len(res.states) == len(t_list)


@pytest.mark.unit
def test_engine_idle_trotterized_decay(engine_setup):
    """Verify O(1) idle chunk execution applies phase/decay accurately over massive gaps."""
    system, engine, t_list, drives = engine_setup

    # Start in excited state
    initial_state = qutip.tensor(
        qutip.basis(system.cfg.qubits["q0"].N_q, 1),
        qutip.basis(system.cfg.resonators["q0"].N_res, 0),
    )

    # Force engine to jump the entire grid using the fast Trotterized idle path
    chunks = [("idle", 0, 100)]
    res = engine.run(system, drives, t_list, initial_state, chunks=chunks)

    assert len(res.states) == 101

    # Ensure the state has dynamically decayed by checking fidelity drop
    assert qutip.fidelity(res.states[0], res.states[-1]) < 0.99
