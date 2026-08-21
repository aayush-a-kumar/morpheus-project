# tests/test_physics.py
# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox

import numpy as np
import pytest
import qutip

from qblox_sim.config import (
    CouplingConfig,
    QubitConfig,
    ResonatorConfig,
    SimulationConfig,
)
from qblox_sim.physics import QuantumSystem


@pytest.mark.unit
def test_global_tensor_dimensions():
    """Verify that multi-qubit and multi-resonator dimensions are calculated correctly."""
    cfg = SimulationConfig(
        qubits={"q0": QubitConfig(N_q=2), "q1": QubitConfig(N_q=3)},
        resonators={"q0": ResonatorConfig(N_res=4)},
        couplings=[CouplingConfig(q1="q0", q2="q1", J=5e6)],
    )
    system = QuantumSystem(cfg)

    # Total Hilbert space dimension should be 2 * 3 * 4 = 24
    assert system.dims == [2, 3, 4]

    # Verify generalized operator shapes
    assert system.b["q0"].shape == (24, 24)
    assert system.a["q0"].shape == (24, 24)

    # Static Hamiltonian should map to global Hilbert space
    h_static = system.build_static_hamiltonian()
    assert h_static.shape == (24, 24)


@pytest.mark.unit
def test_two_level_pauli_operators():
    """Verify that standard Pauli matrices are applied when N_q = 2."""
    cfg = SimulationConfig(qubits={"q0": QubitConfig(N_q=2)}, resonators={})
    system = QuantumSystem(cfg)

    # Qutip native Pauli matrices
    expected_sx = qutip.sigmax()
    expected_sy = qutip.sigmay()
    expected_sz = qutip.sigmaz()

    assert (system.sx["q0"] - expected_sx).norm() < 1e-12
    assert (system.sy["q0"] - expected_sy).norm() < 1e-12
    assert (system.sz["q0"] - expected_sz).norm() < 1e-12


@pytest.mark.unit
def test_collapse_operators_generation():
    """Verify amplitude damping (T1), dephasing (T2), and cavity decay (kappa) generate precise scaling factors."""
    t1_time = 10e-6
    t2_time = 5e-6
    kappa_rate = 1e6

    cfg = SimulationConfig(
        qubits={"q0": QubitConfig(N_q=2, T1=t1_time, T2=t2_time)},
        resonators={"q0": ResonatorConfig(N_res=2, kappa=kappa_rate)},
    )
    system = QuantumSystem(cfg)
    c_ops = system.build_collapse_operators()

    assert len(c_ops) == 3

    # 1. Amplitude Damping (T1): C_1 = sqrt(1/T1) * b
    t1_expected_factor = np.sqrt(1.0 / t1_time)
    c_op_t1 = c_ops[0]
    np.testing.assert_allclose(
        c_op_t1.full(), (t1_expected_factor * system.b["q0"]).full(), atol=1e-10
    )

    # 2. Pure Dephasing (T2): C_2 = sqrt(2 * gamma_phi) * nq
    # gamma_phi = (1/T2) - (1/(2*T1))
    gamma_phi = (1.0 / t2_time) - (0.5 / t1_time)
    t2_expected_factor = np.sqrt(2 * gamma_phi)
    c_op_t2 = c_ops[1]
    np.testing.assert_allclose(
        c_op_t2.full(), (t2_expected_factor * system.nq["q0"]).full(), atol=1e-10
    )

    # 3. Cavity Decay (kappa): C_3 = sqrt(kappa) * a
    kappa_expected_factor = np.sqrt(kappa_rate)
    c_op_kappa = c_ops[2]
    np.testing.assert_allclose(
        c_op_kappa.full(), (kappa_expected_factor * system.a["q0"]).full(), atol=1e-10
    )
