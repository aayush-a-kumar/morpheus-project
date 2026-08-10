import numpy as np
import qutip
from qblox_sim.config import SimulationConfig
from qblox_sim.physics import QuantumSystem


def test_quantum_system_dimensions():
    cfg = SimulationConfig.from_dict({'N_q': 3, 'N_res': 5})
    system = QuantumSystem(cfg)

    # Check state space dimensions (3 * 5 = 15)
    assert system.b.shape == (15, 15)
    assert system.a.shape == (15, 15)
    assert system.sz.shape == (15, 15)


def test_static_hamiltonian_and_c_ops():
    cfg = SimulationConfig.from_dict({
        'f_q': 5.0e9,
        'f_d': 5.0e9,
        'T1': 10e-6,
        'kappa': 1e6
    })
    system = QuantumSystem(cfg)

    h_static = system.build_static_hamiltonian()
    c_ops = system.build_collapse_operators()

    assert isinstance(h_static, qutip.Qobj)
    assert len(c_ops) == 2  # T1 and kappa collapse operators present