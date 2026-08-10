import numpy as np
import pytest
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.config import SimulationConfig, QubitConfig, ResonatorConfig, CouplingConfig
from qblox_sim.physics import QuantumSystem
from qblox_sim.simulator import QbloxQutipSimulator


def test_quantum_system_dimensions():
    cfg = SimulationConfig.from_dict({
        "qubits": {"q0": {"N_q": 3}},
        "resonators": {"q0": {"N_res": 5}}
    })
    system = QuantumSystem(cfg)

    assert system.b["q0"].shape == (15, 15)
    assert system.a["q0"].shape == (15, 15)
    assert system.sz["q0"].shape == (15, 15)


def test_static_hamiltonian_and_c_ops():
    cfg = SimulationConfig.from_dict({
        "qubits": {"q0": {"f_q": 5.0e9, "f_d": 5.0e9, "T1": 10e-6}},
        "resonators": {"q0": {"kappa": 1e6}}
    })
    system = QuantumSystem(cfg)

    h_static = system.build_static_hamiltonian()
    c_ops = system.build_collapse_operators()

    assert isinstance(h_static, qutip.Qobj)
    assert len(c_ops) == 2


def test_multi_qubit_topology_physics():
    cfg = SimulationConfig(
        qubits={"q0": QubitConfig(N_q=2), "q1": QubitConfig(N_q=2)},
        resonators={"q0": ResonatorConfig(N_res=3)},
        couplings=[CouplingConfig(q1="q0", q2="q1", J=5e6)]
    )
    system = QuantumSystem(cfg)

    assert system.dims == [2, 2, 3]
    assert system.b["q0"].shape == (12, 12)
    assert system.b["q1"].shape == (12, 12)
    assert system.a["q0"].shape == (12, 12)

    h_static = system.build_static_hamiltonian()
    assert h_static.shape == (12, 12)


def test_t1_relaxation(default_qubit_params):
    """Verify state |1> decays back to |0> over T1 time according to theoretical expectation."""
    sim = QbloxQutipSimulator(default_qubit_params)

    t1 = sim.cfg.qubit.T1
    decay_time = 10e-6

    state_1 = qutip.tensor(
        qutip.basis(sim.cfg.qubit.N_q, 1), 
        qutip.basis(sim.cfg.resonator.N_res, 0)
    )

    sched = Schedule("T1 Decay")
    sched.add_resource(ClockResource(name="q0.01", freq=sim.cfg.qubit.f_q))
    sched.add(SquarePulse(amp=0.0, duration=decay_time, port="q0:mw", clock="q0.01"))

    res = sim.simulate(sched, initial_state=state_1)
    result = res["result"]

    final_z = qutip.expect(sim.system.sz['q0'], result.states[-1]).real
    expected_z = 1.0 - 2.0 * np.exp(-decay_time / t1)

    assert np.isclose(final_z, expected_z, atol=1e-3)


def test_rabi_pulse_inversion(default_qubit_params):
    """Verify a Pi-pulse (0.5V, 100ns at 10 MHz/V) rotates <Z> from ground state +1.0 to excited state near -1.0."""
    sim = QbloxQutipSimulator(default_qubit_params)

    sched = Schedule("Pi Pulse")
    sched.add_resource(ClockResource(name="q0.01", freq=sim.cfg.qubit.f_q))
    sched.add(SquarePulse(amp=0.5, duration=100e-9, port="q0:mw", clock="q0.01"))

    res = sim.simulate(sched)
    result = res["result"]

    final_z = qutip.expect(sim.system.sz['q0'], result.states[-1]).real
    assert final_z < -0.95