import numpy as np
import pytest
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator


def test_t1_relaxation(default_qubit_params):
    """Verify state |1> decays back to |0> over T1 time according to theoretical expectation."""
    sim = QbloxQutipSimulator(default_qubit_params)

    t1 = default_qubit_params.get("T1", 5e-6)
    decay_time = 10e-6  # 2 * T1

    state_1 = qutip.tensor(
        qutip.basis(sim.N_q, 1), qutip.basis(default_qubit_params["N_res"], 0)
    )

    sched = Schedule("T1 Decay")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params["f_q"]))
    sched.add(SquarePulse(amp=0.0, duration=decay_time, port="q0:mw", clock="q0.01"))

    res = sim.simulate(sched, initial_state=state_1)
    result = res["result"]

    final_z = qutip.expect(sim.sz, result.states[-1]).real

    # Theoretical <Z> for T1 decay: <Z>(t) = 1 - 2 * exp(-t / T1)
    expected_z = 1.0 - 2.0 * np.exp(-decay_time / t1)

    assert np.isclose(final_z, expected_z, atol=1e-3), (
        f"T1 decay inaccurate: expected <Z> ≈ {expected_z:.4f}, got {final_z:.4f}"
    )


def test_rabi_pulse_inversion(default_qubit_params):
    """Verify a Pi-pulse rotates <Z> from ground state +1.0 to excited state near -1.0 (with T1 decay)."""
    sim = QbloxQutipSimulator(default_qubit_params)

    sched = Schedule("Pi Pulse")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params["f_q"]))
    sched.add(SquarePulse(amp=0.1, duration=100e-9, port="q0:mw", clock="q0.01"))

    res = sim.simulate(sched)
    result = res["result"]

    final_z = qutip.expect(sim.sz, result.states[-1]).real
    
    # Account for ~1.5% T1 decay during the 100ns pulse window
    assert final_z < -0.95, f"Expected Pi-pulse to invert qubit near excited state |1>, got <Z> = {final_z}"