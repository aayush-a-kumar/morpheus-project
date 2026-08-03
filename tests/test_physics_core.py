import pytest
import numpy as np
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator

def test_t1_relaxation(default_qubit_params):
    """Verify state |1> decays back to |0> over T1 time."""
    sim = QbloxQutipSimulator(default_qubit_params)
    
    # Initialize directly in state |1, 0> (excited qubit, 0 cavity photons)
    state_1 = qutip.tensor(qutip.basis(2, 1), qutip.basis(default_qubit_params['N_res'], 0))
    
    # Empty schedule to let state decay freely for 10 us (2 * T1)
    sched = Schedule("T1 Decay")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    sched.add(SquarePulse(amp=0.0, duration=10e-6, port="q0:mw", clock="q0.01"))
    
    res = sim.simulate(sched, initial_state=state_1)
    final_state = res['result'].states[-1]
    rho_q = final_state.ptrace(0) if final_state.type == 'oper' else qutip.ket2dm(final_state).ptrace(0)
    
    # In QuTiP convention, ground state |0> has <Z> = +1.0
    final_z = qutip.expect(qutip.sigmaz(), rho_q).real
    assert final_z > 0.5, f"T1 decay failed: state did not relax to |0> (<Z> = {final_z:.3f})"

def test_rabi_pulse_inversion(default_qubit_params):
    """Verify a Pi-pulse rotates <Z> from ground state +1.0 to excited state -1.0."""
    sim = QbloxQutipSimulator(default_qubit_params)
    
    # 0.1V pulse for 100ns with 50MHz/V drive strength = Pi rotation
    sched = Schedule("Pi Pulse")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    sched.add(SquarePulse(amp=0.1, duration=100e-9, port="q0:mw", clock="q0.01"))
    
    res = sim.simulate(sched)
    final_state = res['result'].states[-1]
    rho_q = final_state.ptrace(0) if final_state.type == 'oper' else qutip.ket2dm(final_state).ptrace(0)
    
    final_z = qutip.expect(qutip.sigmaz(), rho_q).real
    assert final_z < -0.8, f"Pi pulse failed to invert state: <Z> = {final_z:.3f}"

def test_resonator_photon_population(default_qubit_params):
    """Verify driving the readout port populates photons in the cavity (<n> > 0)."""
    sim = QbloxQutipSimulator(default_qubit_params)
    
    sched = Schedule("Readout Cavity Drive")
    sched.add_resource(ClockResource(name="q0.ro", freq=default_qubit_params['f_res']))
    sched.add(SquarePulse(amp=0.2, duration=300e-9, port="q0:res", clock="q0.ro"))
    
    res = sim.simulate(sched)
    final_state = res['result'].states[-1]
    rho_res = final_state.ptrace(1) if final_state.type == 'oper' else qutip.ket2dm(final_state).ptrace(1)
    
    n_avg = qutip.expect(qutip.num(default_qubit_params['N_res']), rho_res).real
    assert n_avg > 0.1, f"Readout drive failed to populate cavity photons: <n> = {n_avg:.4f}"
