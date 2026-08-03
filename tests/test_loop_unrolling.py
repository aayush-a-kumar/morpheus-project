import pytest
import qutip
from qblox_scheduler import Schedule, linspace, DType
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxLoopSimulator

def test_loop_amplitude_sweep(default_qubit_params):
    """Verify loop simulator dynamically updates pulse amplitudes per iteration."""
    sim = QbloxLoopSimulator(default_qubit_params)
    
    sched = Schedule("Amplitude Sweep Loop")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    
    # Sweep amplitude from 0.0 to 0.2 in 3 steps
    amp_domain = linspace(0.0, 0.2, 3, dtype=DType.AMPLITUDE)
    
    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amplitude=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
    
    res = sim.simulate(sched)
    t_list = res['t_list']
    result = res['result']
    
    # Sample <Z> state at the end of the final loop step
    final_state = result.states[-1]
    rho_q = final_state.ptrace(0) if final_state.type == 'oper' else qutip.ket2dm(final_state).ptrace(0)
    final_z = qutip.expect(qutip.sigmaz(), rho_q).real
    
    # If loop amplitude substitution worked, non-zero drive should have rotated <Z> away from 1.0
    assert final_z < 0.9, f"Loop failed to sweep amplitude dynamically: final <Z> = {final_z:.3f}"
