import numpy as np
import matplotlib.pyplot as plt
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse, IdlePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator

def validate_t1_decay():
    # 1. Define Qubit Parameters
    f_q = 5.0e9
    rabi_freq = 50e6 # 50 MHz/V
    t1_time = 5e-6   # 5 us Relaxation time
    t2_time = 10e-6  # No extra dephasing beyond T1 limit (T2 = 2*T1)
    
    params = {
        'f_q': f_q,
        'f_d': f_q,
        'rabi_freq_per_volt': rabi_freq,
        'T1': t1_time,
        'T2': t2_time,
        'N_res': 2 # Keep resonator small for speed
    }

    # 2. Calculate Pi-pulse (to reach |1>)
    # For a square pulse: Area = amp * duration * rabi_freq = 0.5 (for Pi pulse in cycles)
    # Let's use 0.1V amp -> duration = 0.5 / (0.1 * 50e6) = 100ns
    amp = 0.1
    duration = 100e-9

    # 3. Create Schedule
    sched = Schedule("T1 Decay Validation")
    sched.add_resource(ClockResource(name="q0.01", freq=f_q))
    
    # Pi pulse
    p1 = sched.add(SquarePulse(amplitude=amp, duration=duration, port="q0:mw", clock="q0.01"))
    
    # Long idle (4 * T1)
    sched.add(IdlePulse(duration=4 * t1_time), ref_op=p1, ref_pt="end")

    # 4. Simulate
    sim = QbloxQutipSimulator(params)
    print(f"Simulating decay with T1 = {t1_time*1e6:.1f} us...")
    res_dict = sim.simulate(sched)
    result = res_dict['result']
    t_list = res_dict['t_list']

    # 5. Extract Z expectation value
    # <Z> = 1 for |0>, <Z> = -1 for |1>
    expt_z = []
    for s in result.states:
        rho_q = s.ptrace(0) if s.type == 'oper' else qutip.ket2dm(s).ptrace(0)
        expt_z.append(qutip.expect(qutip.sigmaz(), rho_q).real)

    # 6. Theoretical Prediction
    # After the pi pulse (at t = 100ns), Z should decay as:
    # Z(t) = 1 - 2 * exp(-(t - t_pulse)/T1)
    t_decay = t_list[t_list >= duration]
    z_theory = 1 - 2 * np.exp(-(t_decay - duration) / t1_time)

    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_list * 1e6, expt_z, label='Simulated <Z>')
    plt.plot(t_decay * 1e6, z_theory, 'k--', alpha=0.7, label='Theoretical T1 Decay')
    
    plt.axvline(x=duration*1e6, color='gray', linestyle=':', label='Pi-pulse End')
    plt.xlabel('Time (us)')
    plt.ylabel('Expectation Value <Z>')
    plt.title(f'Validation of Qubit Decay (T1 = {t1_time*1e6:.1f} us)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('t1_decay_validation.png')
    print("Validation completed. Plot saved to t1_decay_validation.png")

if __name__ == "__main__":
    validate_t1_decay()
