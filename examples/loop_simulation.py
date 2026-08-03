import numpy as np
import matplotlib.pyplot as plt
import qutip
from qblox_scheduler import Schedule, SerialCompiler, QuantumDevice, BasicTransmonElement
from qblox_scheduler.operations import SquarePulse, linspace, DType
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxLoopSimulator

def run_loop_simulation():
    # 1. Setup device
    device = QuantumDevice(name="device")
    q0 = BasicTransmonElement("q0")
    q0.clock_freqs.f01 = 5.0e9
    device.add_element(q0)
    
    params = {
        'f_q': 5.0e9,
        'f_d': 5.0e9,
        'rabi_freq_per_volt': 100e6, # 100 MHz per Volt
        'T1': 20e-6,
        'T2': 10e-6,
    }

    # 2. Create Schedule with a loop
    sched = Schedule("Rabi Amplitude Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=5.0e9))
    
    # Sweep amplitude from 0.0 to 0.5 in 5 steps
    amp_domain = linspace(0.0, 0.5, 5, dtype=DType.AMPLITUDE)
    
    with sched.loop(amp_domain) as amp:
        # 50ns pulse
        sched.add(SquarePulse(amp=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
        # 50ns wait
        sched.add(SquarePulse(amp=0, duration=50e-9, port="q0:mw", clock="q0.01"))

    # 3. Setup Loop Simulator
    sim = QbloxLoopSimulator(params)
    
    # 4. Run Simulation
    print("Running looped simulation...")
    res_dict = sim.simulate(sched)
    result = res_dict['result']
    t_list = res_dict['t_list']

    # 5. Plot Results
    expt_z = []
    for s in result.states:
        rho_q = s.ptrace(0) if s.type == 'oper' else qutip.ket2dm(s).ptrace(0)
        expt_z.append(qutip.expect(qutip.sigmaz(), rho_q).real)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_list * 1e9, expt_z, label='<Z>')
    
    # Add vertical lines to show loop boundaries
    for i in range(6):
        plt.axvline(x=i*100, color='gray', linestyle='--', alpha=0.5)
        
    plt.xlabel('Time (ns)')
    plt.ylabel('Expectation Value <Z>')
    plt.title('Rabi Amplitude Sweep Simulation')
    plt.legend()
    plt.grid(True)
    plt.savefig('loop_simulation.png')
    print("Simulation completed successfully. Plot saved to loop_simulation.png")

if __name__ == "__main__":
    run_loop_simulation()
