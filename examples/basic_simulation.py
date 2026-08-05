import numpy as np
import matplotlib.pyplot as plt
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import GaussPulse, SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator

def run_rabi_simulation():
    # 1. Define Qubit Parameters
    f_q = 5.0e9 # 5 GHz qubit
    qubit_params = {
        'f_q': f_q,
        'f_d': f_q, # Resonant drive
        'rabi_freq_per_volt': 50e6, # 50 MHz per Volt
        'T1': 20e-6, # 20 us
        'T2': 10e-6, # 10 us
        'N_q': 3,    # 3-level transmon (default)
    }

    # 2. Create the Schedule
    sched = Schedule("Rabi Experiment")
    sched.add_resource(ClockResource(name="q0.01", freq=f_q))
    
    # 100ns pulse with 0.1V amplitude
    sched.add(GaussPulse(amplitude=0.1, phase=0, duration=100e-9, port="q0:mw", clock="q0.01"))
    
    # 3. Setup Simulator
    sim = QbloxQutipSimulator(qubit_params)
    
    # 4. Run Simulation
    res_dict = sim.simulate(sched)
    result = res_dict['result']
    t_list = res_dict['t_list']

    # 5. Extract and Plot Results
    # FIX: Use get_expectation() on SimulationResult (dimension-agnostic for N_q=2 or N_q>=3)
    expt_x = result.get_expectation('sx')
    expt_y = result.get_expectation('sy')
    expt_z = result.get_expectation('sz')
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_list * 1e9, expt_x, label='<X>')
    plt.plot(t_list * 1e9, expt_y, label='<Y>')
    plt.plot(t_list * 1e9, expt_z, label='<Z>')
    
    plt.xlabel('Time (ns)')
    plt.ylabel('Expectation Value')
    plt.title('Rabi Pulse Simulation (Gaussian Envelope)')
    plt.legend()
    plt.grid(True)
    plt.savefig('rabi_simulation.png')
    print("Simulation completed successfully. Plot saved to rabi_simulation.png")

if __name__ == "__main__":
    run_rabi_simulation()