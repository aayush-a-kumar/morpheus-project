import numpy as np
import matplotlib.pyplot as plt
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import GaussPulse, SquarePulse, Measure
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator

def run_readout_simulation():
    # 1. Define Qubit and Resonator Parameters
    params = {
        'f_q': 5.0e9,
        'f_res': 6.0e9,
        'chi': 2.0e6,           # 2 MHz dispersive shift
        'rabi_freq_per_volt': 50e6,
        'rabi_freq_res_per_volt': 20e6, # Drive strength for resonator
        'T1': 20e-6,
        'T2': 10e-6,
        'kappa': 2e6,           # 2 MHz resonator decay
        'N_res': 10             # 10 levels for resonator
    }

    # 2. Create the Schedule
    sched = Schedule("Readout Experiment")
    sched.add_resource(ClockResource(name="q0.01", freq=params['f_q']))
    sched.add_resource(ClockResource(name="q0.ro", freq=params['f_res']))
    
    # Apply a Pi-pulse to the qubit to put it in |1>
    # Pi pulse duration for 0.1V: 1 / (2 * rabi_freq_per_volt * 0.1) = 1e-7 = 100ns
    # We'll use a SquarePulse for simplicity in calculating the area
    p1 = sched.add(SquarePulse(amp=0.1, duration=100e-9, port="q0:mw", clock="q0.01"))
    
    # Apply a measurement pulse to the resonator (port q0:res)
    # The Measure operation itself usually defines the acquisition, 
    # but we need to manually add the pulse to the resonator port to see the physics.
    sched.add(SquarePulse(amp=0.2, duration=500e-9, port="q0:res", clock="q0.ro"), 
              rel_time=10e-9, ref_op=p1, ref_pt="end")
    
    # Add the acquisition
    sched.add(Measure("q0", acq_index=0), rel_time=10e-9, ref_op=p1, ref_pt="end")
    
    # 3. Setup Simulator
    sim = QbloxQutipSimulator(params)
    
    # Run simulation for |1> state (with Pi pulse)
    print("Simulating |1> state...")
    res_dict_1 = sim.simulate(sched)
    
    # Run simulation for |0> state (no Pi pulse)
    print("Simulating |0> state...")
    sched_0 = Schedule("Ground State")
    sched_0.add_resource(ClockResource(name="q0.01", freq=params['f_q']))
    sched_0.add_resource(ClockResource(name="q0.ro", freq=params['f_res']))
    # No pi pulse, just measurement
    sched_0.add(SquarePulse(amp=0.2, duration=500e-9, port="q0:res", clock="q0.ro"))
    sched_0.add(Measure("q0", acq_index=0))
    res_dict_0 = sim.simulate(sched_0)

    # 5. Compare Results
    if res_dict_1['measurements'] and res_dict_0['measurements']:
        p1 = res_dict_1['measurements'][0]['prob_1']
        p0 = res_dict_0['measurements'][0]['prob_1']
        print(f"Prob(|1>) when prepared in |1>: {p1:.4f}")
        print(f"Prob(|1>) when prepared in |0>: {p0:.4f}")
        print(f"Contrast: {p1 - p0:.4f}")

if __name__ == "__main__":
    run_readout_simulation()
