# examples/profile_simulator.py
import numpy as np
from line_profiler import LineProfiler

from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse, Measure
from qblox_scheduler.resources import ClockResource

from qblox_sim.simulator import QbloxQutipSimulator
from qblox_sim.engine import QuTiPEngine
from qblox_sim.signals import ScheduleSignalProvider

def build_benchmark_schedule():
    """Builds a moderately complex schedule to stress the simulator."""
    sched = Schedule("Profiler Benchmark")
    sched.add_resource(ClockResource(name="q0.01", freq=5.0e9))
    sched.add_resource(ClockResource(name="q0.ro", freq=6.0e9))
    
    # Add multiple pulses and gaps to trigger active/idle chunking
    for i in range(5):
        sched.add(SquarePulse(amp=0.2, duration=50e-9, port="q0:mw", clock="q0.01"))
        # Implicit gap of 50ns before the next pulse
        
    sched.add(Measure("q0", clock="q0.ro"))
    return sched

def run_profiler():
    # 1. Setup the baseline simulator configuration with a valid 'q0' topology
    params = {
        "qubits": {"q0": {}},
        "resonators": {"q0": {}},
        "dt": 1.0e-9
    }
    sim = QbloxQutipSimulator(params)
    sched = build_benchmark_schedule()

    # 2. Initialize the Line Profiler
    profiler = LineProfiler()

    # 3. Target the suspected bottlenecks by adding them to the profiler
    profiler.add_function(sim._simulate_processed)
    profiler.add_function(QuTiPEngine.run)
    profiler.add_function(QuTiPEngine._build_active_hamiltonian)
    profiler.add_function(ScheduleSignalProvider.get_drives)

    # 4. Wrap the main entry point and execute it
    profiled_simulate = profiler(sim.simulate)
    
    print("Running simulation with line_profiler attached...")
    profiled_simulate(sched)
    
    # 5. Output the results
    print("\n--- PROFILING RESULTS ---")
    profiler.print_stats()

if __name__ == "__main__":
    run_profiler()