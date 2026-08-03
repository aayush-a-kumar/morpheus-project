# Qblox-QuTiP Simulator

This codebase provides a bridge between the **Qblox-Scheduler** and **QuTiP**, allowing you to simulate the physics of a qubit under the control pulses defined in a Qblox `Schedule`.

## Features
- Translates Qblox `Schedule` objects into time-dependent Hamiltonians for QuTiP.
- Supports common pulse shapes: `SquarePulse`, `GaussianPulse`, and `DRAGPulse`.
- Incorporates decoherence effects ($T_1$ and $T_2$) via collapse operators.
- Simulates measurement (Acquisition) and returns expectation values and measurement outcomes.
- Handles rotating frame transformations and detuning.

## Installation
Ensure you have the following packages installed:
```bash
pip install qblox-scheduler qutip numpy pandas matplotlib scipy
```

## Usage
1. Define your qubit parameters (frequency, T1, T2, Rabi strength).
2. Create a Qblox `Schedule` with pulses and acquisitions.
3. Use the `QbloxQutipSimulator` to run the simulation.

```python
from qblox_sim.simulator import QbloxQutipSimulator
from qblox_scheduler import Schedule
from qblox_scheduler.operations import GaussianPulse

# 1. Setup parameters
params = {
    'f_q': 5.0e9,
    'rabi_freq_per_volt': 50e6,
    'T1': 20e-6,
    'T2': 10e-6
}

# 2. Create Schedule
sched = Schedule("MyExperiment")
sched.add(GaussianPulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))

# 3. Simulate
sim = QbloxQutipSimulator(params)
res = sim.simulate(sched)

print(f"Final state: {res['result'].states[-1]}")
if res['measurements']:
    print(f"Measurement Probability: {res['measurements'][0]['prob_1']}")
```

## Directory Structure
- `src/qblox_sim/`: Core simulator logic.
- `examples/`: Example scripts (e.g., Rabi oscillation).
- `requirements.txt`: List of dependencies.

## Physics Model
The simulator uses the following Hamiltonian in the rotating frame:
$$H = \frac{1}{2} \hbar \Delta \sigma_z + \frac{1}{2} \hbar \Omega(t) (I(t) \sigma_x + Q(t) \sigma_y)$$
where:
- $\Delta = \omega_q - \omega_d$ is the detuning.
- $\Omega(t)$ is derived from the pulse amplitude and the `rabi_freq_per_volt` parameter.
- $I(t)$ and $Q(t)$ are the in-phase and quadrature envelopes from the schedule.
