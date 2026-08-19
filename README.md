# Qblox-QuTiP Simulator

This codebase provides a bridge between the **Qblox-Scheduler** and **QuTiP**, allowing you to simulate the dynamics of an N-qubit topology coupled to readout resonators under the control pulses defined in a Qblox `Schedule`.

## Features

- Translates Qblox `Schedule` objects into time-dependent Hamiltonians for QuTiP.
- Supports common pulse shapes: square, Gaussian, and DRAG pulses.
- Handles multi-qubit and multi-resonator systems with static exchange coupling and dispersive shifts.
- Incorporates decoherence effects (qubit relaxation, qubit dephasing, and resonator photon loss) via Lindblad collapse operators.
- Simulates measurements, supporting time-of-flight traces, single-sideband integration, and state discrimination thresholding.

## Installation

The project relies on a `pyproject.toml` build system. Ensure you have the required dependencies installed:

```bash
pip install qblox-instruments qblox-scheduler q1simulator qutip numpy pandas matplotlib scipy
```

## Usage

With the new architecture, parameters are organized into nested dictionaries for qubits, resonators, and acquisitions.

```python
from qblox_sim.simulator import QbloxQutipSimulator
from qblox_scheduler import Schedule
from qblox_scheduler.operations import GaussianPulse

# 1. Setup multi-qubit/resonator parameters
params = {
    "qubits": {
        "q0": {"f_q": 5.0e9, "rabi_freq_per_volt": 50e6, "T1": 20e-6, "T2": 10e-6}
    },
    "resonators": {"q0": {"f_res": 6.0e9, "kappa": 1.0e6}},
    "dt": 1.0e-9,
}

# 2. Create Schedule
sched = Schedule("MyExperiment")
sched.add(GaussianPulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))

# 3. Simulate
sim = QbloxQutipSimulator(params)
res = sim.simulate(sched)

# 4. Access Results
print(f"Final state: {res['result'].states[-1]}")
if res["measurements"]:
    print(f"Measurement Probability (Q0): {res['measurements'][0]['prob_1']}")
```

## Directory Structure

The simulator core is organized modularly under `src/qblox_sim/`:

- `config.py`: Dataclasses defining the quantum system configuration (qubits, resonators, couplings).
- `physics.py`: Dynamically builds the multi-qubit Hilbert space and static Hamiltonians.
- `engine.py`: Executes the time-evolution using QuTiP's `mesolve`.
- `signals.py`: Parses Qblox timing tables into vectorized time-series drive signals.
- `acquisitions.py`: Handles measurement strategies like integration, traces, and thresholding.
- `simulator.py`: The main user interfaces (`QbloxQutipSimulator` and `QbloxQ1Simulator`).

## Physics Model

The simulator builds a generalized multi-qubit un-driven static Hamiltonian in the rotating frame. The model incorporates qubit detuning, anharmonicity, resonator detuning, dispersive shifts, and exchange coupling:

$$H\_{static} = \\sum\_{q} (-\\Delta_q n_q + \\pi \\alpha_q (b_q^\\dagger b_q^\\dagger b_q b_q)) + \\sum\_{r} \\Delta_r n_r + \\sum\_{q,r} 2 \\pi \\chi\_{q,r} n_r n_q + \\sum\_{\\langle i, j \\rangle} 2 \\pi J\_{i,j} (b_i^\\dagger b_j + b_i b_j^\\dagger)$$

Time-dependent drives are then mapped to specific components based on hardware ports:

- **Microwave Drives (`qX:mw`)**: Applied as $\\frac{1}{2}\\Omega(t)(b + b^\\dagger)$ for in-phase and $\\frac{1}{2}i\\Omega(t)(b^\\dagger - b)$ for quadrature signals.
- **Flux Drives (`qX:fl`)**: Shift the qubit frequency dynamically, mapping voltage envelopes either linearly or via a non-linear transmon tuning arc to $-n_q \\Delta(t)$.
- **Resonator Drives (`qX:res`)**: Applied to the resonator creation/annihilation operators.
