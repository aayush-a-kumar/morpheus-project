# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
import numpy as np
import qutip

from qblox_sim.config import QubitConfig, SimulationConfig
from qblox_sim.engine import QuTiPEngine
from qblox_sim.physics import QuantumSystem
from qblox_sim.signals import ScheduleSignalProvider


def test_flux_drive_phase_accumulation():
    """
    Tests that a flux drive applied to 'q0:fl' correctly shifts the qubit frequency
    and accumulates a relative Z-phase over time compared to an undriven qubit.
    """
    # 1. Setup system with zero detuning so natural phase evolution is 0
    # FIX: Use 25 MHz/V so a 10ns pulse creates a 90-degree phase shift (not 10 full rotations!)
    cfg = SimulationConfig(
        qubits={"q0": QubitConfig(f_q=5e9, f_d=5e9, flux_freq_per_volt=25e6)}, dt=1e-9
    )
    system = QuantumSystem(cfg)
    engine = QuTiPEngine()

    t_list = np.linspace(0, 10e-9, 11)  # 10 ns evolution

    # Create an initial superposition state |+> = (|0> + |1>) / sqrt(2)
    ground = system.get_default_initial_state()
    excited = qutip.tensor(
        qutip.basis(cfg.qubits["q0"].N_q, 1), qutip.basis(cfg.resonators["q0"].N_res, 0)
    )
    initial_state = (ground + excited).unit()

    # 2. Run WITHOUT flux
    drives_no_flux = {"q0:mw": np.zeros_like(t_list), "q0:fl": np.zeros_like(t_list)}
    res_no_flux = engine.run(system, drives_no_flux, t_list, initial_state)
    state_no_flux = res_no_flux.states[-1]

    # 3. Run WITH 1.0 Volt flux pulse
    drives_with_flux = {"q0:mw": np.zeros_like(t_list), "q0:fl": np.ones_like(t_list)}
    res_with_flux = engine.run(system, drives_with_flux, t_list, initial_state)
    state_with_flux = res_with_flux.states[-1]

    # 4. Verify the flux altered the state's phase
    fidelity = qutip.fidelity(state_no_flux, state_with_flux)

    # A 90-degree phase shift on the equator results in an exact fidelity of 1/sqrt(2) ≈ 0.7071
    expected_fidelity = 1 / np.sqrt(2)
    assert np.isclose(fidelity, expected_fidelity, atol=1e-3), (
        f"Flux drive failed to accumulate correct phase! Fidelity: {fidelity}"
    )


def test_schedule_signal_provider_phase_tracking():
    """
    Tests that Virtual-Z gates (baked into the 'phase' parameter of a pulse)
    correctly rotate the complex microwave envelope in the signal provider.
    """
    t_list = np.linspace(0, 100e-9, 100)

    # Mock two consecutive pulses. The compiler should bake the Virtual-Z into the second pulse's phase.
    pulses_list = [
        {
            "port": "q0:mw",
            "abs_time": 0.0,
            "duration": 20e-9,
            "amp": 1.0,
            "phase": 0.0,  # First pulse: 0 degrees (Pure I-quadrature)
            "wf_func": "square",
        },
        {
            "port": "q0:mw",
            "abs_time": 50e-9,
            "duration": 20e-9,
            "amp": 1.0,
            "phase": 90.0,  # Second pulse: 90 degrees (Pure Q-quadrature after Virtual Z)
            "wf_func": "square",
        },
    ]

    provider = ScheduleSignalProvider(pulses_list)
    drives = provider.get_drives(t_list)

    mw_drive = drives["q0:mw"]

    # Verify Pulse 1 (0 to 20ns) is entirely real (I)
    pulse_1_slice = mw_drive[5:15]  # Sample middle of pulse 1
    assert np.allclose(np.real(pulse_1_slice), 1.0)
    assert np.allclose(np.imag(pulse_1_slice), 0.0)

    # Verify Pulse 2 (50 to 70ns) is entirely imaginary (Q) due to the 90-degree phase shift
    pulse_2_slice = mw_drive[55:65]  # Sample middle of pulse 2
    assert np.allclose(np.real(pulse_2_slice), 0.0, atol=1e-10)
    assert np.allclose(np.imag(pulse_2_slice), 1.0)
