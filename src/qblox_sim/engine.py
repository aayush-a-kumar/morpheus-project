# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Executes the time-domain evolution of the quantum system.

This module bridges the static quantum physics definitions and the dynamic
time-series drives, mapping hardware-level microwave and flux signals into
a time-dependent Hamiltonian that QuTiP's mesolve can process.
"""

from typing import Any

import numpy as np
import qutip

from qblox_sim.physics import QuantumSystem


class QuTiPEngine:
    """Executes the generalized multi-qubit time-evolution."""

    def run(
        self,
        system: QuantumSystem,
        drives: dict[str, np.ndarray],
        t_list: np.ndarray,
        initial_state: qutip.Qobj,
        options: dict[str, Any] | None = None,
    ) -> qutip.Result:
        """
        Simulates the system dynamics over a specified time grid using QuTiP.

        Args:
            system (QuantumSystem): The static quantum system containing operators
                and configuration parameters.
            drives (dict[str, np.ndarray]): A mapping of hardware port names
                (e.g., 'q0:mw', 'q0:fl') to their respective time-series drive arrays.
            t_list (np.ndarray): The time grid over which the simulation is evaluated.
            initial_state (qutip.Qobj): The starting state vector or density matrix
                of the global system.
            options (dict[str, Any] | None, optional): Additional solver options
                passed directly to qutip.mesolve. Defaults to None.

        Returns:
            qutip.Result: An object containing the time-evolved expectation values
                and states of the system.
        """

        h_static = system.build_static_hamiltonian()
        c_ops = system.build_collapse_operators()

        # PYLANCE FIX: Explicitly type 'h' as List[Any] to allow QuTiP's mixed-type Hamiltonian format
        # WHY: QuTiP expects time-dependent Hamiltonians as a list where the first element
        # is the static H0, and subsequent elements are lists of [operator, time_series_array].
        h: list[Any] = [h_static]

        # 1. Map MW drives to all configured qubits
        for q_name, q_cfg in system.cfg.qubits.items():
            port_name = f"{q_name}:mw"
            if port_name in drives:
                q_drive = drives[port_name]
                # WHY: Microwave pulses are defined as complex baseband signals.
                # The Real (I) component drives rotations around the X-axis,
                # and the Imaginary (Q) component drives rotations around the Y-axis.
                q_i, q_q = np.real(q_drive), np.imag(q_drive)

                omega_q = 2 * np.pi * q_cfg.rabi_freq_per_volt
                b = system.b[q_name]
                bd = system.bd[q_name]

                # WHY: (b + bd) acts as the generalized Pauli-X operator for the transmon.
                if np.any(q_i):
                    h.append([(b + bd) * (omega_q / 2), q_i])  # type: ignore[operator]
                # WHY: 1j * (bd - b) acts as the generalized Pauli-Y operator.
                if np.any(q_q):
                    h.append([(1j * (bd - b)) * (omega_q / 2), q_q])  # type: ignore[operator]

            # --- Flux (Z) Drive ---
            fl_port = f"{q_name}:fl"
            if fl_port in drives:
                fl_drive = drives[fl_port]
                # WHY: Flux pulses modulate the SQUID loop magnetic flux.
                # Qblox baseband outputs are real-valued voltages, so we discard any imaginary components.
                fl_v = np.real(fl_drive)

                nq = system.nq[q_name]

                # Determine frequency shift array based on available config
                if hasattr(q_cfg, "v_phi0") and q_cfg.v_phi0 is not None:
                    # Non-linear Transmon Tuning Arc
                    f_max = q_cfg.f_max if q_cfg.f_max is not None else q_cfg.f_q

                    # WHY: Transmons behave as tunable non-linear oscillators.
                    # The frequency follows an absolute cosine arc relative to the applied voltage,
                    # where v_phi0 is the voltage required to thread one flux quantum (Φ0).
                    freq_shift = f_max * (
                        np.sqrt(np.abs(np.cos(np.pi * fl_v / q_cfg.v_phi0))) - 1.0
                    )

                    # Convert the linear frequency shift into angular frequency detuning for the Hamiltonian.
                    flux_coupling_array = 2 * np.pi * freq_shift
                else:
                    # Fallback to linear shift (for test_flux_and_phase.py)
                    # WHY: Linear fallback allows testing simple Z-gates without needing full SQUID parameters.
                    flux_coupling = 2 * np.pi * q_cfg.flux_freq_per_volt
                    flux_coupling_array = fl_v * flux_coupling

                if np.any(fl_v):
                    # WHY: Flux shifts change the energy of the |1> state relative to |0>.
                    # We map this to the -nq operator to properly detune the transmon frequency.
                    h.append([-nq, flux_coupling_array])  # type: ignore[operator]

        # 2. Map Readout drives to all configured resonators
        for r_name, r_cfg in system.cfg.resonators.items():
            port_name = f"{r_name}:res"
            if port_name in drives:
                res_drive = drives[port_name]
                # WHY: Similar to qubit MW drives, readout pulses use complex IQ envelopes
                # to drive the readout cavity's X and Y quadratures.
                res_i, res_q = np.real(res_drive), np.imag(res_drive)

                omega_res = 2 * np.pi * r_cfg.rabi_freq_res_per_volt
                a = system.a[r_name]
                ad = system.ad[r_name]

                # WHY: Drives the (a + ad) cavity displacement field.
                if np.any(res_i):
                    h.append([(a + ad) * (omega_res / 2), res_i])  # type: ignore[operator]
                # WHY: Drives the orthogonal cavity momentum quadrature.
                if np.any(res_q):
                    h.append([(1j * (ad - a)) * (omega_res / 2), res_q])  # type: ignore[operator]

        solver_options = options or {"nsteps": 500000}

        # WHY: Execute the Lindblad Master Equation.
        return qutip.mesolve(
            h, initial_state, t_list, c_ops=c_ops, options=solver_options
        )
