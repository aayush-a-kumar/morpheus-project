# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Executes the time-domain evolution of the quantum system.

This module bridges the static quantum physics definitions and the dynamic
time-series drives, mapping hardware-level microwave and flux signals into
a time-dependent Hamiltonian that QuTiP's mesolve can process.
"""

import typing
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
        chunks: list[tuple[str, int, int]] | None = None,
    ) -> qutip.Result:
        """
        Simulates system dynamics by chunking execution between active ODE solving and idle spectral leaps.

        Args:
            system (QuantumSystem): The static quantum system containing operators and configuration.
            drives (dict[str, np.ndarray]): Mapping of hardware port names to time-series drive arrays.
            t_list (np.ndarray): The strictly uniform global time grid array.
            initial_state (qutip.Qobj): The starting state vector or density matrix.
            options (dict[str, Any] | None, optional): Solver options for qutip.mesolve. Defaults to None.
            chunks (list[tuple[str, int, int]] | None, optional): List of execution blocks defining ('type', start_idx, end_idx). Defaults to None.

        Returns:
            qutip.Result: An object containing the stitched time-evolved states of the system.
        """
        # WHY: QuTiP 5 mesolve accepts a standard dictionary for options.
        solver_options = options or {"nsteps": 500000}

        # WHY: Bridge compatibility between QuTiP 4.7 and 5.x.
        # QuTiP 5 strictly expects 'store_states' to exist during Result instantiation.
        result_options = solver_options.copy()
        result_options.setdefault("store_states", True)
        result_options.setdefault("store_final_state", False)

        c_ops = system.build_collapse_operators()

        # WHY: Fallback to a single active monolithic run if the schedule contains no distinct chunks.
        if chunks is None:
            chunks = [("active", 0, len(t_list) - 1)]

        global_states = []
        current_state = initial_state

        # WHY: Retrieve pre-diagonalized static Hamiltonian data from the physics cache.
        eigenvalues = system.eigenvalues
        eigenvectors = system.eigenvectors

        for chunk_type, start_idx, end_idx in chunks:
            # WHY: Slice the time grid for the current execution block to maintain timeline alignment.
            chunk_t = t_list[start_idx : end_idx + 1]

            if chunk_type == "active":
                # WHY: Reconstruct the time-dependent Hamiltonian explicitly for this active window.
                h = self._build_active_hamiltonian(
                    system, drives, start_idx, end_idx + 1
                )

                # WHY: Execute the standard Lindblad Master Equation over the continuous drive arrays.
                res = qutip.mesolve(
                    h, current_state, chunk_t, c_ops=c_ops, options=solver_options
                )

                # WHY: Drop the first state when appending (unless it's the first chunk) to prevent duplicate boundary timestamps.
                states_to_add = res.states if start_idx == 0 else res.states[1:]
                global_states.extend(states_to_add)
                current_state = res.states[-1]

            elif chunk_type == "idle":
                # WHY: Execute O(1) analytical unitary evolution and Trotterized decay over the empty gap.
                idle_states = self._run_idle_vectorized(
                    system, current_state, chunk_t, eigenvalues, eigenvectors
                )

                states_to_add = idle_states if start_idx == 0 else idle_states[1:]
                global_states.extend(states_to_add)
                current_state = idle_states[-1]

        # WHY: Package the stitched states back into a standard QuTiP Result object.
        # We pass result_options to satisfy QuTiP 5's strict initialization checks,
        # while casting to Any to keep Pylance happy.
        final_result = qutip.Result(
            e_ops={}, options=typing.cast(typing.Any, result_options)
        )
        final_result.states = global_states

        return final_result

    def _build_active_hamiltonian(
        self,
        system: QuantumSystem,
        drives: dict[str, np.ndarray],
        start_idx: int,
        end_idx: int,
    ) -> list[Any]:
        """
        Slices the global hardware drives into a localized time-dependent Hamiltonian array for QuTiP.

        Args:
            system (QuantumSystem): The static quantum system.
            drives (dict[str, np.ndarray]): Global mapping of hardware drives.
            start_idx (int): Global index start bound.
            end_idx (int): Global index end bound.

        Returns:
            list[Any]: QuTiP-compatible time-dependent Hamiltonian list.
        """
        h_static = system.build_static_hamiltonian()
        h: list[Any] = [h_static]

        # WHY: Iterate configured qubits to map MW and Flux drives.
        for q_name, q_cfg in system.cfg.qubits.items():
            mw_port = f"{q_name}:mw"
            if mw_port in drives:
                # WHY: Slice the global drive array to strictly match the chunk's length.
                q_drive = drives[mw_port][start_idx:end_idx]
                q_i, q_q = np.real(q_drive), np.imag(q_drive)

                omega_q = 2 * np.pi * q_cfg.rabi_freq_per_volt
                b, bd = system.b[q_name], system.bd[q_name]

                if np.any(q_i):
                    h.append([(b + bd) * (omega_q / 2), q_i])
                if np.any(q_q):
                    h.append([(1j * (bd - b)) * (omega_q / 2), q_q])

            fl_port = f"{q_name}:fl"
            if fl_port in drives:
                fl_v = np.real(drives[fl_port][start_idx:end_idx])
                nq = system.nq[q_name]

                if hasattr(q_cfg, "v_phi0") and q_cfg.v_phi0 is not None:
                    f_max = q_cfg.f_max if q_cfg.f_max is not None else q_cfg.f_q
                    freq_shift = f_max * (
                        np.sqrt(np.abs(np.cos(np.pi * fl_v / q_cfg.v_phi0))) - 1.0
                    )
                    flux_coupling_array = 2 * np.pi * freq_shift
                else:
                    flux_coupling_array = fl_v * (2 * np.pi * q_cfg.flux_freq_per_volt)

                if np.any(fl_v):
                    h.append([-nq, flux_coupling_array])

        # WHY: Map Readout drives to configured resonators.
        for r_name, r_cfg in system.cfg.resonators.items():
            res_port = f"{r_name}:res"
            if res_port in drives:
                res_drive = drives[res_port][start_idx:end_idx]
                res_i, res_q = np.real(res_drive), np.imag(res_drive)

                omega_res = 2 * np.pi * r_cfg.rabi_freq_res_per_volt
                a, ad = system.a[r_name], system.ad[r_name]

                if np.any(res_i):
                    h.append([(a + ad) * (omega_res / 2), res_i])
                if np.any(res_q):
                    h.append([(1j * (ad - a)) * (omega_res / 2), res_q])

        return h

    def _run_idle_vectorized(
        self,
        system: QuantumSystem,
        initial_state: qutip.Qobj,
        t_slice: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> list[qutip.Qobj]:
        """
        Executes analytical unitary evolution and Lie-Trotter decay over an idle time block.

        Args:
            system (QuantumSystem): The static quantum system containing cached spectral data.
            initial_state (qutip.Qobj): The state at the beginning of the idle chunk.
            t_slice (np.ndarray): The segmented time array for the idle chunk.
            eigenvalues (np.ndarray): 1D array of static Hamiltonian eigenvalues.
            eigenvectors (np.ndarray): 2D array of static Hamiltonian eigenvectors.

        Returns:
            list[qutip.Qobj]: The analytically computed state history matching the time slice.
        """
        # WHY: Open-system decay (T1/T2) requires density matrices.
        rho = (
            initial_state
            if initial_state.type == "oper"
            else qutip.ket2dm(initial_state)
        )
        rho_dense = rho.full()

        # WHY: Transform the initial density matrix into the eigenbasis using cached v_dag.
        rho_eigen = system.v_dag @ rho_dense @ eigenvectors

        dt_array = t_slice - t_slice[0]
        states = []

        for dt in dt_array:
            # 1. Unitary Spectral Leap
            # WHY: Evaluate unitary phase accumulation using the cached delta_e energy gaps.
            phase_matrix = np.exp(-1j * system.delta_e * dt)

            # 2. Lie-Trotter Decay Mapping
            # WHY: Apply the cached global off-diagonal decay rate matrix scaling factor.
            decay_matrix = np.exp(-system.decay_rate_matrix * dt)

            # WHY: Element-wise multiply the eigenstate density matrix by both the phase
            # and decay matrices simultaneously. This applies phenomenological damping without nested loops.
            rho_t_eigen = rho_eigen * phase_matrix * decay_matrix

            # 3. Transform back to the laboratory basis
            rho_lab = eigenvectors @ rho_t_eigen @ system.v_dag

            # WHY: Repackage into QuTiP Qobj to maintain pipeline compatibility.
            qobj_state = qutip.Qobj(rho_lab, dims=rho.dims)
            states.append(qobj_state)

        return states
