# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
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

        h_static = system.build_static_hamiltonian()
        c_ops = system.build_collapse_operators()

        # PYLANCE FIX: Explicitly type 'h' as List[Any] to allow QuTiP's mixed-type Hamiltonian format
        h: list[Any] = [h_static]

        # 1. Map MW drives to all configured qubits
        for q_name, q_cfg in system.cfg.qubits.items():
            port_name = f"{q_name}:mw"
            if port_name in drives:
                q_drive = drives[port_name]
                q_i, q_q = np.real(q_drive), np.imag(q_drive)

                omega_q = 2 * np.pi * q_cfg.rabi_freq_per_volt
                b = system.b[q_name]
                bd = system.bd[q_name]

                if np.any(q_i):
                    h.append([(b + bd) * (omega_q / 2), q_i])  # type: ignore[operator]
                if np.any(q_q):
                    h.append([(1j * (bd - b)) * (omega_q / 2), q_q])  # type: ignore[operator]

            # --- Flux (Z) Drive ---
            fl_port = f"{q_name}:fl"
            if fl_port in drives:
                fl_drive = drives[fl_port]
                fl_v = np.real(fl_drive)  # Flux pulses are baseband (real voltage)

                nq = system.nq[q_name]

                # Determine frequency shift array based on available config
                if hasattr(q_cfg, "v_phi0") and q_cfg.v_phi0 is not None:
                    # Non-linear Transmon Tuning Arc
                    f_max = q_cfg.f_max if q_cfg.f_max is not None else q_cfg.f_q

                    # Map voltage array to frequency shift array
                    freq_shift = f_max * (
                        np.sqrt(np.abs(np.cos(np.pi * fl_v / q_cfg.v_phi0))) - 1.0
                    )

                    # Convert to angular frequency detuning
                    flux_coupling_array = 2 * np.pi * freq_shift
                else:
                    # Fallback to linear shift (for test_flux_and_phase.py)
                    flux_coupling = 2 * np.pi * q_cfg.flux_freq_per_volt
                    flux_coupling_array = fl_v * flux_coupling

                if np.any(fl_v):
                    h.append([-nq, flux_coupling_array])  # type: ignore[operator]

        # 2. Map Readout drives to all configured resonators
        for r_name, r_cfg in system.cfg.resonators.items():
            port_name = f"{r_name}:res"
            if port_name in drives:
                res_drive = drives[port_name]
                res_i, res_q = np.real(res_drive), np.imag(res_drive)

                omega_res = 2 * np.pi * r_cfg.rabi_freq_res_per_volt
                a = system.a[r_name]
                ad = system.ad[r_name]

                if np.any(res_i):
                    h.append([(a + ad) * (omega_res / 2), res_i])  # type: ignore[operator]
                if np.any(res_q):
                    h.append([(1j * (ad - a)) * (omega_res / 2), res_q])  # type: ignore[operator]

        solver_options = options or {"nsteps": 500000}
        return qutip.mesolve(
            h, initial_state, t_list, c_ops=c_ops, options=solver_options
        )
