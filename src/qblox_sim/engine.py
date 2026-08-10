import qutip
import numpy as np
from typing import Dict, Optional, Any
from qblox_sim.physics import QuantumSystem

class QuTiPEngine:
    """Executes the time-evolution of the quantum system using QuTiP."""
    
    def run(
        self, 
        system: QuantumSystem, 
        drives: Dict[str, np.ndarray], 
        t_list: np.ndarray,
        initial_state: qutip.Qobj,
        options: Optional[Dict[str, Any]] = None
    ) -> qutip.Result:
        
        omega_q = 2 * np.pi * system.cfg.qubit.rabi_freq_per_volt
        omega_res = 2 * np.pi * system.cfg.resonator.rabi_freq_res_per_volt
        
        q_i, q_q = np.real(drives.get("q_drive", 0.0)), np.imag(drives.get("q_drive", 0.0))
        res_i, res_q = np.real(drives.get("res_drive", 0.0)), np.imag(drives.get("res_drive", 0.0))

        h_static = system.build_static_hamiltonian()
        c_ops = system.build_collapse_operators()

        h = [
            h_static,
            [(system.b + system.bd) * (omega_q / 2), q_i],
            [1j * (system.bd - system.b) * (omega_q / 2), q_q],
            [(system.a + system.ad) * (omega_res / 2), res_i],
            [1j * (system.ad - system.a) * (omega_res / 2), res_q]
        ]
        
        solver_options = options or {"nsteps": 500000}
        return qutip.mesolve(h, initial_state, t_list, c_ops=c_ops, options=solver_options)