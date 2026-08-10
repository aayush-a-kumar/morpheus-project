import qutip
import numpy as np
from typing import Dict, Optional, Any, List
from qblox_sim.physics import QuantumSystem

class QuTiPEngine:
    """Executes the generalized multi-qubit time-evolution."""
    
    def run(
        self, 
        system: QuantumSystem, 
        drives: Dict[str, np.ndarray], 
        t_list: np.ndarray,
        initial_state: qutip.Qobj,
        options: Optional[Dict[str, Any]] = None
    ) -> qutip.Result:
        
        h_static = system.build_static_hamiltonian()
        c_ops = system.build_collapse_operators()
        
        # PYLANCE FIX: Explicitly type 'h' as List[Any] to allow QuTiP's mixed-type Hamiltonian format
        h: List[Any] = [h_static]

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
        return qutip.mesolve(h, initial_state, t_list, c_ops=c_ops, options=solver_options)