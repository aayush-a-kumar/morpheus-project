import typing
import numpy as np
import qutip
from qblox_sim.config import SimulationConfig


class QuantumSystem:
    """Encapsulates Hilbert spaces, operators, static Hamiltonian, and decay terms."""
    
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        
        N_q = self.cfg.qubit.N_q
        N_res = self.cfg.resonator.N_res

        # -------------------------------------------------------------
        # Transmon & Readout Operators (Hilbert Space: N_q x N_res)
        # -------------------------------------------------------------
        self.b = qutip.tensor(qutip.destroy(N_q), qutip.identity(N_res))
        self.bd = self.b.dag()
        self.nq = self.bd * self.b  # type: ignore[operator]

        self.a = qutip.tensor(qutip.identity(N_q), qutip.destroy(N_res))
        self.ad = self.a.dag()
        self.n = self.ad * self.a  # type: ignore[operator]

        # -------------------------------------------------------------
        # Backward-Compatible Pauli / Subspace Operators
        # -------------------------------------------------------------
        self.sm = self.b
        
        if N_q == 2:
            self.sx = qutip.tensor(qutip.sigmax(), qutip.identity(N_res))
            self.sy = qutip.tensor(qutip.sigmay(), qutip.identity(N_res))
            self.sz = qutip.tensor(qutip.sigmaz(), qutip.identity(N_res))
        else:
            self.sx = self.b + self.bd
            self.sy = 1j * (self.bd - self.b)
            proj_0 = qutip.tensor(qutip.basis(N_q, 0) * qutip.basis(N_q, 0).dag(), qutip.identity(N_res))  # type: ignore[operator]
            proj_1 = qutip.tensor(qutip.basis(N_q, 1) * qutip.basis(N_q, 1).dag(), qutip.identity(N_res))  # type: ignore[operator]
            self.sz = proj_0 - proj_1

    def get_default_initial_state(self) -> qutip.Qobj:
        """Returns the ground state |0, 0> in the tensor space N_q x N_res."""
        return qutip.tensor(qutip.basis(self.cfg.qubit.N_q, 0), qutip.basis(self.cfg.resonator.N_res, 0))

    def build_static_hamiltonian(self) -> qutip.Qobj:
        """Constructs the time-independent portion of the Hamiltonian."""
        delta_q = 2 * np.pi * (self.cfg.qubit.f_q - self.cfg.qubit.f_d)
        delta_res = 2 * np.pi * (self.cfg.resonator.f_res - self.cfg.resonator.f_d_res)
        
        return (
            -delta_q * self.nq 
            + np.pi * self.cfg.qubit.alpha * (self.bd * self.bd * self.b * self.b)  # type: ignore[operator]
            + delta_res * self.n 
            + 2 * np.pi * self.cfg.resonator.chi * self.n * self.nq  # type: ignore[operator]
        )

    def build_collapse_operators(self) -> list[qutip.Qobj]:
        """Constructs the list of Lindblad collapse operators (T1, T2, kappa)."""
        c_ops = []
        if self.cfg.qubit.T1 < np.inf: 
            c_ops.append(np.sqrt(1.0 / self.cfg.qubit.T1) * self.b)
            
        if self.cfg.qubit.T2 < np.inf:
            gamma_phi = (1.0 / self.cfg.qubit.T2) - (0.5 / self.cfg.qubit.T1 if self.cfg.qubit.T1 < np.inf else 0)
            if gamma_phi > 0: 
                c_ops.append(np.sqrt(2 * gamma_phi) * self.nq)
                
        if self.cfg.resonator.kappa > 0: 
            c_ops.append(np.sqrt(self.cfg.resonator.kappa) * self.a)
            
        return c_ops