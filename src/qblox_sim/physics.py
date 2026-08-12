import typing
import numpy as np
import qutip
from qblox_sim.config import SimulationConfig

class QuantumSystem:
    """Dynamically builds multi-qubit/multi-resonator Hilbert spaces and Hamiltonians."""
    
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        
        self.q_names = list(self.cfg.qubits.keys())
        self.r_names = list(self.cfg.resonators.keys())
        self.all_names = self.q_names + self.r_names
        
        self.dims = [self.cfg.qubits[q].N_q for q in self.q_names] + \
                    [self.cfg.resonators[r].N_res for r in self.r_names]

        # Core Operators
        self.b: typing.Dict[str, qutip.Qobj] = {}
        self.bd: typing.Dict[str, qutip.Qobj] = {}
        self.nq: typing.Dict[str, qutip.Qobj] = {}
        self.a: typing.Dict[str, qutip.Qobj] = {}
        self.ad: typing.Dict[str, qutip.Qobj] = {}
        self.n: typing.Dict[str, qutip.Qobj] = {}
        
        # Pauli Operators
        self.sx: typing.Dict[str, qutip.Qobj] = {}
        self.sy: typing.Dict[str, qutip.Qobj] = {}
        self.sz: typing.Dict[str, qutip.Qobj] = {}

        # Build generalized operators for the global tensor space
        for i, name in enumerate(self.all_names):
            dim = self.dims[i]
            
            op_list = [qutip.identity(d) for d in self.dims]
            op_list[i] = qutip.destroy(dim)
            global_op = qutip.tensor(op_list)

            # FIX 1: Use strictly bounded index 'i' to prevent 'q0' naming collisions
            if i < len(self.q_names):
                self.b[name] = global_op
                self.bd[name] = global_op.dag()
                self.nq[name] = self.bd[name] * self.b[name] # type: ignore[operator]
                
                # FIX 3: Consistent Pauli dictionaries
                x_list = [qutip.identity(d) for d in self.dims]
                y_list = [qutip.identity(d) for d in self.dims]
                z_list = [qutip.identity(d) for d in self.dims]
                
                if dim == 2:
                    x_list[i] = qutip.sigmax()
                    y_list[i] = qutip.sigmay()
                    z_list[i] = qutip.sigmaz()
                    self.sx[name] = qutip.tensor(x_list)
                    self.sy[name] = qutip.tensor(y_list)
                    self.sz[name] = qutip.tensor(z_list)
                else:
                    self.sx[name] = self.b[name] + self.bd[name]
                    self.sy[name] = 1j * (self.bd[name] - self.b[name])
                    
                    proj0 = qutip.basis(dim, 0) * qutip.basis(dim, 0).dag() #type: ignore[operator]
                    proj1 = qutip.basis(dim, 1) * qutip.basis(dim, 1).dag() #type: ignore[operator]
                    z_list[i] = proj0 - proj1 # type: ignore[operator]
                    self.sz[name] = qutip.tensor(z_list)
            else:
                self.a[name] = global_op
                self.ad[name] = global_op.dag()
                self.n[name] = self.ad[name] * self.a[name] # type: ignore[operator]

    def get_default_initial_state(self) -> qutip.Qobj:
        """Returns the global ground state |0, 0, ..., 0>."""
        state_list = [qutip.basis(d, 0) for d in self.dims]
        return qutip.tensor(state_list)

    def build_static_hamiltonian(self) -> qutip.Qobj:
        """Constructs the multi-qubit un-driven Hamiltonian with coupling terms."""
        H = qutip.tensor([qutip.qzero(d) for d in self.dims])

        for q_name, q_cfg in self.cfg.qubits.items():
            delta_q = 2 * np.pi * (q_cfg.f_q - q_cfg.f_d)
            b, bd, nq = self.b[q_name], self.bd[q_name], self.nq[q_name]
            H += -delta_q * nq + np.pi * q_cfg.alpha * (bd * bd * b * b)  # type: ignore[operator]

        for r_name, r_cfg in self.cfg.resonators.items():
            delta_res = 2 * np.pi * (r_cfg.f_res - r_cfg.f_d_res)
            a, n = self.a[r_name], self.n[r_name]
            H += delta_res * n

            if r_name in self.cfg.qubits:
                nq = self.nq[r_name]
                H += 2 * np.pi * r_cfg.chi * n * nq  # type: ignore[operator]

        for coupling in self.cfg.couplings:
            b1, bd1 = self.b[coupling.q1], self.bd[coupling.q1]
            b2, bd2 = self.b[coupling.q2], self.bd[coupling.q2]
            J = 2 * np.pi * coupling.J
            H += J * (bd1 * b2 + b1 * bd2)  # type: ignore[operator]

        return H

    def build_collapse_operators(self) -> list[qutip.Qobj]:
        """Constructs Lindblad operators for all configured components."""
        c_ops = []
        
        for q_name, q_cfg in self.cfg.qubits.items():
            if q_cfg.T1 < np.inf: 
                c_ops.append(np.sqrt(1.0 / q_cfg.T1) * self.b[q_name])
            if q_cfg.T2 < np.inf:
                gamma_phi = (1.0 / q_cfg.T2) - (0.5 / q_cfg.T1 if q_cfg.T1 < np.inf else 0)
                if gamma_phi > 0: 
                    c_ops.append(np.sqrt(2 * gamma_phi) * self.nq[q_name])
                    
        for r_name, r_cfg in self.cfg.resonators.items():
            if r_cfg.kappa > 0: 
                c_ops.append(np.sqrt(r_cfg.kappa) * self.a[r_name])
                
        return c_ops