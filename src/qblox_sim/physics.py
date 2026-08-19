# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Defines the static physics models for the quantum simulation.

This module is responsible for constructing the global Hilbert space,
building generalized creation/annihilation operators for qubits and resonators,
and assembling the un-driven (static) Hamiltonian and Lindblad collapse operators.
"""

import numpy as np
import qutip
import scipy.linalg

from qblox_sim.config import SimulationConfig


class QuantumSystem:
    """
    Dynamically builds multi-qubit/multi-resonator Hilbert spaces and Hamiltonians.

    This class reads the SimulationConfig and provisions the required tensor-product
    space for all components. It acts as the central state holder for all static
    operators used by the solver and acquisition processors.
    """

    def __init__(self, cfg: SimulationConfig):
        """
        Initializes the QuantumSystem and constructs the global tensor space operators.

        Args:
            cfg (SimulationConfig): The parsed simulation configuration dictating
                qubit frequencies, dimensions, and coupling typologies.
        """
        self.cfg = cfg

        self.q_names = list(self.cfg.qubits.keys())
        self.r_names = list(self.cfg.resonators.keys())
        self.all_names = self.q_names + self.r_names

        # WHY: The total Hilbert space is the tensor product of all individual
        # sub-systems (qubits and readout resonators). We need the dimensions of each
        # to construct global operators.
        self.dims = [self.cfg.qubits[q].N_q for q in self.q_names] + [
            self.cfg.resonators[r].N_res for r in self.r_names
        ]

        # Core Operators
        self.b: dict[str, qutip.Qobj] = {}
        self.bd: dict[str, qutip.Qobj] = {}
        self.nq: dict[str, qutip.Qobj] = {}
        self.a: dict[str, qutip.Qobj] = {}
        self.ad: dict[str, qutip.Qobj] = {}
        self.n: dict[str, qutip.Qobj] = {}

        # Pauli Operators
        self.sx: dict[str, qutip.Qobj] = {}
        self.sy: dict[str, qutip.Qobj] = {}
        self.sz: dict[str, qutip.Qobj] = {}

        # Build generalized operators for the global tensor space
        for i, name in enumerate(self.all_names):
            dim = self.dims[i]

            # WHY: To apply an operator to a single subsystem within the global space,
            # we tensor it with identity matrices for all other sub-systems.
            op_list = [qutip.identity(d) for d in self.dims]
            op_list[i] = qutip.destroy(dim)
            global_op = qutip.tensor(*op_list)

            # FIX 1: Use strictly bounded index 'i' to prevent 'q0' naming collisions
            if i < len(self.q_names):
                # WHY: Qubits (transmons) use b and b^dagger to represent their excitations.
                self.b[name] = global_op
                self.bd[name] = global_op.dag()
                self.nq[name] = self.bd[name] * self.b[name]  # type: ignore[operator]

                # FIX 3: Consistent Pauli dictionaries
                x_list = [qutip.identity(d) for d in self.dims]
                y_list = [qutip.identity(d) for d in self.dims]
                z_list = [qutip.identity(d) for d in self.dims]

                if dim == 2:
                    # WHY: If strictly a 2-level system, standard Pauli matrices perfectly describe the dynamics.
                    x_list[i] = qutip.sigmax()
                    y_list[i] = qutip.sigmay()
                    z_list[i] = qutip.sigmaz()
                    self.sx[name] = qutip.tensor(*x_list)
                    self.sy[name] = qutip.tensor(*y_list)
                    self.sz[name] = qutip.tensor(*z_list)
                else:
                    # WHY: If multi-level (dim > 2), we use generalized operators.
                    # Transmon X and Y rotations map well to (b + bd) and 1j(bd - b).
                    self.sx[name] = self.b[name] + self.bd[name]
                    self.sy[name] = 1j * (self.bd[name] - self.b[name])

                    # WHY: Generalized Sigma-Z is defined as the projector onto the computational basis difference: |0><0| - |1><1|.
                    proj0 = qutip.basis(dim, 0) * qutip.basis(dim, 0).dag()  # type: ignore[operator]
                    proj1 = qutip.basis(dim, 1) * qutip.basis(dim, 1).dag()  # type: ignore[operator]
                    z_list[i] = proj0 - proj1  # type: ignore[operator]
                    self.sz[name] = qutip.tensor(*z_list)
            else:
                # WHY: Resonators are linear harmonic oscillators, using a and a^dagger operators.
                self.a[name] = global_op
                self.ad[name] = global_op.dag()
                self.n[name] = self.ad[name] * self.a[name]  # type: ignore[operator]
        # WHY: Cache the static Hamiltonian to avoid rebuilding it during integration.
        self.h_static = self.build_static_hamiltonian()

        # --- Spectral & Decay Pre-calculations ---
        # WHY: Evaluate spectral decomposition once at setup so engine.py can
        # perform O(1) unitary leaps over idle time gaps.
        h_static_dense = self.h_static.full()

        # Guard against completely empty topologies breaking eigh
        if h_static_dense.shape == (1, 1) and np.all(h_static_dense == 0):
            self.eigenvalues = np.array([0.0])
            self.eigenvectors = np.array([[1.0]])
        else:
            self.eigenvalues, self.eigenvectors = scipy.linalg.eigh(h_static_dense)

        self.v_dag = self.eigenvectors.conj().T

        # WHY: Pre-calculate the energy differences (E_j - E_k) for the phase rotation matrix.
        self.delta_e = self.eigenvalues[:, None] - self.eigenvalues[None, :]

        # WHY: Construct the global Lie-Trotter decay rate matrix.
        # This applies phenomenological T1/T2 damping across the system off-diagonals.
        dim = h_static_dense.shape[0]
        self.decay_rate_matrix = np.zeros((dim, dim))

        total_gamma = 0.0
        for q_cfg in self.cfg.qubits.values():
            if q_cfg.T1 < np.inf:
                total_gamma += 1.0 / q_cfg.T1
            if q_cfg.T2 < np.inf:
                total_gamma += 1.0 / q_cfg.T2

        # Apply total gamma damping strictly to the off-diagonal elements
        mask = ~np.eye(dim, dtype=bool)
        self.decay_rate_matrix[mask] = total_gamma / 2.0

    def get_default_initial_state(self) -> qutip.Qobj:
        """
        Returns the global ground state |0, 0, ..., 0>.

        Returns:
            qutip.Qobj: A tensor product representing the pure ground state of the system.
        """
        state_list = [qutip.basis(d, 0) for d in self.dims]

        # WHY: If the SimulationConfig contains no qubits or resonators, state_list is empty.
        # qutip.tensor(*[]) raises a TypeError, so we fallback to a safe 1D scalar ground state.
        if not state_list:
            return qutip.basis(1, 0)
        return qutip.tensor(*state_list)

    def build_static_hamiltonian(self) -> qutip.Qobj:
        """
        Constructs the multi-qubit un-driven Hamiltonian with coupling terms.

        Returns:
            qutip.Qobj: The static Hamiltonian (H0) matrix.
        """
        H = qutip.tensor(*[qutip.qzero(d) for d in self.dims])

        for q_name, q_cfg in self.cfg.qubits.items():
            # WHY: Transforms into the drive frame. Subtracting the drive frequency
            # removes the fast-oscillating terms (Rotating Wave Approximation).
            delta_q = 2 * np.pi * (q_cfg.f_q - q_cfg.f_d)
            b, bd, nq = self.b[q_name], self.bd[q_name], self.nq[q_name]

            # WHY: Transmons are modeled as Duffing oscillators.
            # -delta_q * nq is the detuning term.
            # np.pi * alpha * (bd * bd * b * b) applies the negative anharmonicity
            # penalty, pushing the |1> -> |2> transition down in frequency by alpha.
            H += -delta_q * nq + np.pi * q_cfg.alpha * (bd * bd * b * b)  # type: ignore[operator]

        for r_name, r_cfg in self.cfg.resonators.items():
            # WHY: Resonator detuning in the drive frame.
            delta_res = 2 * np.pi * (r_cfg.f_res - r_cfg.f_d_res)
            n = self.n[r_name]
            H += delta_res * n

            if r_name in self.cfg.qubits:
                nq = self.nq[r_name]
                # WHY: Dispersive coupling (chi) shifts the resonator frequency
                # based on the qubit state. This allows for QND (Quantum Non-Demolition)
                # readout by probing the resonator frequency.
                H += 2 * np.pi * r_cfg.chi * n * nq  # type: ignore[operator]

        for coupling in self.cfg.couplings:
            b1, bd1 = self.b[coupling.q1], self.bd[coupling.q1]
            b2, bd2 = self.b[coupling.q2], self.bd[coupling.q2]
            J = 2 * np.pi * coupling.J
            # WHY: Transverse dipole-dipole exchange coupling (Jaynes-Cummings/XY model).
            # This term allows excitation swapping between two coupled qubits,
            # necessary for two-qubit gates like iSWAP.
            H += J * (bd1 * b2 + b1 * bd2)  # type: ignore[operator]

        return H

    def build_collapse_operators(self) -> list[qutip.Qobj]:
        """
        Constructs Lindblad operators for all configured components to model dissipation.

        Returns:
            list[qutip.Qobj]: A list of collapse operators used by the Master Equation.
        """
        c_ops = []

        for q_name, q_cfg in self.cfg.qubits.items():
            if q_cfg.T1 < np.inf:
                # WHY: T1 represents amplitude damping (energy relaxation).
                # The collapse operator scales the annihilation operator by sqrt(1/T1).
                c_ops.append(np.sqrt(1.0 / q_cfg.T1) * self.b[q_name])
            if q_cfg.T2 < np.inf:
                # WHY: Pure dephasing rate (Gamma_phi) is the total dephasing rate (1/T2)
                # minus the dephasing caused inherently by T1 relaxation (1/(2*T1)).
                gamma_phi = (1.0 / q_cfg.T2) - (
                    0.5 / q_cfg.T1 if q_cfg.T1 < np.inf else 0
                )
                if gamma_phi > 0:
                    # WHY: Dephasing acts on the Z basis, causing loss of phase coherence
                    # without energy loss, modeled using the number operator.
                    c_ops.append(np.sqrt(2 * gamma_phi) * self.nq[q_name])

        for r_name, r_cfg in self.cfg.resonators.items():
            if r_cfg.kappa > 0:
                # WHY: Kappa is the photon decay rate of the readout resonator
                # into the environment/transmission line.
                c_ops.append(np.sqrt(r_cfg.kappa) * self.a[r_name])

        return c_ops
