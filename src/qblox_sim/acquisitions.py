# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Translates quantum states into simulated hardware measurement records.

This module is responsible for projecting QuTiP state vectors or density matrices
into probabilities, and then mapping those probabilities onto physical hardware
observables like noisy IQ voltages, thresholded binary outcomes, or continuous
time-domain traces.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import qutip

from qblox_sim.config import SimulationConfig


class AcquisitionHandler(ABC):
    """Abstract base class for acquisition processing strategies."""

    @abstractmethod
    def process(
        self,
        state: qutip.Qobj,
        t_list: np.ndarray,
        states: list[qutip.Qobj],
        acq_time: float,
        acq_duration: float,
        acq_info: dict[str, Any],
        cfg: SimulationConfig,
        a_op: qutip.Qobj,
        ad_op: qutip.Qobj,
    ) -> dict[str, Any]:
        """
        Process a state into a measurement dictionary.

        Args:
            state (qutip.Qobj): The QuTiP state at the exact time of the acquisition trigger.
            t_list (np.ndarray): The simulation time grid.
            states (list[qutip.Qobj]): The full history of QuTiP states over time.
            acq_time (float): The absolute start time of the acquisition.
            acq_duration (float): The length of the integration or trace window.
            acq_info (dict[str, Any]): Dictionary of schedule parameters (e.g., rotation, threshold).
            cfg (SimulationConfig): The static simulation hardware configuration.
            a_op (qutip.Qobj): The annihilation operator for the relevant readout resonator.
            ad_op (qutip.Qobj): The creation operator for the relevant readout resonator.

        Returns:
            dict[str, Any]: A dictionary containing measurement results (e.g., I, Q, outcome, trace).
        """

    def _get_joint_probabilities(
        self, state: qutip.Qobj, cfg: SimulationConfig
    ) -> dict[str, float]:
        """
        Extracts N-qubit joint computational basis probabilities and marginals dynamically.

        Args:
            state (qutip.Qobj): The quantum state (ket or density matrix).
            cfg (SimulationConfig): Configuration containing the list of configured qubits.

        Returns:
            dict[str, float]: Probabilities for all bitstrings, marginals, and leakage.
        """
        rho = state if state.type == "oper" else qutip.ket2dm(state)
        n_qubits = len(cfg.qubits)

        # WHY: The global Hilbert space includes resonator photon states. We must partial-trace
        # out the resonators to isolate the reduced density matrix of just the qubits.
        if len(rho.dims[0]) > n_qubits:
            rho_q = rho.ptrace(list(range(n_qubits)))
        else:
            rho_q = rho

        q_dims = rho_q.dims[0]
        results = {}
        total_comp_prob = 0.0

        # Dynamically generate all 2^N computational bitstrings: '00..0' to '11..1'
        for bits in itertools.product([0, 1], repeat=n_qubits):
            bitstring = "".join(map(str, bits))

            # Construct tensor product basis state |b0, b1, ..., bN-1>
            basis_kets = [qutip.basis(q_dims[i], bits[i]) for i in range(n_qubits)]
            ket = qutip.tensor(*basis_kets)

            # WHY: Project the reduced density matrix onto each computational basis state
            # to find the probability of measuring that specific bitstring.
            p_bit = float(np.real(qutip.expect(qutip.ket2dm(ket), rho_q)))
            results[f"prob_{bitstring}"] = p_bit
            total_comp_prob += p_bit

        # Marginal probabilities for Qubit 0 (Primary readout reference)
        results["prob_0"] = sum(
            v for k, v in results.items() if k.startswith("prob_") and k[5] == "0"
        )
        results["prob_1"] = sum(
            v for k, v in results.items() if k.startswith("prob_") and k[5] == "1"
        )

        # Total population lost outside the 2^N computational manifold (e.g. |2> states)
        # WHY: If transmons leak into the |2> state (e.g., due to poor DRAG tuning),
        # the total probability in the {0,1} subspace will be less than 1.0.
        results["leakage_prob_2"] = max(0.0, 1.0 - total_comp_prob)

        return results


class SSBIntegrationHandler(AcquisitionHandler):
    """Projects ground/excited probabilities onto complex voltage centroids with noise."""

    def process(
        self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op
    ):
        joint_probs = self._get_joint_probabilities(state, cfg)
        p0 = joint_probs["prob_0"]
        p1 = joint_probs["prob_1"]

        # WHY: Single Sideband (SSB) Integration emulates the hardware demodulation.
        # The resulting complex voltage is a weighted average of the pre-calibrated
        # ground and excited state centroids in the IQ plane.
        centroid = p0 * cfg.acquisition.v_0 + p1 * cfg.acquisition.v_1

        # WHY: Add Gaussian noise to simulate thermal and amplifier noise in the readout chain.
        i_val = float(
            np.real(centroid) + np.random.normal(0, cfg.acquisition.noise_sigma)
        )
        q_val = float(
            np.imag(centroid) + np.random.normal(0, cfg.acquisition.noise_sigma)
        )
        val = i_val + 1j * q_val

        # WHY: Generate a discrete measurement outcome by collapsing the state
        # based on the calculated marginal probability of being in |1>.
        outcome = 1 if np.random.random() < p1 else 0

        res = {"outcome": outcome, "I": i_val, "Q": q_val, "value": val}
        res.update(joint_probs)
        return res


class ThresholdedAcquisitionHandler(SSBIntegrationHandler):
    """State discrimination mapping to a discrete outcome based on rotation and threshold."""

    def process(
        self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op
    ):
        base_result = super().process(
            state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op
        )

        acq_rotation = acq_info.get("acq_rotation", None)
        acq_threshold = acq_info.get("acq_threshold", None)

        if (
            acq_rotation is not None
            and acq_threshold is not None
            and acq_threshold != 0
        ):
            # WHY: Hardware state discrimination typically rotates the IQ voltage blob
            # so that the separation between |0> and |1> lies entirely along the Real (I) axis.
            rot_rad = np.deg2rad(acq_rotation)
            val_rot = base_result["value"] * np.exp(1j * rot_rad)

            # WHY: A hard threshold is applied to the rotated Real axis to digitize the state.
            base_result["outcome"] = 1 if np.real(val_rot) > acq_threshold else 0
            base_result["value"] = base_result["outcome"]
        else:
            base_result["value"] = base_result["outcome"]

        return base_result


class TraceAcquisitionHandler(AcquisitionHandler):
    """Time of Flight (TOF) digitized time-series voltage wave <a(t) + a^dagger(t)>."""

    def process(
        self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op
    ):
        joint_probs = self._get_joint_probabilities(state, cfg)

        # WHY: Cable delay shifts the window where the hardware digitizes the returning pulse.
        acq_delay = acq_info.get("acq_delay", cfg.acquisition.cable_delay)
        t_eff_start = acq_time + acq_delay
        t_eff_end = t_eff_start + acq_duration

        mask = (t_list >= t_eff_start - 1e-12) & (t_list <= t_eff_end + 1e-12)
        indices = np.where(mask)[0]
        sigma = cfg.acquisition.noise_sigma

        if len(indices) == 0:
            num_samples = max(100, round(acq_duration * 1e9))
            # WHY: The continuous voltage trace is proportional to the expectation value
            # of the cavity field displacement <a + a^dagger>.
            exp_val = float(np.real(qutip.expect(a_op + ad_op, state)))
            trace_data = exp_val + np.random.normal(0, sigma, size=num_samples)
        else:
            # WHY: Reconstruct the continuous Time of Flight (TOF) measurement record
            # by evaluating the cavity field at each time step in the acquisition window.
            trace_data = np.array(
                [
                    float(np.real(qutip.expect(a_op + ad_op, states[i])))
                    + np.random.normal(0, sigma)
                    for i in indices
                ]
            )

        p1 = joint_probs["prob_1"]
        outcome = 1 if np.random.random() < p1 else 0
        i_val = float(np.mean(trace_data)) if len(trace_data) > 0 else 0.0

        res = {"outcome": outcome, "I": i_val, "Q": 0.0, "value": trace_data}
        res.update(joint_probs)
        return res


from typing import ClassVar


class AcquisitionRegistry:
    """Maps protocol names to their concrete processing strategy."""

    _handlers: ClassVar[dict[str, AcquisitionHandler]] = {
        "SSBIntegrationComplex": SSBIntegrationHandler(),
        "SSBIntegration": SSBIntegrationHandler(),
        "Integration": SSBIntegrationHandler(),
        "ThresholdedAcquisition": ThresholdedAcquisitionHandler(),
        "Thresholded": ThresholdedAcquisitionHandler(),
        "Trace": TraceAcquisitionHandler(),
        "TraceAcquisition": TraceAcquisitionHandler(),
    }

    @classmethod
    def get_handler(cls, protocol: str) -> AcquisitionHandler:
        """
        Retrieves the appropriate acquisition processor for a given protocol string.
        """
        return cls._handlers.get(protocol, SSBIntegrationHandler())
