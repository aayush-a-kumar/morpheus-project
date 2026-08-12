import itertools
import numpy as np
import qutip
from typing import Dict, Any, Tuple, List, Optional
from abc import ABC, abstractmethod
from qblox_sim.config import SimulationConfig

class AcquisitionHandler(ABC):
    """Abstract base class for acquisition processing strategies."""
    
    @abstractmethod
    def process(
        self, 
        state: qutip.Qobj, 
        t_list: np.ndarray, 
        states: List[qutip.Qobj], 
        acq_time: float,
        acq_duration: float,
        acq_info: Dict[str, Any], 
        cfg: SimulationConfig,
        a_op: qutip.Qobj, 
        ad_op: qutip.Qobj
    ) -> Dict[str, Any]:
        """Process a state into a measurement dictionary."""
        pass

    def _get_joint_probabilities(self, state: qutip.Qobj, cfg: SimulationConfig) -> Dict[str, float]:
        """Extracts N-qubit joint computational basis probabilities and marginals dynamically."""
        rho = state if state.type == 'oper' else qutip.ket2dm(state)
        n_qubits = len(cfg.qubits)
        
        # Trace out resonator modes if present in the global Hilbert space
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
            ket = qutip.tensor(basis_kets)
            
            p_bit = float(np.real(qutip.expect(qutip.ket2dm(ket), rho_q)))
            results[f'prob_{bitstring}'] = p_bit
            total_comp_prob += p_bit

        # Marginal probabilities for Qubit 0 (Primary readout reference)
        results['prob_0'] = sum(v for k, v in results.items() if k.startswith('prob_') and k[5] == '0')
        results['prob_1'] = sum(v for k, v in results.items() if k.startswith('prob_') and k[5] == '1')
        
        # Total population lost outside the 2^N computational manifold (e.g. |2> states)
        results['leakage_prob_2'] = max(0.0, 1.0 - total_comp_prob)
        
        return results


class SSBIntegrationHandler(AcquisitionHandler):
    """Projects ground/excited probabilities onto complex voltage centroids with noise."""
    
    def process(self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op):
        joint_probs = self._get_joint_probabilities(state, cfg)
        p0 = joint_probs['prob_0']
        p1 = joint_probs['prob_1']

        centroid = p0 * cfg.acquisition.v_0 + p1 * cfg.acquisition.v_1
        i_val = float(np.real(centroid) + np.random.normal(0, cfg.acquisition.noise_sigma))
        q_val = float(np.imag(centroid) + np.random.normal(0, cfg.acquisition.noise_sigma))
        val = i_val + 1j * q_val
        outcome = 1 if np.random.random() < p1 else 0

        res = {
            'outcome': outcome, 'I': i_val, 'Q': q_val, 'value': val
        }
        res.update(joint_probs)
        return res


class ThresholdedAcquisitionHandler(SSBIntegrationHandler):
    """State discrimination mapping to a discrete outcome based on rotation and threshold."""
    
    def process(self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op):
        base_result = super().process(state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op)
        
        acq_rotation = acq_info.get('acq_rotation', None)
        acq_threshold = acq_info.get('acq_threshold', None)

        if acq_rotation is not None and acq_threshold is not None and acq_threshold != 0:
            rot_rad = np.deg2rad(acq_rotation)
            val_rot = base_result['value'] * np.exp(1j * rot_rad)
            base_result['outcome'] = 1 if np.real(val_rot) > acq_threshold else 0
            base_result['value'] = base_result['outcome']
        else:
            base_result['value'] = base_result['outcome']
            
        return base_result


class TraceAcquisitionHandler(AcquisitionHandler):
    """Time of Flight (TOF) digitized time-series voltage wave <a(t) + a^dagger(t)>."""
    
    def process(self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op):
        joint_probs = self._get_joint_probabilities(state, cfg)

        acq_delay = acq_info.get('acq_delay', cfg.acquisition.cable_delay)
        t_eff_start = acq_time + acq_delay
        t_eff_end = t_eff_start + acq_duration

        mask = (t_list >= t_eff_start - 1e-12) & (t_list <= t_eff_end + 1e-12)
        indices = np.where(mask)[0]
        sigma = cfg.acquisition.noise_sigma

        if len(indices) == 0:
            num_samples = max(100, int(round(acq_duration * 1e9)))
            exp_val = float(np.real(qutip.expect(a_op + ad_op, state)))
            trace_data = exp_val + np.random.normal(0, sigma, size=num_samples)
        else:
            trace_data = np.array([
                float(np.real(qutip.expect(a_op + ad_op, states[i]))) + np.random.normal(0, sigma)
                for i in indices
            ])

        p1 = joint_probs['prob_1']
        outcome = 1 if np.random.random() < p1 else 0
        i_val = float(np.mean(trace_data)) if len(trace_data) > 0 else 0.0

        res = {
            'outcome': outcome, 'I': i_val, 'Q': 0.0, 'value': trace_data
        }
        res.update(joint_probs)
        return res


class AcquisitionRegistry:
    """Maps protocol names to their concrete processing strategy."""
    _handlers = {
        'SSBIntegrationComplex': SSBIntegrationHandler(),
        'SSBIntegration': SSBIntegrationHandler(),
        'Integration': SSBIntegrationHandler(),
        'ThresholdedAcquisition': ThresholdedAcquisitionHandler(),
        'Thresholded': ThresholdedAcquisitionHandler(),
        'Trace': TraceAcquisitionHandler(),
        'TraceAcquisition': TraceAcquisitionHandler(),
    }

    @classmethod
    def get_handler(cls, protocol: str) -> AcquisitionHandler:
        return cls._handlers.get(protocol, SSBIntegrationHandler())