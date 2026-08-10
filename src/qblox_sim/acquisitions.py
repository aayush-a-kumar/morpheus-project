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

    def _get_probabilities(self, rho_q: qutip.Qobj, cfg: SimulationConfig) -> Tuple[float, float, float]:
        """Helper to extract level probabilities dynamically based on actual state dimension."""
        # Dynamically read the dimension of the reduced density matrix
        dim = rho_q.shape[0]
        
        prob_0 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(dim, 0)), rho_q))) if dim > 0 else 0.0
        prob_1 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(dim, 1)), rho_q))) if dim > 1 else 0.0
        prob_2 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(dim, 2)), rho_q))) if dim > 2 else 0.0
        
        return prob_0, prob_1, prob_2


class SSBIntegrationHandler(AcquisitionHandler):
    """Projects ground/excited probabilities onto complex voltage centroids with noise."""
    
    def process(self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op):
        rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
        p0, p1, p2 = self._get_probabilities(rho_q, cfg)

        centroid = p0 * cfg.acquisition.v_0 + p1 * cfg.acquisition.v_1
        i_val = float(np.real(centroid) + np.random.normal(0, cfg.acquisition.noise_sigma))
        q_val = float(np.imag(centroid) + np.random.normal(0, cfg.acquisition.noise_sigma))
        val = i_val + 1j * q_val
        outcome = 1 if np.random.random() < p1 else 0

        return {
            'prob_0': p0, 'prob_1': p1, 'leakage_prob_2': p2,
            'outcome': outcome, 'I': i_val, 'Q': q_val, 'value': val
        }


class ThresholdedAcquisitionHandler(SSBIntegrationHandler):
    """State discrimination mapping to a discrete 0 or 1 outcome based on rotation and threshold."""
    
    def process(self, state, t_list, states, acq_time, acq_duration, acq_info, cfg, a_op, ad_op):
        # Base class handles probability extraction and noise integration
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
        rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
        p0, p1, p2 = self._get_probabilities(rho_q, cfg)

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

        outcome = 1 if np.random.random() < p1 else 0
        i_val = float(np.mean(trace_data)) if len(trace_data) > 0 else 0.0

        return {
            'prob_0': p0, 'prob_1': p1, 'leakage_prob_2': p2,
            'outcome': outcome, 'I': i_val, 'Q': 0.0, 'value': trace_data
        }


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