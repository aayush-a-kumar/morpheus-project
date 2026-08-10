import numpy as np
import pandas as pd
from typing import Protocol, Dict, List, Any, Union

def extract_amplitude(source: Union[dict, pd.Series], default: Any = 0.0) -> Any:
    """Safely extract amplitude from a dict or pandas Series, handling None and NaN values.
    Preserves Variable objects for symbolic/loop resolution.
    """
    d = source.to_dict() if isinstance(source, pd.Series) else source
    
    if isinstance(d, dict):
        for key in ('amplitude', 'amp'):
            val = d.get(key)
            if val is not None:
                try:
                    if not pd.isna(val):
                        return val
                except Exception:
                    return val
    return default

class SignalProvider(Protocol):
    """Abstract interface for drive signal providers."""
    def get_drives(self, t_list: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns a map of drive channels to complex envelope arrays."""
        ...

class ScheduleSignalProvider:
    """Parses Qblox Schedule timing tables into IQ time-series arrays."""
    
    def __init__(self, pulses_list: List[Dict[str, Any]]):
        self.pulses_list = pulses_list

    def _pulse_envelope_vectorized(self, t_rel: np.ndarray, pulse_info: dict) -> np.ndarray:
        duration = pulse_info['duration']
        amp = extract_amplitude(pulse_info)
        phase_rad = np.deg2rad(pulse_info.get('phase', 0.0))

        wf_raw = pulse_info.get('wf_func')
        wf_func = str(wf_raw).lower() if wf_raw else 'square'
        
        if 'gauss' in wf_func:
            sigma = pulse_info.get('sigma', duration / 4)
            sigma = 1e-12 if (sigma is None or sigma == 0) else sigma
            t_mid = duration / 2
            envelope = amp * np.exp(-(t_rel - t_mid)**2 / (2 * sigma**2))
        elif 'drag' in wf_func:
            sigma = pulse_info.get('sigma', duration / 4)
            sigma = 1e-12 if (sigma is None or sigma == 0) else sigma
            beta = pulse_info.get('beta', 0.0)
            t_mid = duration / 2
            envelope = amp * np.exp(-(t_rel - t_mid)**2 / (2 * sigma**2))
            envelope_dot = -(t_rel - t_mid) / (sigma**2) * envelope
            return (envelope + 1j * (-beta * envelope_dot / (2 * np.pi))) * np.exp(1j * phase_rad)
        else:
            envelope = np.full_like(t_rel, amp, dtype=complex)
        
        return envelope * np.exp(1j * phase_rad)

    def get_drives(self, t_list: np.ndarray) -> Dict[str, np.ndarray]:
        q_drive = np.zeros_like(t_list, dtype=complex)
        res_drive = np.zeros_like(t_list, dtype=complex)
        
        if len(t_list) <= 1:
            return {"q_drive": q_drive, "res_drive": res_drive}
            
        dt_actual = t_list[1] - t_list[0]
        t_start_grid = t_list[0]

        for p in self.pulses_list:
            port = p.get('port')
            t_start = p['abs_time']
            duration = p['duration']
            t_end = t_start + duration
            
            idx_start = max(0, int(np.floor((t_start - t_start_grid) / dt_actual)))
            idx_end = min(len(t_list), int(np.ceil((t_end - t_start_grid) / dt_actual)) + 1)
            
            if idx_start >= len(t_list) or idx_end <= 0:
                continue
                
            t_slice = t_list[idx_start:idx_end]
            t_rel = t_slice - t_start
            mask = (t_rel >= 0) & (t_rel <= duration)
            
            if not np.any(mask):
                continue
                
            t_rel_valid = t_rel[mask]
            signal = self._pulse_envelope_vectorized(t_rel_valid, p)
            
            if port == 'q0:mw':
                q_drive[idx_start:idx_end][mask] += signal
            elif port == 'q0:res':
                res_drive[idx_start:idx_end][mask] += signal
                
        return {"q_drive": q_drive, "res_drive": res_drive}