from unittest import result
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule, SerialCompiler, QuantumDevice, BasicTransmonElement
from qblox_scheduler.operations import SquarePulse, GaussPulse, DRAGPulse, LoopOperation
from qblox_sim.config import SimulationConfig
from qblox_sim.physics import QuantumSystem
import typing

class SimulationResult:
    """Clean container for stitched time-series simulation outputs."""
    def __init__(
        self, 
        states: list, 
        t_list: np.ndarray, 
        measurements: list,
        simulator: typing.Optional[typing.Any] = None
    ):
        self.states = states
        self.t_list = t_list
        self.measurements = measurements
        # Explicit attribute declaration for type checkers
        self._sim: typing.Optional[typing.Any] = simulator

    def get_expectation(self, operator: str = 'sz') -> np.ndarray:
        """Helper to extract expectation values without worrying about N_q dimensions."""
        op = getattr(self._sim, operator, None) if self._sim else None
        if op is None:
            raise ValueError(f"Operator '{operator}' not found on simulator.")
            
        return np.array([qutip.expect(op, s).real for s in self.states])


class QbloxQutipSimulator:
    # ... (existing methods)
    """
    A simulator that takes a Qblox-Scheduler Schedule and uses QuTiP to simulate 
    the dynamics of a qubit coupled to a readout resonator.
    """

    def __init__(self, params: dict, configs: typing.Optional[dict] = None):
        self.params = params  # Keeping original reference for safety during refactor
        self.configs = configs
        
        # --- PHASE 1: Wire in Configuration Layer ---
        self.cfg = SimulationConfig.from_dict(params)
        
        # --- STRANGLER PATTERN: Temporary mapping ---
        # We bind these to self.* so the rest of the legacy code still works.
        # These will be completely removed in Phase 2!
        self.f_q = self.cfg.qubit.f_q
        self.f_d = self.cfg.qubit.f_d
        self.f_res = self.cfg.resonator.f_res
        self.f_d_res = self.cfg.resonator.f_d_res
        self.chi = self.cfg.resonator.chi
        self.cable_delay = self.cfg.acquisition.cable_delay
        
        self.N_q = self.cfg.qubit.N_q
        self.alpha = self.cfg.qubit.alpha
        self.rabi_freq_per_volt = self.cfg.qubit.rabi_freq_per_volt
        self.rabi_freq_res_per_volt = self.cfg.resonator.rabi_freq_res_per_volt
        self.T1 = self.cfg.qubit.T1
        self.T2 = self.cfg.qubit.T2
        self.kappa = self.cfg.resonator.kappa
        self.N_res = self.cfg.resonator.N_res

        self.system = QuantumSystem(self.cfg)
        
        # STRANGLER FIX: Map operators so external users (like `get_expectation`) don't break
        self.b = self.system.b
        self.bd = self.system.bd
        self.nq = self.system.nq
        self.a = self.system.a
        self.ad = self.system.ad
        self.n = self.system.n
        self.sm = self.system.sm
        self.sx = self.system.sx
        self.sy = self.system.sy
        self.sz = self.system.sz

    # =================================----------------========================
    # Acquisition Protocol Handlers
    # =========================================================================
    def _process_ssb_integration(self, rho_q: qutip.Qobj, acq_info: typing.Optional[dict] = None) -> typing.Tuple[complex, float, float]:
        """
        1. SSBIntegrationComplex Handler
        Projects ground/excited probabilities onto complex voltage centroids with noise.
        """
        sigma = self.cfg.acquisition.noise_sigma
        v_0 = self.cfg.acquisition.v_0
        v_1 = self.cfg.acquisition.v_1

        prob_0 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.cfg.qubit.N_q, 0)), rho_q)))
        prob_1 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.cfg.qubit.N_q, 1)), rho_q)))

        centroid = prob_0 * v_0 + prob_1 * v_1
        I_val = float(np.real(centroid) + np.random.normal(0, sigma))
        Q_val = float(np.imag(centroid) + np.random.normal(0, sigma))
        return I_val + 1j * Q_val, I_val, Q_val

    def _process_thresholded(self, rho_q: qutip.Qobj, acq_info: typing.Optional[dict] = None) -> int:
        """
        2. ThresholdedAcquisition Handler
        State discrimination mapping to discrete 0 or 1.
        """
        acq_info = acq_info or {}
        
        # Cleanly use config instead of self.N_q
        prob_1 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.cfg.qubit.N_q, 1)), rho_q)))

        acq_rotation = acq_info.get('acq_rotation', None)
        acq_threshold = acq_info.get('acq_threshold', None)

        if acq_rotation is not None and acq_threshold is not None and acq_threshold != 0:
            val, _, _ = self._process_ssb_integration(rho_q, acq_info)
            rot_rad = np.deg2rad(acq_rotation)
            val_rot = val * np.exp(1j * rot_rad)
            outcome = 1 if np.real(val_rot) > acq_threshold else 0
        else:
            outcome = 1 if np.random.random() < prob_1 else 0

        return int(outcome)

    def _process_trace(
        self, 
        t_list: np.ndarray, 
        states: list, 
        acq_start_sec: float, 
        acq_duration: float, 
        acq_info: typing.Optional[dict] = None
    ) -> np.ndarray:
        r"""
        Time of Flight (TOF) digitized time-series voltage wave <a(t) + a^\dagger(t)>.
        """
        acq_info = acq_info or {}

        # 1. STRANGLER FIX: Use clean config for cable delay
        default_delay = self.cfg.acquisition.cable_delay
        acq_delay = acq_info.get('acq_delay', default_delay)

        t_effective_start = acq_start_sec + acq_delay
        t_effective_end = t_effective_start + acq_duration

        mask = (t_list >= t_effective_start - 1e-12) & (t_list <= t_effective_end + 1e-12)
        indices = np.where(mask)[0]

        # 2. STRANGLER FIX: Use clean config for noise
        sigma = self.cfg.acquisition.noise_sigma

        if len(indices) == 0:
            num_samples = max(100, int(round(acq_duration * 1e9)))
            idx_closest = np.argmin(np.abs(t_list - acq_start_sec))
            state = states[idx_closest]
            exp_val = float(np.real(qutip.expect(self.a + self.ad, state)))
            trace_data = exp_val + np.random.normal(0, sigma, size=num_samples)
        else:
            trace_data = np.array([
                float(np.real(qutip.expect(self.a + self.ad, states[i]))) + np.random.normal(0, sigma)
                for i in indices
            ])

        return trace_data

    # =========================================================================
    # Helpers & Compilation
    # =========================================================================
    @staticmethod
    def _extract_amplitude(source: typing.Union[dict, pd.Series], default: typing.Any = 0.0) -> typing.Any:
        """Safely extract amplitude from a dict or pandas Series, handling None and NaN values.
        Preserves Variable objects for symbolic/loop resolution.
        """
        # Type narrowing: only call .to_dict() if source is a pd.Series
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

    def _flatten_operations(self, operations_dict: dict, all_ops: typing.Optional[dict] = None) -> dict:
        """Recursively flatten nested schedule operations dictionary."""
        if all_ops is None:
            all_ops = {}
        for h, op in operations_dict.items():
            all_ops[h] = op
            body = getattr(op, 'body', None)
            if body is not None and hasattr(body, 'operations'):
                self._flatten_operations(getattr(body, 'operations'), all_ops)
            elif hasattr(op, 'operations'):
                self._flatten_operations(getattr(op, 'operations'), all_ops)
        return all_ops

    def _get_op_from_hash(self, op_hash: typing.Any, ops_dict: dict) -> dict:
        """Retrieve operation dictionary cleanly with string/int hash fallback."""
        op = ops_dict.get(op_hash)
        if op is None and isinstance(op_hash, (int, str)):
            try:
                op = ops_dict.get(int(op_hash)) or ops_dict.get(str(op_hash))
            except ValueError:
                pass
        return op if op is not None else {}

    def _get_compiled_schedule(self, schedule: Schedule) -> typing.Tuple[pd.DataFrame, dict]:
        try:
            return schedule.timing_table.data, schedule.operations # type: ignore[attr-defined]
        except Exception:
            device = QuantumDevice(name="dummy_device")
            try:
                q0 = BasicTransmonElement("q0") #type: ignore
                # Assign non-NaN default floats to the fallback dummy device
                q0.clock_freqs.readout = 7.0e9
                q0.clock_freqs.f01 = 5.0e9
                device.add_element(q0)
            except Exception:
                pass
            compiler = SerialCompiler(name="compiler", quantum_device=device)
            compiled_sched = compiler.compile(schedule)
            return compiled_sched.timing_table.data, compiled_sched.operations # type: ignore[attr-defined]

    #NOTE: Deprecated!!
    def _pulse_envelope(self, t: float, pulse_info: dict) -> complex:
        t_start = pulse_info['abs_time'] 
        duration = pulse_info['duration']
        t_rel = t - t_start
        
        if t_rel < 0 or t_rel > duration:
            return 0.0j
        
        amp = self._extract_amplitude(pulse_info)
        phase_deg = pulse_info.get('phase', 0.0)
        phase_rad = np.deg2rad(phase_deg)

        #janky waveform function handling 
        wf_raw = pulse_info.get('wf_func')
        wf_func = str(wf_raw).lower() if wf_raw else 'square'
        
        if 'gauss' in wf_func:
            sigma = pulse_info.get('sigma', duration / 4)
            if sigma is None: sigma = duration / 4
            if sigma == 0: sigma = 1e-12
            t_mid = duration / 2
            envelope = amp * np.exp(-(t_rel - t_mid)**2 / (2 * sigma**2))
        elif 'drag' in wf_func:
            sigma = pulse_info.get('sigma', duration / 4)
            if sigma is None: sigma = duration / 4
            if sigma == 0: sigma = 1e-12
            beta = pulse_info.get('beta', 0.0)
            t_mid = duration / 2
            envelope = amp * np.exp(-(t_rel - t_mid)**2 / (2 * sigma**2))
            envelope_dot = -(t_rel - t_mid) / (sigma**2) * envelope
            return (envelope + 1j * (-beta * envelope_dot / (2 * np.pi))) * np.exp(1j * phase_rad)
        else:
            # Default to square pulse for any unspecified or square waveform
            envelope = amp
        
        return envelope * np.exp(1j * phase_rad)

    def _pulse_envelope_vectorized(self, t_rel: np.ndarray, pulse_info: dict) -> np.ndarray:
        """Vectorized evaluation of the pulse envelope over an array of relative times."""
        duration = pulse_info['duration']
        amp = self._extract_amplitude(pulse_info)
        phase_rad = np.deg2rad(pulse_info.get('phase', 0.0))

        wf_raw = pulse_info.get('wf_func')
        wf_func = str(wf_raw).lower() if wf_raw else 'square'
        
        if 'gauss' in wf_func:
            sigma = pulse_info.get('sigma', duration / 4)
            if sigma is None or sigma == 0: 
                sigma = 1e-12
            t_mid = duration / 2
            envelope = amp * np.exp(-(t_rel - t_mid)**2 / (2 * sigma**2))
        elif 'drag' in wf_func:
            sigma = pulse_info.get('sigma', duration / 4)
            if sigma is None or sigma == 0: 
                sigma = 1e-12
            beta = pulse_info.get('beta', 0.0)
            t_mid = duration / 2
            envelope = amp * np.exp(-(t_rel - t_mid)**2 / (2 * sigma**2))
            envelope_dot = -(t_rel - t_mid) / (sigma**2) * envelope
            return (envelope + 1j * (-beta * envelope_dot / (2 * np.pi))) * np.exp(1j * phase_rad)
        else:
            # Broadcast the scalar amplitude across the entire time array
            envelope = np.full_like(t_rel, amp, dtype=complex)
        
        return envelope * np.exp(1j * phase_rad)

    def simulate(self, schedule: Schedule, initial_state: typing.Optional[qutip.Qobj] = None):
        timing_table, raw_operations_dict = self._get_compiled_schedule(schedule)
        operations_dict = self._flatten_operations(raw_operations_dict)
        
        pulses = timing_table[timing_table['is_acquisition'] == False].copy()
        acquisitions = timing_table[timing_table['is_acquisition'] == True]
        
        # Robust amplitude extraction
        amps = []
        phases = []
        durations = []
        wfs = []
        for _, row in pulses.iterrows():
            op_hash = row['operation_hash']
            op = self._get_op_from_hash(op_hash, operations_dict)
            # Handle both quantify-style and older/newer qblox-style
            data = getattr(op, 'data', op) if hasattr(op, 'data') else op
            
            a = self._extract_amplitude(row)
            p = 0.0
            dur = row['duration']
            wf = 'square'
            
            # 1. Check top-level attributes (sometimes present in row)
            if 'amp' in row and not pd.isna(row['amp']): a = row['amp']
            elif 'amplitude' in row and not pd.isna(row['amplitude']): a = row['amplitude']
            
            # 2. Check operation data
            p_info_list = data.get('pulse_info', [])
            if isinstance(p_info_list, dict):
                p_info_list = [p_info_list]
                
            if isinstance(p_info_list, list) and len(p_info_list) > 0:
                # We need to find the right pulse_info. 
                # If there's only one, it's easy.
                p_info = p_info_list[0] 
                if isinstance(p_info, dict):
                    if a == 0:
                        a = self._extract_amplitude(p_info)
                    p = p_info.get('phase', 0.0)
                    dur = p_info.get('duration', dur)
                    wf = p_info.get('wf_func', 'square')
            
            amps.append(a)
            phases.append(p)
            durations.append(dur)
            wfs.append(wf)
            
        pulses['amplitude'] = amps
        pulses['amp'] = amps
        pulses['phase'] = phases
        pulses['duration'] = durations
        pulses['wf_func'] = wfs

        return self._simulate_processed(pulses, acquisitions, operations_dict=operations_dict, initial_state=initial_state)

    def _simulate_processed(self, pulses, acquisitions, operations_dict=None, initial_state=None):
        operations_dict = operations_dict or {}

        # 1. Normalize pulse time units to seconds
        pulses_list = pulses.to_dict('records')
        for p in pulses_list:
            p['abs_time'] = p['abs_time'] * 1e-9 if p['abs_time'] > 1e-3 else p['abs_time']
            p['duration'] = p['duration'] * 1e-9 if p['duration'] > 1e-3 else p['duration']

        # 2. Determine total simulation duration
        if len(pulses_list) == 0:
            total_duration = 1e-6
        else:
            total_duration = max(p['abs_time'] + p['duration'] for p in pulses_list)

        if len(acquisitions) > 0:
            acq_max = acquisitions['abs_time'].max() + acquisitions['duration'].max()
            acq_max = acq_max * 1e-9 if acq_max > 1e-3 else acq_max
            total_duration = max(total_duration, acq_max)

# 3. Create a STRICTLY UNIFORM time grid (Configurable resolution)
        # Check params for a custom dt, otherwise default to 1 ns for better performance
        step_size = self.cfg.dt 
        num_points = max(1000, int(np.ceil(total_duration / step_size)) + 1)
        t_list = np.linspace(0, total_duration, num_points)
        dt_actual = t_list[1] - t_list[0] if len(t_list) > 1 else step_size

        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(self.N_q, 0), qutip.basis(self.N_res, 0))
            
        # 4 & 5. Pre-sample drive signals into 1D NumPy arrays using vectorized slicing
        q_drive = np.zeros_like(t_list, dtype=complex)
        res_drive = np.zeros_like(t_list, dtype=complex)
        t_start_grid = t_list[0]

        for p in pulses_list:
            port = p.get('port')
            t_start = p['abs_time']
            duration = p['duration']
            t_end = t_start + duration
            
            # Calculate array index slice matching this pulse window
            idx_start = max(0, int(np.floor((t_start - t_start_grid) / dt_actual)))
            idx_end = min(len(t_list), int(np.ceil((t_end - t_start_grid) / dt_actual)) + 1)
            
            if idx_start >= len(t_list) or idx_end <= 0:
                continue
                
            # Get the actual time values for this slice and create a precise mask
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

        qubit_drive_i = np.real(q_drive)
        qubit_drive_q = np.imag(q_drive)
        res_drive_i = np.real(res_drive)
        res_drive_q = np.imag(res_drive)

        # (Replace everything from omega_q = ... down to the end of c_ops = [...])
        omega_q = 2 * np.pi * self.cfg.qubit.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.cfg.resonator.rabi_freq_res_per_volt
        
        h_static = self.system.build_static_hamiltonian()
        c_ops = self.system.build_collapse_operators()

        h = [
            h_static,
            [(self.system.b + self.system.bd) * (omega_q / 2), qubit_drive_i],
            [1j * (self.system.bd - self.system.b) * (omega_q / 2), qubit_drive_q],
            [(self.system.a + self.system.ad) * (omega_res / 2), res_drive_i],
            [1j * (self.system.ad - self.system.a) * (omega_res / 2), res_drive_q]
        ]
        
        # 6. Execute solver
        options = {"nsteps": 500000}
        result = qutip.mesolve(h, initial_state, t_list, c_ops=c_ops, options=options)

        # 7. Measurement extraction
        measurements = []
        for _, acq in acquisitions.iterrows():
            acq_time = acq['abs_time']
            acq_time = acq_time * 1e-9 if acq_time > 1e-3 else acq_time
            acq_duration = acq.get('duration', 1e-6)
            acq_duration = acq_duration * 1e-9 if acq_duration > 1e-3 else acq_duration

            op_hash = acq.get('operation_hash', None)
            op = self._get_op_from_hash(op_hash, operations_dict) if op_hash else {}
            data = getattr(op, 'data', op) if hasattr(op, 'data') else op
            
            acq_info_list = data.get('acquisition_info', []) if isinstance(data, dict) else []
            if isinstance(acq_info_list, dict):
                acq_info_list = [acq_info_list]
            acq_info = acq_info_list[0] if (isinstance(acq_info_list, list) and len(acq_info_list) > 0) else {}

            protocol = acq_info.get('protocol', acq.get('acq_protocol', acq.get('protocol', 'SSBIntegrationComplex')))
            if protocol is None or (isinstance(protocol, float) and np.isnan(protocol)):
                protocol = 'SSBIntegrationComplex'

            idx = np.argmin(np.abs(t_list - acq_time))
            state = result.states[idx]
            rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)

            prob_0 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 0)), rho_q)))
            prob_1 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 1)), rho_q)))
            prob_2 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 2)), rho_q))) if self.N_q >= 3 else 0.0

            if protocol in ('SSBIntegrationComplex', 'SSBIntegration', 'Integration'):
                val, I_val, Q_val = self._process_ssb_integration(rho_q, acq_info)
                outcome = 1 if np.random.random() < prob_1 else 0
            elif protocol in ('ThresholdedAcquisition', 'Thresholded'):
                outcome = self._process_thresholded(rho_q, acq_info)
                val = outcome
                _, I_val, Q_val = self._process_ssb_integration(rho_q, acq_info)
            elif protocol in ('Trace', 'TraceAcquisition'):
                val = self._process_trace(t_list, result.states, acq_time, acq_duration, acq_info)
                I_val = float(np.mean(val)) if len(val) > 0 else 0.0
                Q_val = 0.0
                outcome = 1 if np.random.random() < prob_1 else 0
            else:
                val, I_val, Q_val = self._process_ssb_integration(rho_q, acq_info)
                outcome = 1 if np.random.random() < prob_1 else 0

            acq_channel = acq_info.get('acq_channel', acq.get('acq_channel', acq.get('acq_index', 'acq')))

            measurements.append({
                'name': acq_channel,
                'protocol': protocol,
                'time': acq_time,
                'duration': acq_duration,
                'prob_0': prob_0,
                'prob_1': prob_1,
                'leakage_prob_2': prob_2,
                'outcome': outcome,
                'I': I_val,
                'Q': Q_val,
                'value': val
            })

        result_container = SimulationResult(
            states=result.states if result is not None else [],
            t_list=t_list,
            measurements=measurements,
            simulator=self
        )
            
        return {
            'result': result_container, 
            't_list': t_list, 
            'measurements': measurements
        }


class QbloxLoopSimulator(QbloxQutipSimulator):
    """
    An extended simulator that adds support for Qblox-Scheduler loops.
    It unrolls the loops and resolves variable parameters before simulation.
    """

    def _flatten_operations(self, operations_dict: dict, all_ops: typing.Optional[dict] = None) -> dict:
        if all_ops is None:
            all_ops = {}
        for h, op in operations_dict.items():
            all_ops[h] = op
            if isinstance(op, LoopOperation):
                self._flatten_operations(op.body.operations, all_ops) # type: ignore
            elif hasattr(op, 'operations'): # Nested Schedule
                self._flatten_operations(op.operations, all_ops)
        return all_ops

    def _find_loops(self, operations_dict: dict, offset: float = 0.0) -> list:
        loops = []
        for h, op in operations_dict.items():
            if isinstance(op, LoopOperation):
                cf_info = op.data.get('control_flow_info', {})
                t0 = cf_info.get('t0', 0.0)
                repetitions = cf_info.get('repetitions', 1)
                duration = repetitions * op.body.duration
                loops.append({
                    'op': op,
                    't_start': offset + t0,
                    't_end': offset + t0 + duration,
                    'iteration_duration': op.body.duration,
                    'domain': cf_info.get('domain', {})
                })
                # Recursively find nested loops in body
                loops.extend(self._find_loops(op.body.operations, offset + t0)) #type: ignore
            elif hasattr(op, 'operations'):
                t0 = op.data.get('t0', 0.0) if hasattr(op, 'data') else 0.0
                loops.extend(self._find_loops(op.operations, offset + t0))
        return loops

    def _resolve_value(self, val, mapping: dict):
        if hasattr(val, 'substitute'):
            return val.substitute(mapping)
        if hasattr(val, 'name') and val.name in mapping:
            return mapping[val.name]
        return val

    def simulate(self, schedule: Schedule, initial_state: typing.Optional[qutip.Qobj] = None) -> dict:
        uncompiled_ops = self._flatten_operations(schedule.operations)
        timing_table, operations_dict = self._get_compiled_schedule(schedule)
        loops = self._find_loops(operations_dict)

        pulses = timing_table[timing_table['is_acquisition'] == False].copy()
        acquisitions = timing_table[timing_table['is_acquisition'] == True].copy()

        
        pulses = self._resolve_loop_pulses(pulses, uncompiled_ops, loops)
        # 2. Route to shot sweep or single execution
        if loops:
            return self._simulate_shot_sweep(pulses, acquisitions, loops, uncompiled_ops, initial_state)
        return self._simulate_processed(pulses, acquisitions, uncompiled_ops, initial_state)

    def _resolve_loop_pulses(self, pulses: pd.DataFrame, uncompiled_ops: dict, loops: list) -> pd.DataFrame:
        """Resolve symbolic Variable parameters (amp, phase, duration) for every pulse row."""
        resolved_amps, resolved_phases, resolved_durations, resolved_wfs = [], [], [], []

        for _, row in pulses.iterrows():
            t = row['abs_time']
            op_hash = row['operation_hash']
            op = self._get_op_from_hash(op_hash, uncompiled_ops)
            data = getattr(op, 'data', op) if hasattr(op, 'data') else op

            mapping = {}
            for l in loops:
                if l['t_start'] <= t < l['t_end'] + 1e-15:
                    it_idx = int((t - l['t_start']) // l['iteration_duration'])
                    repetitions = l['op'].data.get('control_flow_info', {}).get('repetitions', 1)
                    it_idx = min(it_idx, repetitions - 1)
                    
                    for var, domain in l['domain'].items():
                        val = domain.start + it_idx * (domain.stop - domain.start) / (domain.num - 1) if domain.num > 1 else domain.start
                        mapping[var] = val
                        if hasattr(var, 'name'): 
                            mapping[var.name] = val

            a, p, dur, wf = 0.0, 0.0, row['duration'], 'square'
            p_info_list = data.get('pulse_info', [])
            if isinstance(p_info_list, dict): 
                p_info_list = [p_info_list]
                
            if p_info_list:
                p_info = p_info_list[0]
                raw_amp = self._extract_amplitude(p_info)
                a = self._resolve_value(raw_amp, mapping)
                p = self._resolve_value(p_info.get('phase', 0.0), mapping)
                dur = self._resolve_value(p_info.get('duration', row['duration']), mapping)
                wf = p_info.get('wf_func', 'square')

            resolved_amps.append(a)
            resolved_phases.append(p)
            resolved_durations.append(dur)
            resolved_wfs.append(wf)

        pulses['amplitude'] = resolved_amps
        pulses['amp'] = resolved_amps
        pulses['phase'] = resolved_phases
        pulses['duration'] = resolved_durations
        pulses['wf_func'] = resolved_wfs
        return pulses

    def _simulate_shot_sweep(
        self, 
        pulses: pd.DataFrame, 
        acquisitions: pd.DataFrame, 
        loops: list, 
        operations_dict: dict,
        initial_state: typing.Optional[qutip.Qobj]
    ) -> dict:
        loop_start = loops[0]['t_start']
        loop_duration = loops[0]['iteration_duration']
        num_iterations = loops[0]['op'].data.get('control_flow_info', {}).get('repetitions', 1)

        all_results = []
        combined_t_list = []
        combined_states = []

        for it in range(num_iterations):
            it_start = loop_start + it * loop_duration
            it_end = loop_start + (it + 1) * loop_duration

            it_pulses = pulses[(pulses['abs_time'] >= it_start - 1e-12) & (pulses['abs_time'] < it_end - 1e-12)].copy()
            it_pulses['abs_time'] -= it_start

            it_acq = acquisitions[(acquisitions['abs_time'] >= it_start - 1e-12) & (acquisitions['abs_time'] < it_end - 1e-12)].copy()
            if len(it_acq) > 0:
                it_acq['abs_time'] -= it_start

            res = self._simulate_processed(it_pulses, it_acq, operations_dict, initial_state=initial_state)
            all_results.append(res)

            is_last = (it == num_iterations - 1)
            t_slice = res['t_list'] if is_last else res['t_list'][:-1]
            state_slice = res['result'].states if is_last else res['result'].states[:-1]

            combined_t_list.append(t_slice + it_start)
            combined_states.extend(state_slice)

        final_t_list = np.concatenate(combined_t_list)
        result_container = SimulationResult(
            combined_states, 
            final_t_list, 
            [m for r in all_results for m in r['measurements']], 
            simulator=self
        )

        return {
            'result': result_container,
            't_list': final_t_list,
            'measurements': result_container.measurements
        }

from q1simulator import Cluster

import typing
import numpy as np
import qutip
from qcodes.instrument import Instrument
from q1simulator import Cluster
from qblox_scheduler import Schedule

class QbloxQ1Simulator(QbloxQutipSimulator):
    """
    A simulator that takes a Q1Simulator object (https://github.com/sldesnoo-Delft/q1simulator/tree/main) 
    and uses QuTiP to simulate the dynamics of a qubit coupled to a readout resonator.
    """

    def __init__(
        self, 
        params: dict, 
        name: str = 'cluster', 
        modules: typing.Optional[dict] = None, 
        hardware_config: typing.Optional[dict] = None
    ):
        super().__init__(params)

        hardware_config = hardware_config or {}
        drive_config = hardware_config.get('drive', {})
        readout_config = hardware_config.get('readout', {})
        
        self.drive_mod = drive_config.get('module', 2)
        self.drive_seq = drive_config.get('sequencer', 0)
        self.readout_mod = readout_config.get('module', 4)
        self.readout_seq = readout_config.get('sequencer', 0)

        # 1. FIX: Default to drive and readout modules if none provided
        if modules is None:
            modules = {self.drive_mod: 'QCM-RF', self.readout_mod: 'QRM-RF'}

        # 2. FIX: Safely replace existing instrument in QCoDeS registry if name collides
        try:
            if Instrument.exist(name):
                Instrument.find_instrument(name).close()
        except Exception:
            pass

        self.cluster: typing.Optional[Cluster] = Cluster(name=name, modules=modules)

        self.t_max = 0.0
        self.t_min = 0.0
        self.t_sample = 0.0
        self.t_list: np.ndarray = np.array([], dtype=float)

    def close(self):
        """Safely close the underlying QCoDeS cluster instrument."""
        if hasattr(self, 'cluster') and self.cluster is not None:
            try:
                self.cluster.close()
            except Exception:
                pass
            self.cluster = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def simulate(
        self, 
        schedule: typing.Optional[Schedule] = None, 
        initial_state: typing.Optional[qutip.Qobj] = None
    ) -> dict:
        try:
            drive_pulses, readout_pulses = self.get_pulses()
        except Exception as e:
            print("Error extracting pulses from Q1Simulator:", e)
            drive_pulses, readout_pulses = {}, {}
        
        if self.t_max == 0:
            print("Warning: t_max is 0, using default time range (500 ns)")
            self.t_max = 500
            self.t_sample = 1
        
        num_points = int(round(self.t_max / self.t_sample)) if self.t_sample > 0 else 500
        self.t_list = np.linspace(0, self.t_max * 1e-9, num_points)

        # STRANGLER FIX: Use config dimensions
        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(self.cfg.qubit.N_q, 0), qutip.basis(self.cfg.resonator.N_res, 0))

        # --- PHASE 2 PHYSICS REPLACEMENT ---
        omega_q = 2 * np.pi * self.cfg.qubit.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.cfg.resonator.rabi_freq_res_per_volt

        h_static = self.system.build_static_hamiltonian()
        c_ops = self.system.build_collapse_operators()
        # -----------------------------------

        def safe_data(pulse_dict, key):
            arr = np.array(pulse_dict.get(key, {}).get("data", []))
            if len(arr) != len(self.t_list):
                if len(arr) > 0:
                    old_t = np.linspace(0, self.t_max * 1e-9, len(arr))
                    arr = np.interp(self.t_list, old_t, arr)
                else:
                    arr = np.zeros_like(self.t_list)
            return arr

        drive_I = safe_data(drive_pulses, "I")
        drive_Q = safe_data(drive_pulses, "Q")
        readout_I = safe_data(readout_pulses, "I")
        readout_Q = safe_data(readout_pulses, "Q")

        h = [
            h_static,
            [(self.system.b + self.system.bd) * (omega_q / 2), drive_I],
            [1j * (self.system.bd - self.system.b) * (omega_q / 2), drive_Q],
            [(self.system.a + self.system.ad) * (omega_res / 2), readout_I],
            [1j * (self.system.ad - self.system.a) * (omega_res / 2), readout_Q]
        ]

        options = {
            "nsteps": 100000,
            "max_step": 1e-9,
            "rtol": 1e-6,
            "atol": 1e-8,
            "method": "bdf"
        }
        
        result = None
        try:
            print(f"Running QuTiP mesolve ({len(self.t_list)} time points)...")
            result = qutip.mesolve(h, initial_state, self.t_list, c_ops=c_ops, options=options) 
            print("✓ Simulation completed successfully")
        except Exception as e:
            print(f"✗ Solver failed: {e}")
        
        measurements = self.get_measurements(result)
        self.set_acquisition_mock_data(measurements)

        return {'result': result, 't_list': self.t_list, 'measurements': measurements}
    
    def get_pulses(self):
        drive_pulses = {}
        readout_pulses = {}

        # 3. FIX: Pylance guard
        if self.cluster is None:
            return drive_pulses, readout_pulses

        connected = self.cluster.get_connected_modules()
        if self.drive_mod not in connected or self.readout_mod not in connected:
            return drive_pulses, readout_pulses

        qcm_outputs = connected[self.drive_mod].get_output()
        qrm_outputs = connected[self.readout_mod].get_output()

        for path in qcm_outputs.keys():
            if not self.t_max:
                self.t_max = qcm_outputs[path].t_max
                self.t_min = qcm_outputs[path].t_min
                self.t_sample = qcm_outputs[path].sample_rate

            if path.startswith(f'sequencer{self.drive_seq}'):
                drive_pulses[path[-1]] = {"data": qcm_outputs[path].data}
        for path in qrm_outputs.keys():
            if path.startswith(f'sequencer{self.readout_seq}'):
                readout_pulses[path[-1]] = {"data": qrm_outputs[path].data}

        return drive_pulses, readout_pulses

    def get_measurements(self, result: typing.Any) -> list:
        if result is None or not hasattr(result, 'states') or len(getattr(result, 'states', [])) == 0:
            return []

        if self.cluster is None:
            return []

        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod not in connected:
                return []
            
            qrm = connected[self.readout_mod]
            acq_windows = qrm.get_acquisition_windows()

            # Fallback: if acq_windows is empty, check for manually attached test windows on the sequencer
            if not acq_windows or not any(acq_windows.values()):
                if hasattr(qrm, 'sequencers') and len(qrm.sequencers) > self.readout_seq:
                    seq_obj = qrm.sequencers[self.readout_seq]
                    if hasattr(seq_obj, 'acq_windows'):
                        acq_windows = {self.readout_seq: getattr(seq_obj, 'acq_windows')}

        except Exception as e:
            print(f"Warning: Could not get acquisition windows: {e}")
            return []

        # Access windows flexibly by readout sequencer index
        windows = (
            acq_windows.get(self.readout_seq)
            or acq_windows.get(str(self.readout_seq))
            or acq_windows.get(f"sequencer{self.readout_seq}")
            or acq_windows.get(f"seq{self.readout_seq}")
        )

        if not windows:
            return []

        sigma = 0.02
        v_0 = complex(0.05, 0.05)
        v_1 = complex(-0.05, -0.05)

        measurements = []
        for acq in windows:
            try:
                # Handle [(start, dur)], [start, dur], or (start, dur)
                if isinstance(acq, (list, tuple)) and len(acq) > 0 and isinstance(acq[0], (list, tuple)):
                    acq_start_ns, duration_ns = acq[0][0], acq[0][1]
                elif isinstance(acq, (list, tuple)) and len(acq) >= 2:
                    acq_start_ns, duration_ns = acq[0], acq[1]
                else:
                    continue

                acq_start_sec = float(acq_start_ns) * 1e-9
                acq_duration_sec = float(duration_ns) * 1e-9

                # 1. Generate physical 1D time-series trace using QuTiP states
                trace_data = self._process_trace(
                    t_list=self.t_list,
                    states=result.states,
                    acq_start_sec=acq_start_sec,
                    acq_duration=acq_duration_sec,
                    acq_info={'acq_delay': getattr(self, 'params', {}).get('cable_delay', 120e-9)}
                )

                # 2. Compute scalar I/Q centroids and probabilities
                idx = np.argmin(np.abs(self.t_list - acq_start_sec))
                state = result.states[idx]

                rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
                
                prob_0 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 0)), rho_q)))
                prob_1 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 1)), rho_q)))
                prob_2 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 2)), rho_q))) if self.N_q >= 3 else 0.0

                centroid = prob_0 * v_0 + prob_1 * v_1
                I_val = float(np.real(centroid) + np.random.normal(0, sigma))
                Q_val = float(np.imag(centroid) + np.random.normal(0, sigma))

                measurements.append({
                    'time': acq_start_sec,
                    'duration': acq_duration_sec,
                    'prob_0': prob_0,
                    'prob_1': prob_1,
                    'leakage_prob_2': prob_2,
                    'outcome': 1 if np.random.random() < prob_1 else 0,
                    'I': I_val,
                    'Q': Q_val,
                    'trace': trace_data,
                    'trace_I': np.real(trace_data),
                    'trace_Q': np.imag(trace_data)
                })
            except Exception as e:
                print(f"Warning: Could not process acquisition window: {e}")
                continue

        return measurements

    def set_acquisition_mock_data(self, measurements: list):
        """Pass mock acquisition data back to hardware module if present."""
        if self.cluster is None or not measurements:
            return

        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod in connected:
                qrm = connected[self.readout_mod]
                if 'I' in measurements[0] and 'Q' in measurements[0]:
                    I_arr = np.array([m['I'] for m in measurements])
                    Q_arr = np.array([m['Q'] for m in measurements])
                    qrm.sequencers[self.readout_seq].set_acquisition_mock_data(I_arr + 1j * Q_arr)
        except Exception as e:
            print(f"Warning: Could not set acquisition mock data: {e}")    