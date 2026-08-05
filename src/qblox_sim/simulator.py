from unittest import result
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule, SerialCompiler, QuantumDevice, BasicTransmonElement
from qblox_scheduler.operations import SquarePulse, GaussPulse, DRAGPulse, LoopOperation
import typing

class QbloxQutipSimulator:
    # ... (existing methods)
    """
    A simulator that takes a Qblox-Scheduler Schedule and uses QuTiP to simulate 
    the dynamics of a qubit coupled to a readout resonator.
    """

    def __init__(self, params: dict, configs: typing.Optional[dict] = None):
        self.params = params
        self.configs = configs
        
        # System Frequencies & Parameters
        self.f_q = params.get('f_q', 5.0e9)
        self.f_d = params.get('f_d', self.f_q)
        self.f_res = params.get('f_res', 6.0e9)
        self.f_d_res = params.get('f_d_res', self.f_res)
        self.chi = params.get('chi', 1.0e6)
        
        # Transmon Hilbert Space Dimensions & Anharmonicity
        self.N_q = params.get('N_q', 3)          # Default to 3 levels for transmon
        self.alpha = params.get('alpha', -300.0e6) # Anharmonicity in Hz
        
        self.rabi_freq_per_volt = params.get('rabi_freq_per_volt', 10.0e6)
        self.rabi_freq_res_per_volt = params.get('rabi_freq_res_per_volt', 10.0e6)
        self.T1 = params.get('T1', np.inf)
        self.T2 = params.get('T2', np.inf)
        self.kappa = params.get('kappa', 1e6)
        self.N_res = params.get('N_res', 5)

        # -------------------------------------------------------------
        # Transmon & Readout Operators (Hilbert Space: N_q x N_res)
        # -------------------------------------------------------------
        
        # Transmon ladder / lowering operator
        self.b = qutip.tensor(qutip.destroy(self.N_q), qutip.identity(self.N_res))
        self.bd = self.b.dag()
        # Qubit number operator (b^\dagger b)
        self.nq = self.bd * self.b  #type: ignore

        # Readout Cavity operators
        self.a = qutip.tensor(qutip.identity(self.N_q), qutip.destroy(self.N_res))
        self.ad = self.a.dag()
        self.n = self.ad * self.a #type: ignore

        # -------------------------------------------------------------
        # Backward-Compatible Pauli / Subspace Operators
        # -------------------------------------------------------------
        # Lowering operator (takes |1> -> |0> and |2> -> |1>)
        self.sm = self.b
        
        if self.N_q == 2:
            # Exact 2-level Pauli matrices
            self.sx = qutip.tensor(qutip.sigmax(), qutip.identity(self.N_res))
            self.sy = qutip.tensor(qutip.sigmay(), qutip.identity(self.N_res))
            self.sz = qutip.tensor(qutip.sigmaz(), qutip.identity(self.N_res))
        else:
            # Generalized drive quadratures for N_q >= 3
            self.sx = self.b + self.bd
            self.sy = 1j * (self.bd - self.b)
            
            # Generalized sz projecting onto {|0>, |1>} computational subspace
            proj_0 = qutip.tensor(qutip.basis(self.N_q, 0) * qutip.basis(self.N_q, 0).dag(), qutip.identity(self.N_res)) #type: ignore
            proj_1 = qutip.tensor(qutip.basis(self.N_q, 1) * qutip.basis(self.N_q, 1).dag(), qutip.identity(self.N_res)) #type: ignore
            self.sz = proj_0 - proj_1

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
                q0 = BasicTransmonElement("q0")
                device.add_element(q0)
            except Exception:
                pass
            compiler = SerialCompiler(name="compiler", quantum_device=device)
            compiled_sched = compiler.compile(schedule)
            return compiled_sched.timing_table.data, compiled_sched.operations # type: ignore[attr-defined]

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

        return self._simulate_processed(pulses, acquisitions, initial_state)

    def _simulate_processed(self, pulses, acquisitions, initial_state=None):
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

        # 3. Create a STRICTLY UNIFORM time grid (0.1 ns resolution)
        step_size = 0.1e-9  # 0.1 ns step
        num_points = max(1000, int(np.ceil(total_duration / step_size)) + 1)
        t_list = np.linspace(0, total_duration, num_points)

        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(self.N_q, 0), qutip.basis(self.N_res, 0))
            
        # 4. Helper to evaluate drive envelope at time t
        def get_drive(t, port_name, pulses_list):
            val = 0.0j
            eps = 1e-13
            for p in pulses_list:
                if p['port'] == port_name:
                    t_start = p['abs_time']
                    duration = p['duration']
                    if t_start - eps <= t <= t_start + duration + eps:
                        val += self._pulse_envelope(t, p)
            return val
        
        # 5. Pre-sample drive signals into 1D NumPy arrays matching t_list exactly
        q_drive = np.array([get_drive(t, 'q0:mw', pulses_list) for t in t_list])
        res_drive = np.array([get_drive(t, 'q0:res', pulses_list) for t in t_list])

        qubit_drive_i = np.real(q_drive)
        qubit_drive_q = np.imag(q_drive)
        res_drive_i = np.real(res_drive)
        res_drive_q = np.imag(res_drive)

        omega_q = 2 * np.pi * self.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.rabi_freq_res_per_volt
        
        delta_q = 2 * np.pi * (self.f_q - self.f_d)
        delta_res = 2 * np.pi * (self.f_res - self.f_d_res)
        
        # --- TRANSMON HAMILTONIAN WITH ANHARMONICITY ---
        h_static = (
            -delta_q * self.nq 
            + np.pi * self.alpha * (self.bd * self.bd * self.b * self.b) #type: ignore
            + delta_res * self.n 
            + 2 * np.pi * self.chi * self.n * self.nq
        )

        h = [
            h_static,
            [(self.b + self.bd) * (omega_q / 2), qubit_drive_i],
            [1j * (self.bd - self.b) * (omega_q / 2), qubit_drive_q],
            [(self.a + self.ad) * (omega_res / 2), res_drive_i],
            [1j * (self.ad - self.a) * (omega_res / 2), res_drive_q]
        ]
        
        c_ops = []
        if self.T1 < np.inf: 
            c_ops.append(np.sqrt(1.0 / self.T1) * self.b)
            
        if self.T2 < np.inf:
            gamma_phi = (1.0 / self.T2) - (0.5 / self.T1 if self.T1 < np.inf else 0)
            if gamma_phi > 0: 
                c_ops.append(np.sqrt(2 * gamma_phi) * self.nq)
                
        if self.kappa > 0: 
            c_ops.append(np.sqrt(self.kappa) * self.a)

        # 6. Execute solver
        options = {"nsteps": 500000}
        result = qutip.mesolve(h, initial_state, t_list, c_ops=c_ops, options=options)

        # 7. Measurement extraction
        measurements = []
        for _, acq in acquisitions.iterrows():
            acq_time = acq['abs_time']
            acq_time = acq_time * 1e-9 if acq_time > 1e-3 else acq_time
            idx = np.argmin(np.abs(t_list - acq_time))
            state = result.states[idx]
            
            rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
            
            prob_0 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 0)), rho_q))
            prob_1 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 1)), rho_q))
            prob_2 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 2)), rho_q)) if self.N_q >= 3 else 0.0
            
            measurements.append({
                'name': acq.get('acq_index', acq.get('acq_channel', 'acq')),
                'time': acq_time, 
                'prob_0': prob_0,
                'prob_1': prob_1,
                'leakage_prob_2': prob_2,
                'outcome': 1 if np.random.random() < prob_1 else 0
            })

        # 8. Wrap output in SimulationResult container
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

        # 1. Unroll symbolic loop variables onto pulse dataframe
        pulses = self._resolve_loop_pulses(pulses, uncompiled_ops, loops)

        # 2. Route to shot sweep or single execution
        if loops:
            return self._simulate_shot_sweep(pulses, acquisitions, loops, initial_state)
        return self._simulate_processed(pulses, acquisitions, initial_state)

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
        initial_state: typing.Optional[qutip.Qobj]
    ) -> dict:
        """Simulate each loop iteration as an independent experiment shot re-initialized from initial_state."""
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

            res = self._simulate_processed(it_pulses, it_acq, initial_state=initial_state)
            all_results.append(res)

            # Deduplicate boundary points: drop final point of all iterations except the last
            is_last = (it == num_iterations - 1)
            t_slice = res['t_list'] if is_last else res['t_list'][:-1]
            state_slice = res['result'].states if is_last else res['result'].states[:-1]

            combined_t_list.append(t_slice + it_start)
            combined_states.extend(state_slice)

        final_t_list = np.concatenate(combined_t_list)
        result_container = SimulationResult(combined_states, final_t_list, [m for r in all_results for m in r['measurements']], simulator=self)

        return {
            'result': result_container,
            't_list': final_t_list,
            'measurements': result_container.measurements
        }

from q1simulator import Cluster

class QbloxQ1Simulator:
    # ... (existing methods)
    """
    A simulator that takes a Q1Simulator object (https://github.com/sldesnoo-Delft/q1simulator/tree/main) and uses QuTiP to simulate 
    the dynamics of a qubit coupled to a readout resonator.
    """

    def __init__(self, params: dict, name: str = 'cluster', modules: typing.Optional[dict] = None, hardware_config: typing.Optional[dict] = None):
        self.cluster = Cluster(name=name, modules=modules)

        hardware_config = hardware_config or {}
        drive_config = hardware_config.get('drive', {})
        readout_config = hardware_config.get('readout', {})
        
        self.drive_mod = drive_config.get('module', 2)
        self.drive_seq = drive_config.get('sequencer', 0)
        self.readout_mod = readout_config.get('module', 4)
        self.readout_seq = readout_config.get('sequencer', 0)

        self.t_max = 0
        self.t_min = 0
        self.t_sample = 0
        self.t_list = []

        self.params = params

        # System Frequencies & Coupling Parameters
        self.f_q = params.get('f_q', 5.0e9)
        self.f_d = params.get('f_d', self.f_q)
        self.f_res = params.get('f_res', 6.0e9)
        self.f_d_res = params.get('f_d_res', self.f_res)
        self.chi = params.get('chi', 1.0e6)
        
        # Transmon Hilbert Space & Anharmonicity
        self.N_q = params.get('N_q', 3)
        self.alpha = params.get('alpha', -300.0e6)
        
        self.rabi_freq_per_volt = params.get('rabi_freq_per_volt', 10.0e6)
        self.rabi_freq_res_per_volt = params.get('rabi_freq_res_per_volt', 10.0e6)
        self.T1 = params.get('T1', np.inf)
        self.T2 = params.get('T2', np.inf)
        self.kappa = params.get('kappa', 1e6)
        self.N_res = params.get('N_res', 5)

        # -------------------------------------------------------------
        # Transmon & Readout Operators (Hilbert Space: N_q x N_res)
        # -------------------------------------------------------------
        self.b = qutip.tensor(qutip.destroy(self.N_q), qutip.identity(self.N_res))
        self.bd = self.b.dag()
        self.nq = self.bd * self.b #type: ignore
 
        self.a = qutip.tensor(qutip.identity(self.N_q), qutip.destroy(self.N_res))
        self.ad = self.a.dag()
        self.n = self.ad * self.a #type: ignore

        # -------------------------------------------------------------
        # Backward-Compatible Pauli / Subspace Operators
        # -------------------------------------------------------------
        self.sm = self.b
        
        if self.N_q == 2:
            self.sx = qutip.tensor(qutip.sigmax(), qutip.identity(self.N_res))
            self.sy = qutip.tensor(qutip.sigmay(), qutip.identity(self.N_res))
            self.sz = qutip.tensor(qutip.sigmaz(), qutip.identity(self.N_res))
        else:
            self.sx = self.b + self.bd
            self.sy = 1j * (self.bd - self.b)
            
            proj_0 = qutip.tensor(qutip.basis(self.N_q, 0) * qutip.basis(self.N_q, 0).dag(), qutip.identity(self.N_res)) #type: ignore
            proj_1 = qutip.tensor(qutip.basis(self.N_q, 1) * qutip.basis(self.N_q, 1).dag(), qutip.identity(self.N_res)) #type: ignore
            self.sz = proj_0 - proj_1

    def simulate(self, initial_state: typing.Optional[qutip.Qobj] = None):
        try:
            drive_pulses, readout_pulses = self.get_pulses()
        except Exception as e:
            print("Error extracting pulses from Q1Simulator:", e)
            drive_pulses, readout_pulses = {}, {}
        
        if self.t_max == 0:
            print("Warning: t_max is 0, using default time range (500 ns)")
            self.t_max = 500  # ns
            self.t_sample = 1 # ns
        
        # 1. Time grid strictly in seconds
        num_points = int(round(self.t_max / self.t_sample)) if self.t_sample > 0 else 500
        self.t_list = np.linspace(0, self.t_max * 1e-9, num_points)

        # 2. Correct initial ground state matching Hilbert space (N_q x N_res)
        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(self.N_q, 0), qutip.basis(self.N_res, 0))

        # 3. Frequencies & Detunings (in rad/s)
        omega_q = 2 * np.pi * self.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.rabi_freq_res_per_volt
        delta_q = 2 * np.pi * (self.f_q - self.f_d)
        delta_res = 2 * np.pi * (self.f_res - self.f_d_res)

        # 4. Transmon + Cavity Hamiltonian
        h_static = (
            -delta_q * self.nq 
            + np.pi * self.alpha * (self.bd * self.bd * self.b * self.b) #type: ignore
            + delta_res * self.n 
            + 2 * np.pi * self.chi * self.n * self.nq
        )

        # Helper to align pulse arrays safely with self.t_list length
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

        # 5. Full Time-Dependent Drive Hamiltonian
        h = [
            h_static,
            [(self.b + self.bd) * (omega_q / 2), drive_I],
            [1j * (self.bd - self.b) * (omega_q / 2), drive_Q],
            [(self.a + self.ad) * (omega_res / 2), readout_I],
            [1j * (self.ad - self.a) * (omega_res / 2), readout_Q]
        ]

        # 6. Collapse Operators
        c_ops = []
        if self.T1 < np.inf: 
            c_ops.append(np.sqrt(1.0 / self.T1) * self.b)
            
        if self.T2 < np.inf:
            gamma_phi = (1.0 / self.T2) - (0.5 / self.T1 if self.T1 < np.inf else 0)
            if gamma_phi > 0: 
                c_ops.append(np.sqrt(2 * gamma_phi) * self.nq)
                
        if self.kappa > 0: 
            c_ops.append(np.sqrt(self.kappa) * self.a)

        # 7. Solver configuration
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

        # Pass mock acquisition data back to hardware module if present
        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod in connected:
                qrm = connected[self.readout_mod]
                if measurements and 'I' in measurements[0] and 'Q' in measurements[0]:
                    I_arr = np.array([m['I'] for m in measurements])
                    Q_arr = np.array([m['Q'] for m in measurements])
                    qrm.sequencers[self.readout_seq].set_acquisition_mock_data(I_arr + 1j * Q_arr)
        except Exception as e:
            print(f"Warning: Could not set acquisition mock data: {e}")

        return {'result': result, 't_list': self.t_list, 'measurements': measurements}
    
    def get_pulses(self):
        drive_pulses = {}
        readout_pulses = {}
        qcm_outputs = self.cluster.get_connected_modules()[self.drive_mod].get_output()
        qrm_outputs = self.cluster.get_connected_modules()[self.readout_mod].get_output()

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

    def get_measurements(self, result):
        if result is None or not hasattr(result, 'states') or len(result.states) == 0:
            return []

        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod not in connected:
                return []
            acq_windows = connected[self.readout_mod].get_acquisition_windows()
        except Exception as e:
            print(f"Warning: Could not get acquisition windows: {e}")
            return []

        seq_key = f'sequencer{self.readout_seq}'
        if seq_key not in acq_windows:
            return []

        sigma = 0.02  # Voltage noise standard deviation (V)
        v_0 = complex(0.05, 0.05)   # Ground state |0> IQ centroid (V)
        v_1 = complex(-0.05, -0.05) # Excited state |1> IQ centroid (V)

        measurements = []
        for acq in acq_windows[seq_key]:
            try:
                # Convert acquisition start time from ns to seconds
                acq_start_sec = acq[0][0] * 1e-9
                idx = np.argmin(np.abs(self.t_list - acq_start_sec))
                state = result.states[idx]

                # Qubit reduced density matrix
                rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
                
                prob_0 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 0)), rho_q))
                prob_1 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 1)), rho_q))
                prob_2 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(self.N_q, 2)), rho_q)) if self.N_q >= 3 else 0.0

                # Compute state-dependent voltage centroid with additive Gaussian noise
                centroid = prob_0 * v_0 + prob_1 * v_1
                I_val = float(np.real(centroid) + np.random.normal(0, sigma))
                Q_val = float(np.imag(centroid) + np.random.normal(0, sigma))

                measurements.append({
                    'time': acq_start_sec,
                    'prob_0': prob_0,
                    'prob_1': prob_1,
                    'leakage_prob_2': prob_2,
                    'outcome': 1 if np.random.random() < prob_1 else 0,
                    'I': I_val,
                    'Q': Q_val
                })
            except Exception as e:
                print(f"Warning: Could not process acquisition window: {e}")
                continue

        return measurements
