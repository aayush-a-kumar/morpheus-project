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

    def __init__(self, params: dict, configs: dict = None):
        self.params = params
        self.configs = configs
        self.f_q = params.get('f_q', 5.0e9)
        self.f_d = params.get('f_d', self.f_q)
        self.f_res = params.get('f_res', 6.0e9)
        self.chi = params.get('chi', 1.0e6)
        self.rabi_freq_per_volt = params.get('rabi_freq_per_volt', 10.0e6)
        self.rabi_freq_res_per_volt = params.get('rabi_freq_res_per_volt', 10.0e6)
        self.T1 = params.get('T1', np.inf)
        self.T2 = params.get('T2', np.inf)
        self.kappa = params.get('kappa', 1e6)
        self.N_res = params.get('N_res', 5)

        # Operators in the joint Hilbert space
        self.sx = qutip.tensor(qutip.sigmax(), qutip.identity(self.N_res))
        self.sy = qutip.tensor(qutip.sigmay(), qutip.identity(self.N_res))
        self.sz = qutip.tensor(qutip.sigmaz(), qutip.identity(self.N_res))
        self.sm = qutip.tensor(qutip.sigmam(), qutip.identity(self.N_res))
        
        self.a = qutip.tensor(qutip.identity(2), qutip.destroy(self.N_res))
        self.ad = self.a.dag()
        self.n = self.ad * self.a

    def _get_compiled_schedule(self, schedule: Schedule) -> typing.Tuple[pd.DataFrame, dict]:
        try:
            return schedule.timing_table.data, schedule.operations
        except Exception:
            device = QuantumDevice(name="dummy_device")
            try:
                q0 = BasicTransmonElement("q0")
                device.add_element(q0)
            except Exception:
                pass
            compiler = SerialCompiler(name="compiler", quantum_device=device)
            compiled_sched = compiler.compile(schedule)
            return compiled_sched.timing_table.data, compiled_sched.operations

    def _pulse_envelope(self, t: float, pulse_info: dict) -> complex:
        t_start = pulse_info['abs_time']
        duration = pulse_info['duration']
        t_rel = t - t_start
        
        if t_rel < 0 or t_rel > duration:
            return 0.0j
        
        amp = pulse_info.get('amp', 0.0)
        phase_deg = pulse_info.get('phase', 0.0)
        phase_rad = np.deg2rad(phase_deg)
        
        wf_func = str(pulse_info.get('wf_func', 'square')).lower()
        
        envelope = 0.0
        if 'square' in wf_func:
            envelope = amp
        elif 'gauss' in wf_func:
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
        
        return envelope * np.exp(1j * phase_rad)

    def simulate(self, schedule: Schedule, initial_state: qutip.Qobj = None):
        timing_table, operations_dict = self._get_compiled_schedule(schedule)
        
        pulses = timing_table[timing_table['is_acquisition'] == False].copy()
        acquisitions = timing_table[timing_table['is_acquisition'] == True]
        
        # Robust amplitude extraction
        amps = []
        phases = []
        durations = []
        wfs = []
        for _, row in pulses.iterrows():
            op_hash = row['operation_hash']
            op = operations_dict.get(op_hash, {})
            # Handle both quantify-style and older/newer qblox-style
            data = op.data if hasattr(op, 'data') else op
            
            a = 0.0
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
                        a = p_info.get('amp', p_info.get('amplitude', 0.0))
                    p = p_info.get('phase', 0.0)
                    dur = p_info.get('duration', dur)
                    wf = p_info.get('wf_func', 'square')
            
            amps.append(a)
            phases.append(p)
            durations.append(dur)
            wfs.append(wf)
            
        pulses['amp'] = amps
        pulses['phase'] = phases
        pulses['duration'] = durations
        pulses['wf_func'] = wfs

        return self._simulate_processed(pulses, acquisitions, initial_state)

    def _simulate_processed(self, pulses, acquisitions, initial_state=None):
        if len(pulses) == 0:
            total_duration = 1e-6
        else:
            total_duration = pulses['abs_time'].max() + pulses['duration'].max()
        if len(acquisitions) > 0:
            total_duration = max(total_duration, acquisitions['abs_time'].max() + acquisitions['duration'].max())

        t_list = np.linspace(0, total_duration, max(500, int(total_duration / 1e-9)))
        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(2, 0), qutip.basis(self.N_res, 0))
            
        def get_drive(t, port_name, pulses_list):
            val = 0.0j
            eps = 1e-13
            for p in pulses_list:
                if p['port'] == port_name:
                    if p['abs_time'] - eps <= t <= p['abs_time'] + p['duration'] + eps:
                        val += self._pulse_envelope(t, p)
            return val
        
        pulses_list = pulses.to_dict('records')
        omega_q = 2 * np.pi * self.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.rabi_freq_res_per_volt
        delta = 2 * np.pi * (self.f_q - self.f_d)
        h_static = (delta / 2) * self.sz + 2 * np.pi * self.chi * self.n * self.sz
        
        def qubit_drive_i(t, args): return np.real(get_drive(t, 'q0:mw', args['pulses']))
        def qubit_drive_q(t, args): return np.imag(get_drive(t, 'q0:mw', args['pulses']))
        def res_drive_i(t, args): return np.real(get_drive(t, 'q0:res', args['pulses']))
        def res_drive_q(t, args): return np.imag(get_drive(t, 'q0:res', args['pulses']))
        
        h = [
            h_static,
            [self.sx * (omega_q / 2), qubit_drive_i],
            [self.sy * (omega_q / 2), qubit_drive_q],
            [self.a + self.ad, lambda t, args: (omega_res / 2) * res_drive_i(t, args)],
            [1j * (self.ad - self.a), lambda t, args: (omega_res / 2) * res_drive_q(t, args)]
        ]
        
        c_ops = []
        if self.T1 < np.inf: c_ops.append(np.sqrt(1.0 / self.T1) * self.sm)
        if self.T2 < np.inf:
            gamma_phi = (1.0 / self.T2) - (0.5 / self.T1 if self.T1 < np.inf else 0)
            if gamma_phi > 0: c_ops.append(np.sqrt(2 * gamma_phi) * self.sz / 2.0)
        if self.kappa > 0: c_ops.append(np.sqrt(self.kappa) * self.a)
                
        result = qutip.mesolve(h, initial_state, t_list, c_ops=c_ops, args={'pulses': pulses_list})
        
        measurements = []
        for _, acq in acquisitions.iterrows():
            acq_time = acq['abs_time']
            idx = np.argmin(np.abs(t_list - acq_time))
            state = result.states[idx]
            rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
            prob_1 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(2, 1)), rho_q))
            measurements.append({
                'name': acq.get('acq_index', acq.get('acq_channel', 'acq')),
                'time': acq_time, 'prob_1': prob_1,
                'outcome': 1 if np.random.random() < prob_1 else 0
            })
            
        return {'result': result, 't_list': t_list, 'measurements': measurements}

class QbloxLoopSimulator(QbloxQutipSimulator):
    """
    An extended simulator that adds support for Qblox-Scheduler loops.
    It unrolls the loops and resolves variable parameters before simulation.
    """

    def _flatten_operations(self, operations_dict: dict, all_ops: dict = None) -> dict:
        if all_ops is None:
            all_ops = {}
        for h, op in operations_dict.items():
            all_ops[h] = op
            if isinstance(op, LoopOperation):
                self._flatten_operations(op.body.operations, all_ops)
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
                loops.extend(self._find_loops(op.body.operations, offset + t0))
            elif hasattr(op, 'operations'):
                t0 = op.data.get('t0', 0.0) if hasattr(op, 'data') else 0.0
                loops.extend(self._find_loops(op.operations, offset + t0))
        return loops

    def _resolve_value(self, val, mapping: dict):
        if hasattr(val, 'substitute'):
            return val.substitute(mapping)
        return val

    def simulate(self, schedule: Schedule, initial_state: qutip.Qobj = None):
        timing_table, operations_dict = self._get_compiled_schedule(schedule)
        
        # 1. Flatten operations to find all nested ones
        all_ops = self._flatten_operations(operations_dict)
        
        # 2. Find all loops and their domains
        loops = self._find_loops(operations_dict)
        
        pulses = timing_table[timing_table['is_acquisition'] == False].copy()
        acquisitions = timing_table[timing_table['is_acquisition'] == True].copy()
        
        def process_rows(df):
            resolved_amps = []
            resolved_phases = []
            resolved_durations = []
            resolved_wfs = []
            
            for _, row in df.iterrows():
                t = row['abs_time']
                op_hash = row['operation_hash']
                op = all_ops.get(op_hash, {})
                data = op.data if hasattr(op, 'data') else op
                
                # Identify loop iterations for this pulse
                mapping = {}
                for l in loops:
                    if l['t_start'] <= t < l['t_end'] + 1e-15:
                        it_idx = int((t - l['t_start']) // l['iteration_duration'])
                        # Cap it_idx to repetitions - 1 to avoid floating point issues at boundaries
                        repetitions = l['op'].data.get('control_flow_info', {}).get('repetitions', 1)
                        it_idx = min(it_idx, repetitions - 1)
                        
                        for var, domain in l['domain'].items():
                            # Resolve LinearDomain value for this iteration
                            val = domain.start + it_idx * (domain.stop - domain.start) / (domain.num - 1) if domain.num > 1 else domain.start
                            mapping[var] = val
                
                # Extract and resolve pulse info
                a, p, dur, wf = 0.0, 0.0, row['duration'], 'square'
                p_info_list = data.get('pulse_info', [])
                if isinstance(p_info_list, dict): p_info_list = [p_info_list]
                
                if p_info_list:
                    p_info = p_info_list[0]
                    a = self._resolve_value(p_info.get('amp', 0.0), mapping)
                    p = self._resolve_value(p_info.get('phase', 0.0), mapping)
                    dur = self._resolve_value(p_info.get('duration', row['duration']), mapping)
                    wf = p_info.get('wf_func', 'square')
                
                resolved_amps.append(a)
                resolved_phases.append(p)
                resolved_durations.append(dur)
                resolved_wfs.append(wf)
            
            df['amp'] = resolved_amps
            df['phase'] = resolved_phases
            df['duration'] = resolved_durations
            df['wf_func'] = resolved_wfs
            return df

        pulses = process_rows(pulses)
        # Acquisitions might also have variables (e.g. acq_index)
        # but for now we focus on pulses for physics.
        
        # Re-use the physics simulation from the base class
        # We need to bypass its own 'pulses' extraction logic.
        # So we'll call a version of simulate that accepts pre-processed pulses.
        return self._simulate_processed(pulses, acquisitions, initial_state)

    def _simulate_processed(self, pulses, acquisitions, initial_state=None):
        # This is a copy-paste of the physics part of QbloxQutipSimulator.simulate
        # In a real refactor, I would extract this part in the base class.
        
        if len(pulses) == 0:
            total_duration = 1e-6
        else:
            total_duration = pulses['abs_time'].max() + pulses['duration'].max()
        if len(acquisitions) > 0:
            total_duration = max(total_duration, acquisitions['abs_time'].max() + acquisitions['duration'].max())

        t_list = np.linspace(0, total_duration, max(500, int(total_duration / 1e-9)))
        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(2, 0), qutip.basis(self.N_res, 0))
            
        def get_drive(t, port_name, pulses_list):
            val = 0.0j
            eps = 1e-13
            for p in pulses_list:
                if p['port'] == port_name:
                    if p['abs_time'] - eps <= t <= p['abs_time'] + p['duration'] + eps:
                        val += self._pulse_envelope(t, p)
            return val
        
        pulses_list = pulses.to_dict('records')
        omega_q = 2 * np.pi * self.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.rabi_freq_res_per_volt
        delta = 2 * np.pi * (self.f_q - self.f_d)
        h_static = (delta / 2) * self.sz + 2 * np.pi * self.chi * self.n * self.sz
        
        def qubit_drive_i(t, args): return np.real(get_drive(t, 'q0:mw', args['pulses']))
        def qubit_drive_q(t, args): return np.imag(get_drive(t, 'q0:mw', args['pulses']))
        def res_drive_i(t, args): return np.real(get_drive(t, 'q0:res', args['pulses']))
        def res_drive_q(t, args): return np.imag(get_drive(t, 'q0:res', args['pulses']))
        
        h = [
            h_static,
            [self.sx * (omega_q / 2), qubit_drive_i],
            [self.sy * (omega_q / 2), qubit_drive_q],
            [self.a + self.ad, lambda t, args: (omega_res / 2) * res_drive_i(t, args)],
            [1j * (self.ad - self.a), lambda t, args: (omega_res / 2) * res_drive_q(t, args)]
        ]
        
        c_ops = []
        if self.T1 < np.inf: c_ops.append(np.sqrt(1.0 / self.T1) * self.sm)
        if self.T2 < np.inf:
            gamma_phi = (1.0 / self.T2) - (0.5 / self.T1 if self.T1 < np.inf else 0)
            if gamma_phi > 0: c_ops.append(np.sqrt(2 * gamma_phi) * self.sz / 2.0)
        if self.kappa > 0: c_ops.append(np.sqrt(2 * np.pi * self.kappa) * self.a)
                
        result = qutip.mesolve(h, initial_state, t_list, c_ops=c_ops, args={'pulses': pulses_list})
        
        measurements = []
        for _, acq in acquisitions.iterrows():
            acq_time = acq['abs_time']
            idx = np.argmin(np.abs(t_list - acq_time))
            state = result.states[idx]
            rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
            prob_1 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(2, 1)), rho_q))
            measurements.append({
                'name': acq.get('acq_index', acq.get('acq_channel', 'acq')),
                'time': acq_time, 'prob_1': prob_1,
                'outcome': 1 if np.random.random() < prob_1 else 0
            })
            
        return {'result': result, 't_list': t_list, 'measurements': measurements}

from q1simulator import Cluster

class QbloxQ1Simulator:
    # ... (existing methods)
    """
    A simulator that takes a Q1Simulator object (https://github.com/sldesnoo-Delft/q1simulator/tree/main) and uses QuTiP to simulate 
    the dynamics of a qubit coupled to a readout resonator.
    """

    def __init__(self, params: dict, name: str = 'cluster', modules: dict = None, hardware_config: dict = None):

        # initialize a Q1 Cluster object with the given modules
        self.cluster = Cluster(name=name, modules=modules)

        if hardware_config is None:
            hardware_config = {}
        
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

        self.f_q = params.get('f_q', 5.0e9)
        self.f_d = params.get('f_d', self.f_q)
        self.f_res = params.get('f_res', 6.0e9)
        self.chi = params.get('chi', 1.0e6)
        self.rabi_freq_per_volt = params.get('rabi_freq_per_volt', 10.0e6)
        self.rabi_freq_res_per_volt = params.get('rabi_freq_res_per_volt', 10.0e6)
        self.T1 = params.get('T1', np.inf)
        self.T2 = params.get('T2', np.inf)
        self.kappa = params.get('kappa', 1e6)
        self.N_res = params.get('N_res', 2)     # number of resonator states to simulate 

        # Operators in the joint Hilbert space
        self.sx = qutip.tensor(qutip.sigmax(), qutip.identity(self.N_res))
        self.sy = qutip.tensor(qutip.sigmay(), qutip.identity(self.N_res))
        self.sz = qutip.tensor(qutip.sigmaz(), qutip.identity(self.N_res))
        self.sm = qutip.tensor(qutip.sigmam(), qutip.identity(self.N_res))
        
        self.a = qutip.tensor(qutip.identity(2), qutip.destroy(self.N_res))
        self.ad = self.a.dag()
        self.n = self.ad * self.a

    def simulate(self, initial_state: qutip.Qobj = None):

        try:
            drive_pulses, readout_pulses = self.get_pulses()
        except Exception as e:
            print("Error extracting pulses from Q1Simulator:", e)
            drive_pulses, readout_pulses = {}, {}
        
        if self.t_max == 0:
            print("Warning: t_max is 0, using default time range")
            self.t_max = 500e-9  # 500 ns
            self.t_sample = 1e-9
        
        self.t_list = np.linspace(0, self.t_max, int(self.t_max/self.t_sample))*1e-9

        if initial_state is None:
            initial_state = qutip.tensor(qutip.basis(2, 0), qutip.basis(self.N_res, 0))

        omega_q = 2 * np.pi * self.rabi_freq_per_volt
        omega_res = 2 * np.pi * self.rabi_freq_res_per_volt
        delta = 2 * np.pi * (self.f_q - self.f_d)
        h_static = (delta / 2) * self.sz

        # Get pulse data - if not available, use simple Gaussian pulse
        drive_I_data = np.array(drive_pulses.get("I", {}).get("data", []))
        drive_Q_data = np.array(drive_pulses.get("Q", {}).get("data", []))

        
        if len(drive_I_data) == 0:
            # Use a simple Gaussian pulse as default
            pulse_duration = 60e-9
            pulse_center = pulse_duration / 2
            sigma = pulse_duration / 4
            drive_I_data = np.exp(-((self.t_list - pulse_center)**2) / (2 * sigma**2))
            drive_Q_data = np.zeros_like(self.t_list)
            print("Using default Gaussian drive pulse")


        # Create smooth interpolation functions with gentle scaling
        def create_smooth_interp(data_array, max_scale=5):
            """Create a smooth interpolation function with scaling"""
            def func(t, args):
                if t < 0 or t > self.t_max or len(data_array) == 0:
                    return 0.0
                # Linear interpolation between points
                idx = t / self.t_max * (len(data_array) - 1)
                if idx >= len(data_array) - 1:
                    val = float(data_array[-1])
                else:
                    i_low = int(idx)
                    i_high = i_low + 1
                    frac = idx - i_low
                    val = float((1 - frac) * data_array[i_low] + frac * data_array[i_high])
                # Scale down to prevent stiffness
                return val * max_scale
            return func

        #drive_I_func = create_smooth_interp(drive_I_data, max_scale=5)
        #drive_Q_func = create_smooth_interp(drive_Q_data, max_scale=5)

        h = [
            h_static,
            [self.sx * (omega_q / 2), drive_I_data],
            [self.sy * (omega_q / 2), drive_Q_data],
        ]
        
        c_ops = []
        if self.T1 < np.inf: 
            c_ops.append(np.sqrt(1.0 / self.T1) * self.sm)
        if self.T2 < np.inf:
            gamma_phi = (1.0 / self.T2) - (0.5 / self.T1 if self.T1 < np.inf else 0)
            if gamma_phi > 0: 
                c_ops.append(np.sqrt(2 * gamma_phi) * self.sz / 2.0)
        
        # Create solver options - very lenient for potentially stiff systems
        # options = qutip.Options(
        #     nsteps=5e6,  # Very high limit
        #     rtol=1e-2,      # Very loose relative tolerance
        #     atol=1e-4,      # Very loose absolute tolerance
        #     method='adams'  # Non-stiff integrator
        # )
        
        try:
            print("Starting QuTiP mesolve simulation...")
            print(f"  - Time range: 0 to {self.t_max:.2f} ns")
            print(f"  - Number of time points: {len(self.t_list)}")
            result = qutip.mesolve(h, initial_state, self.t_list, c_ops=c_ops) #mesolve? 
            print("✓ Simulation completed successfully")
        except Exception as e:
            print(f"✗ Solver failed: {e}")
        
        measurements = self.get_measurements(result)

        try:
            qrm = self.cluster.get_connected_modules()[4]
            if measurements and len(measurements) > 0 and 'I' in measurements[0] and 'Q' in measurements[0]:
                I = np.array([m['I'] for m in measurements])
                Q = np.array([m['Q'] for m in measurements])
                qrm.sequencers[0].set_acquisition_mock_data(I + 1j*Q)
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

        try:
            acq_windows = self.cluster.get_connected_modules()[self.readout_mod].get_acquisition_windows()
        except Exception as e:
            print(f"Warning: Could not get acquisition windows: {e}")
            return []
            
        sigma = 0.1
        mean = 0
        measurements = []
        
        try:
            for acq in acq_windows[f'sequencer{self.readout_seq}']:
                try:
                    acq_start = acq[0][0]
                    idx = np.argmin(np.abs(self.t_list - acq_start))
                    state = result.states[idx]
                    rho_q = state.ptrace(0) if state.type == 'oper' else qutip.ket2dm(state).ptrace(0)
                    prob_1 = np.real(qutip.expect(qutip.ket2dm(qutip.basis(2, 1)), rho_q))
                    I = 1/np.sqrt(2)*prob_1 * np.random.normal(mean, sigma, 1)
                    Q = 1j*1/np.sqrt(2)*prob_1 * np.random.normal(mean, sigma, 1)
                    measurements.append({
                        'time': acq_start, 'prob_1': prob_1,
                        'outcome': 1 if np.random.random() < prob_1 else 0,
                        'I': I,
                        'Q': Q
                    })
                except Exception as e:
                    print(f"Warning: Could not process acquisition: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Error iterating acquisition windows: {e}")
        
        return measurements
