# SPDX-License-Identifier: LicenseRef-Proprietary
import typing
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule, SerialCompiler, QuantumDevice, BasicTransmonElement
from qblox_scheduler.operations import LoopOperation

from qblox_sim.config import SimulationConfig
from qblox_sim.physics import QuantumSystem
from qblox_sim.signals import ScheduleSignalProvider, extract_amplitude
from qblox_sim.acquisitions import AcquisitionRegistry
from qblox_sim.engine import QuTiPEngine

from qcodes.instrument import Instrument
from q1simulator import Cluster

class SimulationResult:
    """Clean container for stitched time-series simulation outputs."""
    def __init__(
        self, 
        states: list, 
        t_list: np.ndarray, 
        measurements: list,
        system: typing.Optional[QuantumSystem] = None
    ):
        self.states = states
        self.t_list = t_list
        self.measurements = measurements
        self._system = system

    def get_expectation(self, op_name: str = 'sz', q_name: str = 'q0') -> np.ndarray:
        """Helper to extract expectation values for a specific qubit."""
        if self._system is None:
            raise ValueError("No QuantumSystem attached to SimulationResult.")
            
        op_dict = getattr(self._system, op_name, None)
        if op_dict is None or not isinstance(op_dict, dict):
            raise ValueError(f"Operator dictionary '{op_name}' not found on QuantumSystem.")
            
        op = op_dict.get(q_name)
        if op is None:
            raise ValueError(f"Operator for qubit '{q_name}' not found in '{op_name}'.")
            
        return np.array([qutip.expect(op, s).real for s in self.states])


class QbloxQutipSimulator:
    """
    A simulator that takes a Qblox-Scheduler Schedule and uses QuTiP to simulate 
    the dynamics of an N-qubit topology coupled to readout resonators.
    """

    def __init__(self, params: dict, configs: typing.Optional[dict] = None):
        self.params = params  
        self.configs = configs
        
        self.cfg = SimulationConfig.from_dict(params)
        self.system = QuantumSystem(self.cfg)
        self.engine = QuTiPEngine()

    # =========================================================================
    # Helpers & Compilation
    # =========================================================================
    def _flatten_operations(self, operations_dict: dict, all_ops: typing.Optional[dict] = None) -> dict:
        if all_ops is None:
            all_ops = {}
        for h, op in operations_dict.items():
            all_ops[h] = op
            if isinstance(op, LoopOperation):
                self._flatten_operations(op.body.operations, all_ops) # type: ignore
            elif hasattr(op, 'operations'):
                self._flatten_operations(op.operations, all_ops)
            body = getattr(op, 'body', None)
            if body is not None and hasattr(body, 'operations'):
                self._flatten_operations(getattr(body, 'operations'), all_ops)
        return all_ops

    def _get_op_from_hash(self, op_hash: typing.Any, ops_dict: dict) -> dict:
        op = ops_dict.get(op_hash)
        if op is None and isinstance(op_hash, (int, str)):
            try:
                op = ops_dict.get(int(op_hash)) or ops_dict.get(str(op_hash))
            except ValueError:
                pass
        return op if op is not None else {}

    def _get_compiled_schedule(self, schedule: Schedule) -> typing.Tuple[pd.DataFrame, dict]:
        try:
            return schedule.timing_table.data, schedule.operations # type: ignore
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
            return compiled_sched.timing_table.data, compiled_sched.operations # type: ignore

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

    # =========================================================================
    # Simulation Execution
    # =========================================================================
    def simulate(self, schedule: Schedule, initial_state: typing.Optional[qutip.Qobj] = None) -> dict:
        uncompiled_ops = self._flatten_operations(schedule.operations)
        timing_table, operations_dict = self._get_compiled_schedule(schedule)
        loops = self._find_loops(operations_dict)

        pulses = timing_table[timing_table['is_acquisition'] == False].copy()
        acquisitions = timing_table[timing_table['is_acquisition'] == True].copy()
        
        pulses = self._resolve_loop_pulses(pulses, uncompiled_ops, loops)
        
        if loops:
            return self._simulate_shot_sweep(pulses, acquisitions, loops, uncompiled_ops, initial_state)
        return self._simulate_processed(pulses, acquisitions, uncompiled_ops, initial_state)

    def _resolve_loop_pulses(self, pulses: pd.DataFrame, uncompiled_ops: dict, loops: list) -> pd.DataFrame:
        resolved_amps, resolved_phases, resolved_durations, resolved_wfs = [], [], [], []
        
        # Track phase shifts by CLOCK, not port
        tracked_phases = {}

        for _, row in pulses.iterrows():
            t = row['abs_time']
            op_hash = row['operation_hash']
            op = self._get_op_from_hash(op_hash, uncompiled_ops)
            data = getattr(op, 'data', op) if hasattr(op, 'data') else op

            # --- 1. Loop Variable Resolution (Defines 'mapping') ---
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

            # --- 2. Intercept Virtual Z Gates (Logical Phase Shifts) ---
            logic_info = data.get('logic_info', data) if isinstance(data, dict) else {}
            if 'phase_shift' in logic_info:
                clock = logic_info.get('clock', data.get('clock', 'default'))
                if clock not in tracked_phases:
                    tracked_phases[clock] = 0.0
                
                # mapping is now safely defined!
                resolved_shift = self._resolve_value(logic_info['phase_shift'], mapping)
                tracked_phases[clock] += float(resolved_shift)

            # --- 3. Standard Pulse Extraction ---
            a, p, dur, wf = 0.0, 0.0, row.get('duration', 0.0), 'square'
            
            p_info_list = data.get('pulse_info', []) if isinstance(data, dict) else []
            if isinstance(p_info_list, dict): 
                p_info_list = [p_info_list]
                
            if p_info_list:
                p_info = p_info_list[0]
                clock = p_info.get('clock', 'default')
                
                if clock not in tracked_phases:
                    tracked_phases[clock] = 0.0

                if 'phase_shift' in p_info:
                    resolved_shift = self._resolve_value(p_info['phase_shift'], mapping)
                    tracked_phases[clock] += float(resolved_shift)

                raw_amp = extract_amplitude(p_info)
                a = self._resolve_value(raw_amp, mapping)
                
                base_phase = self._resolve_value(p_info.get('phase', 0.0), mapping)
                p = float(base_phase) + tracked_phases[clock]
                
                dur = self._resolve_value(p_info.get('duration', row.get('duration', 0.0)), mapping)
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

    def _simulate_shot_sweep(self, pulses: pd.DataFrame, acquisitions: pd.DataFrame, loops: list, operations_dict: dict, initial_state: typing.Optional[qutip.Qobj]) -> dict:
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
        result_container = SimulationResult(combined_states, final_t_list, [m for r in all_results for m in r['measurements']], system=self.system)

        return {'result': result_container, 't_list': final_t_list, 'measurements': result_container.measurements}

    def _simulate_processed(self, pulses, acquisitions, operations_dict=None, initial_state=None):
        operations_dict = operations_dict or {}

        pulses_list = pulses.to_dict('records')
        for p in pulses_list:
            p['abs_time'] = p['abs_time'] * 1e-9 if p['abs_time'] > 1e-3 else p['abs_time']
            p['duration'] = p['duration'] * 1e-9 if p['duration'] > 1e-3 else p['duration']

        if len(pulses_list) == 0:
            total_duration = 1e-6
        else:
            total_duration = max(p['abs_time'] + p['duration'] for p in pulses_list)

        if len(acquisitions) > 0:
            acq_max = acquisitions['abs_time'].max() + acquisitions['duration'].max()
            acq_max = acq_max * 1e-9 if acq_max > 1e-3 else acq_max
            total_duration = max(total_duration, acq_max)

        # 3. Create a STRICTLY UNIFORM time grid (Configurable resolution)
        step_size = self.cfg.dt
        num_points = max(1000, int(np.ceil(total_duration / step_size)) + 1)
        t_list = np.linspace(0, total_duration, num_points)

        if initial_state is None:
            initial_state = self.system.get_default_initial_state()
            
        signal_provider = ScheduleSignalProvider(pulses_list)
        drives = signal_provider.get_drives(t_list)

        result = self.engine.run(system=self.system, drives=drives, t_list=t_list, initial_state=initial_state)

        measurements = self._process_acquisitions(acquisitions, result.states, t_list, operations_dict)

        result_container = SimulationResult(
            states=result.states if result is not None else [],
            t_list=t_list,
            measurements=measurements,
            system=self.system
        )
            
        return {'result': result_container, 't_list': t_list, 'measurements': measurements}

    def _process_acquisitions(self, acquisitions: pd.DataFrame, states: list, t_list: np.ndarray, operations_dict: dict) -> list:
        """Shared measurement extraction loop."""
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
            #FIX: casting protocol to string to avoid issues with float('nan') or None
            protocol = str(protocol) if protocol is not None and not (isinstance(protocol, float) and np.isnan(protocol)) else 'SSBIntegrationComplex'

            idx = np.argmin(np.abs(t_list - acq_time))
            state = states[idx]

            acq_channel = acq_info.get('acq_channel', acq.get('acq_channel', acq.get('acq_index', 'acq')))
            res_name = str(acq_channel).split(':')[0] if isinstance(acq_channel, str) and ':' in acq_channel else 'q0'
            
            # PYLANCE TYPE FIX: Explicitly fallback to guarantee a valid Qobj
            a_op = self.system.a.get(res_name)
            ad_op = self.system.ad.get(res_name)
            if a_op is None or ad_op is None:
                a_op = self.system.a["q0"]
                ad_op = self.system.ad["q0"]

            handler = AcquisitionRegistry.get_handler(protocol)
            processed_data = handler.process(
                state=state, t_list=t_list, states=states, acq_time=acq_time,
                acq_duration=acq_duration, acq_info=acq_info, cfg=self.cfg, 
                a_op=a_op, ad_op=ad_op
            )
            
            meas_dict = {'name': acq_channel, 'protocol': protocol, 'time': acq_time, 'duration': acq_duration}
            meas_dict.update(processed_data)
            measurements.append(meas_dict)
            
        return measurements


class QbloxQ1Simulator(QbloxQutipSimulator):
    """
    A hardware-adapter simulator that acts as a QCoDeS Cluster and passes extracted 
    pulses to the multi-qubit engine.
    """

    def __init__(self, params: dict, name: str = 'cluster', modules: typing.Optional[dict] = None, hardware_config: typing.Optional[dict] = None):
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
        if hasattr(self, 'cluster') and self.cluster is not None:
            try:
                self.cluster.close()
            except Exception:
                pass
            self.cluster = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

    def simulate(self, schedule: typing.Optional[Schedule] = None, initial_state: typing.Optional[qutip.Qobj] = None) -> dict:
        try:
            drive_pulses, readout_pulses = self.get_pulses()
        except Exception as e:
            print("Error extracting pulses from Q1Simulator:", e)
            drive_pulses, readout_pulses = {}, {}
        
        if self.t_max == 0:
            self.t_max = 500
            self.t_sample = 1
        
        num_points = int(round(self.t_max / self.t_sample)) if self.t_sample > 0 else 500
        self.t_list = np.linspace(0, self.t_max * 1e-9, num_points)

        if initial_state is None:
            initial_state = self.system.get_default_initial_state()

        def safe_data(pulse_dict, key):
            arr = np.array(pulse_dict.get(key, {}).get("data", []))
            if len(arr) != len(self.t_list):
                if len(arr) > 0:
                    old_t = np.linspace(0, self.t_max * 1e-9, len(arr))
                    arr = np.interp(self.t_list, old_t, arr)
                else:
                    arr = np.zeros_like(self.t_list)
            return arr

        # Map hardware traces to standardized multi-qubit engine ports
        drives = {
            "q0:mw": safe_data(drive_pulses, "I") + 1j * safe_data(drive_pulses, "Q"),
            "q0:res": safe_data(readout_pulses, "I") + 1j * safe_data(readout_pulses, "Q")
        }

        options = {"nsteps": 100000, "max_step": 1e-9, "rtol": 1e-6, "atol": 1e-8, "method": "bdf"}
        
        result = None
        try:
            result = self.engine.run(system=self.system, drives=drives, t_list=self.t_list, initial_state=initial_state, options=options) 
        except Exception as e:
            print(f"✗ Solver failed: {e}")
        
        measurements = self.get_measurements(result)
        self.set_acquisition_mock_data(measurements)

        result_container = SimulationResult(
            states=result.states if result is not None else [],
            t_list=self.t_list,
            measurements=measurements,
            system=self.system
        )
        return {'result': result_container, 't_list': self.t_list, 'measurements': measurements}
    
    def get_pulses(self):
        drive_pulses, readout_pulses = {}, {}
        # 3. FIX: Pylance guard
        if self.cluster is None: return drive_pulses, readout_pulses

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
            if self.readout_mod not in connected: return []
            qrm = connected[self.readout_mod]
            acq_windows = qrm.get_acquisition_windows()

            if not acq_windows or not any(acq_windows.values()):
                if hasattr(qrm, 'sequencers') and len(qrm.sequencers) > self.readout_seq:
                    seq_obj = qrm.sequencers[self.readout_seq]
                    if hasattr(seq_obj, 'acq_windows'):
                        acq_windows = {self.readout_seq: getattr(seq_obj, 'acq_windows')}
        except Exception:
            return []

        windows = (
            acq_windows.get(self.readout_seq) or acq_windows.get(str(self.readout_seq)) or 
            acq_windows.get(f"sequencer{self.readout_seq}") or acq_windows.get(f"seq{self.readout_seq}")
        )
        if not windows: return []

        measurements = []
        handler = AcquisitionRegistry.get_handler('Trace')
        
        # PYLANCE TYPE FIX
        a_op = self.system.a.get("q0")
        ad_op = self.system.ad.get("q0")
        if a_op is None or ad_op is None: return measurements

        for acq in windows:
            try:
                if isinstance(acq, (list, tuple)) and len(acq) > 0 and isinstance(acq[0], (list, tuple)):
                    acq_start_ns, duration_ns = acq[0][0], acq[0][1]
                elif isinstance(acq, (list, tuple)) and len(acq) >= 2:
                    acq_start_ns, duration_ns = acq[0], acq[1]
                else: continue

                acq_start_sec, acq_duration_sec = float(acq_start_ns) * 1e-9, float(duration_ns) * 1e-9
                idx = np.argmin(np.abs(self.t_list - acq_start_sec))
                
                res_dict = handler.process(
                    state=result.states[idx], t_list=self.t_list, states=result.states,
                    acq_time=acq_start_sec, acq_duration=acq_duration_sec,
                    acq_info={'acq_delay': self.cfg.acquisition.cable_delay},
                    cfg=self.cfg, a_op=a_op, ad_op=ad_op
                )

                measurements.append({
                    'time': acq_start_sec, 'duration': acq_duration_sec,
                    'prob_0': res_dict['prob_0'], 'prob_1': res_dict['prob_1'],
                    'leakage_prob_2': res_dict['leakage_prob_2'], 'outcome': res_dict['outcome'],
                    'I': res_dict['I'], 'Q': res_dict['Q'], 'trace': res_dict['value'],
                    'trace_I': np.real(res_dict['value']), 'trace_Q': np.imag(res_dict['value'])
                })
            except Exception: continue

        return measurements

    def set_acquisition_mock_data(self, measurements: list):
        if self.cluster is None or not measurements: return
        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod in connected:
                qrm = connected[self.readout_mod]
                if 'I' in measurements[0] and 'Q' in measurements[0]:
                    I_arr, Q_arr = np.array([m['I'] for m in measurements]), np.array([m['Q'] for m in measurements])
                    qrm.sequencers[self.readout_seq].set_acquisition_mock_data(I_arr + 1j * Q_arr)
        except Exception: pass