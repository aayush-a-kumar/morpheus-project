import pytest
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations.loop_domains import linspace
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxLoopSimulator


def test_diagnose_stage4_granular(default_qubit_params):
    sim = QbloxLoopSimulator(default_qubit_params)

    sched = Schedule("Debug Loop Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    amp_domain = linspace(0.0, 0.5, 5, dtype=DType.AMPLITUDE)
    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amp=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
        sched.add(SquarePulse(amp=0.0, duration=50e-9, port="q0:mw", clock="q0.01"))

    # Track drive calls and pulse envelope parameters
    callback_logs = []
    
    orig_simulate_processed = sim._simulate_processed

    def spy_simulate_processed(pulses, acquisitions, operations_dict=None, initial_state=None):
        pulses_list = pulses.to_dict('records')
        print(f"\n=== [SIMULATE PROCESSED INSPECTION] ===")
        print(f"Total pulse records passed: {len(pulses_list)}")
        for idx, p in enumerate(pulses_list):
            raw_t = p.get('abs_time', 0.0) or 0.0
            raw_dur = p.get('duration', 0.0) or 0.0
            raw_amp = sim._extract_amplitude(p)
            print(f"Record {idx:2d} | port={repr(p.get('port')):<7s} | t0={raw_t*1e9:5.1f}ns | dur={raw_dur*1e9:4.1f}ns | amp={raw_amp}")

        # Wrap _pulse_envelope to record every time it is called
        orig_env = sim._pulse_envelope
        def spy_env(t: float, pulse_info: dict) -> complex:
            v = orig_env(t, pulse_info)
            callback_logs.append({
                't_ns': t * 1e9,
                'port': pulse_info.get('port'),
                'abs_time_ns': (pulse_info.get('abs_time', 0.0) or 0.0) * 1e9,
                'duration_ns': (pulse_info.get('duration', 0.0) or 0.0) * 1e9,
                'amp': sim._extract_amplitude(pulse_info),
                'voltage': v
            })
            return v

        sim._pulse_envelope = spy_env  # type: ignore[method-assign]
        return orig_simulate_processed(pulses, acquisitions, operations_dict, initial_state)

    sim._simulate_processed = spy_simulate_processed  # type: ignore[method-assign]

    print("\n\n=== [RUNNING SIMULATION] ===")
    res = sim.simulate(sched)

    print(f"\nTotal calls to _pulse_envelope during mesolve: {len(callback_logs)}")
    
    non_zero_calls = [c for c in callback_logs if abs(c['voltage']) > 0]
    print(f"Total non-zero envelope returns:              {len(non_zero_calls)}")

    if callback_logs:
        print("\nFirst 10 envelope evaluations during solver run:")
        for c in callback_logs[:10]:
            print(
                f"  t = {c['t_ns']:6.2f}ns | "
                f"port = {repr(c['port']):7s} | "
                f"pulse_t0 = {c['abs_time_ns']:5.1f}ns | "
                f"dur = {c['duration_ns']:4.1f}ns | "
                f"amp = {c['amp']} -> "
                f"V = {c['voltage']}"
            )

    if non_zero_calls:
        print("\nFirst 10 NON-ZERO envelope evaluations:")
        for c in non_zero_calls[:10]:
            print(
                f"  t = {c['t_ns']:6.2f}ns | "
                f"port = {repr(c['port']):7s} | "
                f"pulse_t0 = {c['abs_time_ns']:5.1f}ns | "
                f"dur = {c['duration_ns']:4.1f}ns | "
                f"amp = {c['amp']} -> "
                f"V = {c['voltage']}"
            )