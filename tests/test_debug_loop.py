import pytest
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule, linspace, DType
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxLoopSimulator


def test_diagnose_stage4_granular(default_qubit_params):
    sim = QbloxLoopSimulator(default_qubit_params)

    sched = Schedule("Debug Loop Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    amp_domain = linspace(0.0, 0.5, 5, dtype=DType.AMPLITUDE)
    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amplitude=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
        sched.add(SquarePulse(amplitude=0.0, duration=50e-9, port="q0:mw", clock="q0.01"))

    # Track ALL calls to _pulse_envelope during sim.simulate(sched)
    all_envelope_calls = []
    orig_envelope = sim._pulse_envelope

    def spy_env(t: float, pulse_info: dict) -> complex:
        res = orig_envelope(t, pulse_info)
        all_envelope_calls.append({
            't_ns': t * 1e9,
            'port': pulse_info.get('port'),
            'abs_time_ns': pulse_info.get('abs_time', 0.0) * 1e9 if pulse_info.get('abs_time') is not None else 0.0,
            'duration_ns': pulse_info.get('duration', 0.0) * 1e9 if pulse_info.get('duration') is not None else 0.0,
            'amp': sim._extract_amplitude(pulse_info),
            'res': res
        })
        return res

    sim._pulse_envelope = spy_env  # type: ignore[method-assign]

    print("\n\n=== [RUNNING SIMULATION WITH SPY] ===")
    res = sim.simulate(sched)

    print(f"Total calls to _pulse_envelope: {len(all_envelope_calls)}")

    non_zero_calls = [c for c in all_envelope_calls if abs(c['res']) > 0]
    print(f"Total non-zero envelope calls:  {len(non_zero_calls)}")

    if all_envelope_calls:
        print("\nFirst 10 calls to _pulse_envelope:")
        for c in all_envelope_calls[:10]:
            print(f"  t = {c['t_ns']:6.2f}ns | port = {repr(c['port']):7s} | pulse_t0 = {c['abs_time_ns']:5.1f}ns | dur = {c['duration_ns']:4.1f}ns | amp = {c['amp']} -> {c['res']}")

    if non_zero_calls:
        print("\nFirst 10 NON-ZERO calls to _pulse_envelope:")
        for c in non_zero_calls[:10]:
            print(f"  t = {c['t_ns']:6.2f}ns | port = {repr(c['port']):7s} | pulse_t0 = {c['abs_time_ns']:5.1f}ns | dur = {c['duration_ns']:4.1f}ns | amp = {c['amp']} -> {c['res']}")