import pytest
import pandas as pd
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator


def test_diagnose_envelope_params(default_qubit_params):
    sim = QbloxQutipSimulator(default_qubit_params)

    sched = Schedule("Diagnostic Pi Pulse")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    sched.add(SquarePulse(amplitude=0.1, duration=100e-9, port="q0:mw", clock="q0.01"))

    # Save original method reference
    orig_env = sim._pulse_envelope

    print("\n\n=== [EVALUATION LOG] ===")
    def spy_env(t: float, pulse_info: dict) -> complex:
        res = orig_env(t, pulse_info)
        print(
            f"t = {t*1e9:6.2f} ns | "
            f"port = {repr(pulse_info.get('port')):7s} | "
            f"abs_time = {pulse_info.get('abs_time', 0)*1e9:5.1f} ns | "
            f"dur = {pulse_info.get('duration', 0)*1e9:5.1f} ns | "
            f"amp = {sim._extract_amplitude(pulse_info):.3f} | "
            f"wf = {repr(pulse_info.get('wf_func'))} -> "
            f"result = {res}"
        )
        return res

    sim._pulse_envelope = spy_env  # type: ignore[assignment]
    sim.simulate(sched)