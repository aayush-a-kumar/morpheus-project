import pytest
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator
from qblox_sim.signals import ScheduleSignalProvider


def test_diagnose_envelope_params(default_qubit_params):
    sim = QbloxQutipSimulator(default_qubit_params)

    sched = Schedule("Diagnostic Pi Pulse")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    sched.add(SquarePulse(amp=0.1, duration=100e-9, port="q0:mw", clock="q0.01"))

    orig_get_drives = ScheduleSignalProvider.get_drives
    logs = []

    def spy_get_drives(self, t_list):
        drives = orig_get_drives(self, t_list)
        max_q_amp = float(np.max(np.abs(drives["q_drive"])))
        
        # FIX: ScheduleSignalProvider stores pulses in self.pulses_list
        pulses_count = len(self.pulses_list)
        
        print("\n\n=== [EVALUATION LOG] ===")
        print(f"Time grid points: {len(t_list)}")
        print(f"Pulses processed: {pulses_count}")
        print(f"Max 'q0:mw' Drive Amplitude: {max_q_amp:.3f}")
        
        logs.append({'t_points': len(t_list), 'max_amp': max_q_amp})
        return drives

    ScheduleSignalProvider.get_drives = spy_get_drives  # type: ignore[method-assign]
    try:
        res = sim.simulate(sched)
    finally:
        ScheduleSignalProvider.get_drives = orig_get_drives

    assert len(logs) == 1
    assert logs[0]['max_amp'] == pytest.approx(0.1, abs=1e-3)
    assert res is not None