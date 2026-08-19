# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
import numpy as np
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import linspace
from qblox_scheduler.resources import ClockResource

from qblox_sim.signals import ScheduleSignalProvider, extract_amplitude
from qblox_sim.simulator import QbloxQutipSimulator


def test_diagnose_stage4_granular(default_qubit_params):
    sim = QbloxQutipSimulator(default_qubit_params)

    f_q = sim.cfg.qubits["q0"].f_q
    sched = Schedule("Debug Loop Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=f_q))
    amp_domain = linspace(0.0, 0.5, 5, dtype=DType.AMPLITUDE)
    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amp=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
        sched.add(SquarePulse(amp=0.0, duration=50e-9, port="q0:mw", clock="q0.01"))

    # Log metrics captured during execution
    provider_logs = []

    orig_simulate_processed = sim._simulate_processed

    def spy_simulate_processed(
        pulses, acquisitions, operations_dict=None, initial_state=None
    ):
        pulses_list = pulses.to_dict("records")
        print("\n=== [SIMULATE PROCESSED INSPECTION] ===")
        print(f"Total pulse records passed: {len(pulses_list)}")
        for idx, p in enumerate(pulses_list):
            raw_t = p.get("abs_time", 0.0) or 0.0
            raw_dur = p.get("duration", 0.0) or 0.0
            raw_amp = extract_amplitude(p)
            print(
                f"Record {idx:2d} | port={p.get('port')!r:<7s} | t0={raw_t * 1e9:5.1f}ns | dur={raw_dur * 1e9:4.1f}ns | amp={raw_amp}"
            )

        orig_get_drives = ScheduleSignalProvider.get_drives

        def spy_get_drives(provider_self, t_list: np.ndarray):
            drives = orig_get_drives(provider_self, t_list)
            q_drive = drives.get("q0:mw", np.zeros_like(t_list))

            non_zero_q = np.count_nonzero(q_drive)
            max_q_amp = np.max(np.abs(q_drive)) if len(q_drive) > 0 else 0.0

            provider_logs.append(
                {
                    "num_time_points": len(t_list),
                    "non_zero_q_samples": non_zero_q,
                    "max_q_amp": max_q_amp,
                    "pulses_count": len(provider_self.pulses_list),
                }
            )
            return drives

        ScheduleSignalProvider.get_drives = spy_get_drives  # type: ignore[method-assign]

        try:
            return orig_simulate_processed(
                pulses, acquisitions, operations_dict, initial_state
            )
        finally:
            ScheduleSignalProvider.get_drives = orig_get_drives  # Restore after call

    sim._simulate_processed = spy_simulate_processed  # type: ignore[method-assign]

    print("\n\n=== [RUNNING SIMULATION] ===")
    res = sim.simulate(sched)

    print(f"\nTotal provider calls (shots/iterations): {len(provider_logs)}")

    for idx, log in enumerate(provider_logs):
        print(
            f"Shot {idx:2d} | "
            f"Time points: {log['num_time_points']:5d} | "
            f"Non-zero Q samples: {log['non_zero_q_samples']:4d} | "
            f"Max Q Amp: {log['max_q_amp']:.4f}"
        )

    # Basic assertion to verify loop sweep resolved pulses properly
    assert len(provider_logs) == 5, (
        f"Expected 5 loop sweep iterations, got {len(provider_logs)}"
    )
    assert res is not None
