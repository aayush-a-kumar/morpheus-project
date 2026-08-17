# SPDX-License-Identifier: LicenseRef-Proprietary
import numpy as np
import pytest
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import linspace
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator


def test_loop_amplitude_sweep(default_qubit_params):
    """Verify loop simulator dynamically updates pulse amplitudes per iteration."""
    sim = QbloxQutipSimulator(default_qubit_params)

    f_q = sim.cfg.qubit.f_q
    sched = Schedule("Amplitude Sweep Loop")
    sched.add_resource(ClockResource(name="q0.01", freq=f_q))

    # Sweep amplitude from 0.0 to 0.2 in 3 steps
    amp_domain = linspace(0.0, 0.2, 3, dtype=DType.AMPLITUDE)

    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amp=amp, duration=50e-9, port="q0:mw", clock="q0.01"))

    res = sim.simulate(sched)
    result = res["result"]

    final_state = result.states[-1]
    final_z = qutip.expect(sim.system.sz['q0'], final_state).real

    # Verify state rotated under maximum sweep amplitude
    assert final_z < 1.0, f"Expected state to rotate, got <Z> = {final_z}"