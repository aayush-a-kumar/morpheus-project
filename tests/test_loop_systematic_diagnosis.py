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


def test_systematic_loop_diagnosis(default_qubit_params):
    """Systematic 4-Stage Diagnostic Test for QbloxLoopSimulator."""
    sim = QbloxLoopSimulator(default_qubit_params)
    sim.rabi_freq_per_volt = 100e6  # 100 MHz / Volt

    sched = Schedule("Diagnostic Rabi Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    amp_domain = linspace(0.0, 0.5, 5, dtype=DType.AMPLITUDE)
    
    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amp=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
        sched.add(SquarePulse(amp=0.0, duration=50e-9, port="q0:mw", clock="q0.01"))

    captured_data = {'pulses_list': []}
    orig_simulate_processed = sim._simulate_processed

    # Spy captures raw pulse dataframes per shot
    def spy_simulate_processed(pulses, acquisitions, operations_dict=None, initial_state=None):
        captured_data['pulses_list'].append(pulses.copy())
        return orig_simulate_processed(pulses, acquisitions, operations_dict, initial_state)

    sim._simulate_processed = spy_simulate_processed  # type: ignore[method-assign]

    # Capture the stitched simulation output returned by sim.simulate()
    sim_output = sim.simulate(sched)
    
    # Concatenate all shot dataframes into a single table of 10 rows
    pulses_df = pd.concat(captured_data['pulses_list'], ignore_index=True)
    
    # -------------------------------------------------------------
    # STAGE 1: Variable Resolution & Unrolling Verification
    # -------------------------------------------------------------
    expected_amps = [0.0, 0.0, 0.125, 0.0, 0.25, 0.0, 0.375, 0.0, 0.5, 0.0]
    actual_amps = pulses_df['amp'].tolist()
    
    assert len(actual_amps) == 10, f"Stage 1 Failed: Expected 10 pulse rows, got {len(actual_amps)}"
    assert np.allclose(actual_amps, expected_amps, atol=1e-5), (
        f"Stage 1 Failed: Expected {expected_amps}, got {actual_amps}"
    )
    print("\n✓ STAGE 1 PASSED: Variable unrolling correctly resolved amplitude sweep per iteration.")

    # -------------------------------------------------------------
    # STAGE 2: Time Alignment & Pre-Sampling Check
    # -------------------------------------------------------------
    t_list = sim_output['t_list']
    pulse_records = pulses_df.to_dict('records')
    
    for idx in range(5):
        pulse_start = idx * 100e-9
        pulse_end = pulse_start + 50e-9
        expected_a = expected_amps[idx * 2]
        
        mask = (t_list >= pulse_start + 1e-9) & (t_list <= pulse_end - 1e-9)
        sampled_times = t_list[mask]

        # FIX: Set abs_time to absolute pulse_start for accurate envelope sampling across shots
        p_info = pulse_records[idx * 2].copy()
        p_info['abs_time'] = pulse_start
        
        sampled_envs = [sim._pulse_envelope(t, p_info) for t in sampled_times]
        
        if len(sampled_envs) > 0:
            max_env = np.max(np.abs(sampled_envs))
            assert np.isclose(max_env, expected_a, atol=1e-5), (
                f"Stage 2 Failed at Iteration {idx}: Expected amp {expected_a}, sampled {max_env}"
            )
    print("✓ STAGE 2 PASSED: Drive pre-sampling grid aligns with active pulse windows.")

    # -------------------------------------------------------------
    # STAGE 3: State Trajectory & Boundary Reset Inspection
    # -------------------------------------------------------------
    result = sim_output['result']
    start_z_values = []
    
    for idx in range(5):
        # Query slightly inside shot window (+0.1 ns) to hit the shot's ground state reset point
        it_start_t = idx * 100e-9 + 0.1e-9
        t_idx = np.argmin(np.abs(t_list - it_start_t))
        state = result.states[t_idx]
        z_val = qutip.expect(sim.sz, state).real
        start_z_values.append(z_val)

    print("\n[STAGE 3 INSPECTION: Expectation <Z> at Start of Loop Iterations]:")
    for idx, z_val in enumerate(start_z_values):
        print(f"  Iteration {idx} (t = {idx * 100} ns): <Z> = {z_val:+.4f}")

    # -------------------------------------------------------------
    # STAGE 4: Mid-Pulse Dynamics Verification (Iteration 4 @ amp=0.5V)
    # -------------------------------------------------------------
    t_mid_it4 = 425e-9
    t_idx_mid4 = np.argmin(np.abs(t_list - t_mid_it4))
    state_mid4 = result.states[t_idx_mid4]
    z_mid4 = qutip.expect(sim.sz, state_mid4).real
    
    print(f"\n[STAGE 4 INSPECTION]: <Z> at t = 425 ns (mid-pulse of amp=0.5V) = {z_mid4:+.4f}")
    assert z_mid4 < 0.95, f"Stage 4 Failed: Qubit state did not rotate under drive (<Z> = {z_mid4:.4f})"
    print("✓ STAGE 4 PASSED: Active pulse induced expected state rotation.\n")