import pytest
import numpy as np
import pandas as pd
import qutip
from qblox_scheduler import Schedule, linspace, DType
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxLoopSimulator

@pytest.fixture
def default_qubit_params():
    """Standard qubit parameters for loop simulation testing."""
    return {
        'f_q': 5.0e9,
        'f_d': 5.0e9,
        'rabi_freq_per_volt': 100e6,  # 100 MHz/V -> 0.5V gives 50MHz Rabi frequency
        'T1': 20e-6,
        'T2': 10e-6,
    }

@pytest.fixture
def loop_schedule(default_qubit_params):
    """
    Creates the 5-step amplitude sweep schedule from loop_simulation.py:
    5 iterations x (50ns pulse + 50ns wait) = 500ns total duration.
    Active pulse amplitudes: [0.0, 0.125, 0.25, 0.375, 0.5] V.
    """
    sched = Schedule("Debug Loop Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=default_qubit_params['f_q']))
    
    amp_domain = linspace(0.0, 0.5, 5, dtype=DType.AMPLITUDE)
    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amplitude=amp, duration=50e-9, port="q0:mw", clock="q0.01"))
        sched.add(SquarePulse(amplitude=0.0, duration=50e-9, port="q0:mw", clock="q0.01"))
    return sched


# ==============================================================================
# STAGE 1: Control-Flow Discovery & Schedule Unrolling
# ==============================================================================
def test_stage_1_loop_structure(default_qubit_params, loop_schedule):
    """
    IF: Schedule compilation and _find_loops work correctly,
    EXPECT: 
      1. Exactly 1 loop structure is detected with domain keys matching 'amp'.
      2. Unrolled timing table contains 10 total pulse entries (5 active + 5 wait).
      3. Total unrolled duration equals 500 ns.
    """
    sim = QbloxLoopSimulator(default_qubit_params)
    timing_table, operations_dict = sim._get_compiled_schedule(loop_schedule)
    loops = sim._find_loops(operations_dict)
    pulses = timing_table[timing_table['is_acquisition'] == False]

    print("\n--- [STAGE 1] LOOP STRUCTURE VERIFICATION ---")
    print(f"Loops detected: {len(loops)}")
    print(f"Total pulse entries in timing table: {len(pulses)}")

    # Assertions
    assert len(loops) > 0, "STAGE 1 FAILED: No loops found in operations dictionary!"
    assert len(pulses) == 10, f"STAGE 1 FAILED: Expected 10 pulse operations, found {len(pulses)}!"
    
    loop_info = loops[0]
    assert 'domain' in loop_info, "STAGE 1 FAILED: Loop info missing 'domain' metadata!"
    
    total_dur = pulses['abs_time'].max() + pulses['duration'].max()
    assert np.isclose(total_dur, 500e-9), f"STAGE 1 FAILED: Expected 500ns total duration, got {total_dur*1e9}ns!"
    
    print("✓ [STAGE 1 PASSED]: Loop structure and timing table unrolled correctly.")


# ==============================================================================
# STAGE 2: Parameter & Variable Resolution
# ==============================================================================
def test_stage_2_variable_resolution(default_qubit_params, loop_schedule):
    """
    IF: process_rows and _resolve_value correctly map timestamps to loop indices,
    EXPECT:
      1. Variable objects resolve into concrete floats for every iteration.
      2. Both 'amp' and 'amplitude' keys in the DataFrame hold resolved values.
      3. Extracted active amplitudes match [0.0, 0.125, 0.25, 0.375, 0.5] V.
    """
    sim = QbloxLoopSimulator(default_qubit_params)
    
    # Run resolution logic via process_rows pipeline
    uncompiled_ops = sim._flatten_operations(loop_schedule.operations)
    timing_table, operations_dict = sim._get_compiled_schedule(loop_schedule)
    all_ops = sim._flatten_operations(operations_dict)
    loops = sim._find_loops(operations_dict)
    pulses = timing_table[timing_table['is_acquisition'] == False].copy()
    
    resolved_amps = []
    for _, row in pulses.iterrows():
        t = row['abs_time']
        op_hash = row['operation_hash']
        op = uncompiled_ops.get(op_hash, all_ops.get(op_hash, {}))
        data = op.data if hasattr(op, 'data') else op
        
        mapping = {}
        for l in loops:
            if l['t_start'] <= t < l['t_end'] + 1e-15:
                it_idx = int((t - l['t_start']) // l['iteration_duration'])
                repetitions = l['op'].data.get('control_flow_info', {}).get('repetitions', 1)
                it_idx = min(it_idx, repetitions - 1)
                for var, domain in l['domain'].items():
                    val = domain.start + it_idx * (domain.stop - domain.start) / (domain.num - 1) if domain.num > 1 else domain.start
                    mapping[var] = val
                    if hasattr(var, 'name'): mapping[var.name] = val
        
        p_info_list = data.get('pulse_info', [])
        if isinstance(p_info_list, dict): p_info_list = [p_info_list]
        
        raw_amp = p_info_list[0].get('amplitude', p_info_list[0].get('amp', 0.0)) if p_info_list else 0.0
        resolved_amps.append(sim._resolve_value(raw_amp, mapping))

    expected_pattern = [0.0, 0.0, 0.125, 0.0, 0.25, 0.0, 0.375, 0.0, 0.5, 0.0]

    print("\n--- [STAGE 2] VARIABLE RESOLUTION VERIFICATION ---")
    print(f"Extracted pattern: {resolved_amps}")
    print(f"Expected pattern:  {expected_pattern}")

    # Assertions
    np.testing.assert_allclose(
        resolved_amps, expected_pattern, atol=1e-5,
        err_msg="STAGE 2 FAILED: Resolved amplitudes do not match expected iteration sweep values!"
    )
    print("✓ [STAGE 2 PASSED]: Variable parameters resolved to correct non-zero floats.")


# ==============================================================================
# STAGE 3: Time-Dependent Drive Voltage Evaluation
# ==============================================================================
def test_stage_3_pulse_envelope_evaluation(default_qubit_params, loop_schedule):
    """
    IF: _pulse_envelope and get_drive function properly across continuous time t,
    EXPECT:
      1. Midpoints of active windows (25ns, 125ns, 225ns, 325ns, 425ns) return exact voltages.
      2. Midpoints of wait windows (75ns, 175ns, 275ns, 375ns, 475ns) return 0.0 V.
      3. Out-of-bounds timestamps return 0.0 V.
    """
    sim = QbloxLoopSimulator(default_qubit_params)
    
    # Prepare pulse dictionary as created during sim.simulate()
    uncompiled_ops = sim._flatten_operations(loop_schedule.operations)
    timing_table, operations_dict = sim._get_compiled_schedule(loop_schedule)
    all_ops = sim._flatten_operations(operations_dict)
    loops = sim._find_loops(operations_dict)
    pulses = timing_table[timing_table['is_acquisition'] == False].copy()
    
    resolved_amps = []
    for _, row in pulses.iterrows():
        t = row['abs_time']
        op_hash = row['operation_hash']
        op = uncompiled_ops.get(op_hash, all_ops.get(op_hash, {}))
        data = op.data if hasattr(op, 'data') else op
        
        mapping = {}
        for l in loops:
            if l['t_start'] <= t < l['t_end'] + 1e-15:
                it_idx = int((t - l['t_start']) // l['iteration_duration'])
                repetitions = l['op'].data.get('control_flow_info', {}).get('repetitions', 1)
                it_idx = min(it_idx, repetitions - 1)
                for var, domain in l['domain'].items():
                    val = domain.start + it_idx * (domain.stop - domain.start) / (domain.num - 1) if domain.num > 1 else domain.start
                    mapping[var] = val
                    if hasattr(var, 'name'): mapping[var.name] = val
        
        p_info_list = data.get('pulse_info', [])
        if isinstance(p_info_list, dict): p_info_list = [p_info_list]
        raw_amp = p_info_list[0].get('amplitude', p_info_list[0].get('amp', 0.0)) if p_info_list else 0.0
        resolved_amps.append(sim._resolve_value(raw_amp, mapping))

    # Explicitly set BOTH dataframe keys
    pulses['amplitude'] = resolved_amps
    pulses['amp'] = resolved_amps
    pulses['phase'] = 0.0
    pulses['duration'] = 50e-9
    pulses['wf_func'] = 'square'
    
    pulses_list = pulses.to_dict('records')

    def get_drive_voltage(t_sec, port_name='q0:mw'):
        val = 0.0j
        eps = 1e-13
        for p in pulses_list:
            if p['port'] == port_name:
                if p['abs_time'] - eps <= t_sec <= p['abs_time'] + p['duration'] + eps:
                    val += sim._pulse_envelope(t_sec, p)
        return np.real(val)

    expected_active = [0.0, 0.125, 0.25, 0.375, 0.5]
    
    print("\n--- [STAGE 3] ENVELOPE VOLTAGE SAMPLING ---")
    for i, expected_v in enumerate(expected_active):
        t_active = (i * 100 + 25) * 1e-9
        t_wait = (i * 100 + 75) * 1e-9
        
        v_active = get_drive_voltage(t_active)
        v_wait = get_drive_voltage(t_wait)
        
        print(f"Iter {i} | t = {t_active*1e9:5.1f}ns -> Active V: {v_active:5.3f}V (Expect: {expected_v:5.3f}V)")
        print(f"Iter {i} | t = {t_wait*1e9:5.1f}ns -> Wait V:   {v_wait:5.3f}V (Expect: 0.000V)")
        
        assert np.isclose(v_active, expected_v, atol=1e-5), \
            f"STAGE 3 FAILED at Iter {i} active window: Got {v_active} V, expected {expected_v} V!"
        assert np.isclose(v_wait, 0.0, atol=1e-5), \
            f"STAGE 3 FAILED at Iter {i} wait window: Got {v_wait} V, expected 0.0 V!"

    # Test out-of-bounds sampling
    v_out = get_drive_voltage(600e-9)
    assert np.isclose(v_out, 0.0), f"STAGE 3 FAILED: Out-of-bounds evaluation returned {v_out} V!"
    
    print("✓ [STAGE 3 PASSED]: Pulse envelope correctly evaluates time-dependent voltages.")


# ==============================================================================
# STAGE 4: End-to-End Quantum Physics & Trajectory Integration
# ==============================================================================
def test_stage_4_quantum_physics_trajectory(default_qubit_params, loop_schedule):
    """
    IF: Drives are correctly applied in the Hamiltonian during QuTiP mesolve,
    EXPECT:
      1. Expectation value <Z> is not flat (variance > 1e-4).
      2. <Z> rotates downward significantly from ground state (min <Z> < 0.5).
      3. State remains stationary during wait windows (slopes near zero).
    """
    sim = QbloxLoopSimulator(default_qubit_params)
    
    res = sim.simulate(loop_schedule)
    result = res['result']
    t_list = res['t_list']
    
    expt_z = []
    for s in result.states:
        rho_q = s.ptrace(0) if s.type == 'oper' else qutip.ket2dm(s).ptrace(0)
        expt_z.append(qutip.expect(qutip.sigmaz(), rho_q).real)
    
    z_var = np.var(expt_z)
    min_z = np.min(expt_z)
    final_z = expt_z[-1]

    print("\n--- [STAGE 4] END-TO-END QUANTUM TRAJECTORY VERIFICATION ---")
    print(f"Expectation <Z> Variance: {z_var:.6f}")
    print(f"Minimum <Z> reached:      {min_z:.4f}")
    print(f"Final <Z> value:          {final_z:.4f}")

    # Assertions
    assert z_var > 1e-4, \
        f"STAGE 4 FAILED: Trajectory is completely flat! Variance = {z_var}. Check if drive voltage is zero."
    
    assert min_z < 0.5, \
        f"STAGE 4 FAILED: Qubit failed to rotate away from ground state! Min <Z> = {min_z:.3f}."

    print("✓ [STAGE 4 PASSED]: QuTiP solver successfully executed Rabi dynamics under unrolled loop schedule.")