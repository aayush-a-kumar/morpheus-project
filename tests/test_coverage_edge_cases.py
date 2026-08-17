# SPDX-License-Identifier: LicenseRef-Proprietary
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import Measure
from qblox_scheduler.resources import ClockResource

from qblox_sim.acquisitions import (
    ThresholdedAcquisitionHandler,
    TraceAcquisitionHandler,
)
from qblox_sim.config import QubitConfig, ResonatorConfig, SimulationConfig
from qblox_sim.engine import QuTiPEngine
from qblox_sim.physics import QuantumSystem
from qblox_sim.signals import ScheduleSignalProvider, extract_amplitude
from qblox_sim.simulator import (
    QbloxQ1Simulator,
    QbloxQutipSimulator,
    SimulationResult,
)


# -----------------------------------------------------------------------------
# 1. acquisitions.py: Rotation, Thresholding, and Out-of-Bounds Trace Fallbacks
# -----------------------------------------------------------------------------
def test_thresholded_acquisition_with_rotation_and_threshold():
    handler = ThresholdedAcquisitionHandler()
    cfg = SimulationConfig()
    state = qutip.basis(3, 1)

    # Test degree rotation and non-zero threshold branch
    acq_info = {"acq_rotation": 180.0, "acq_threshold": -0.01}
    a_op = qutip.destroy(5)

    res = handler.process(
        state=state,
        t_list=np.linspace(0, 100e-9, 10),
        states=[state] * 10,
        acq_time=10e-9,
        acq_duration=50e-9,
        acq_info=acq_info,
        cfg=cfg,
        a_op=a_op,
        ad_op=a_op.dag(),
    )
    assert "outcome" in res
    assert res["value"] in (0, 1)


def test_trace_acquisition_fallback_when_empty_indices():
    handler = TraceAcquisitionHandler()
    cfg = SimulationConfig()
    state = qutip.basis(3, 0)
    a_op = qutip.destroy(3)

    # Request trace at time outside simulated t_list window
    res = handler.process(
        state=state,
        t_list=np.linspace(0, 10e-9, 10),
        states=[state] * 10,
        acq_time=100e-9,
        acq_duration=50e-9,
        acq_info={},
        cfg=cfg,
        a_op=a_op,
        ad_op=a_op.dag(),
    )
    assert len(res["value"]) > 0


# -----------------------------------------------------------------------------
# 2. signals.py: DRAG Envelopes, Series Input, and Grid Bounds
# -----------------------------------------------------------------------------
def test_signals_drag_pulse_and_edge_cases():
    # Test extract_amplitude with pandas Series and invalid objects
    s = pd.Series({"amplitude": 0.8})
    assert extract_amplitude(s) == 0.8
    # Suppress static type checker warning to test runtime fallback on non-mapping types
    assert extract_amplitude("invalid_input", default=0.2) == 0.2  # type: ignore[arg-type]

    # DRAG pulse generation
    t_list = np.linspace(0, 100e-9, 100)
    drag_pulse = [
        {
            "port": "q0:mw",
            "abs_time": 10e-9,
            "duration": 40e-9,
            "amp": 0.5,
            "phase": 45.0,
            "wf_func": "drag",
            "beta": 0.1,
            "sigma": 10e-9,
        }
    ]
    provider = ScheduleSignalProvider(drag_pulse)
    drives = provider.get_drives(t_list)
    assert "q0:mw" in drives
    assert np.any(np.imag(drives["q0:mw"]))

    # Edge cases: 1-point time grid and out-of-bounds pulse window
    assert provider.get_drives(np.array([0.0])) == {}

    out_of_bounds_pulse = [
        {"port": "q0:mw", "abs_time": 500e-9, "duration": 10e-9, "amp": 0.5}
    ]
    p_out = ScheduleSignalProvider(out_of_bounds_pulse)
    drives_out = p_out.get_drives(t_list)
    assert np.all(drives_out.get("q0:mw", 0) == 0)


# -----------------------------------------------------------------------------
# 3. physics.py: Two-level Qubit (N_q=2) Pauli Operators & Dephasing (T_2)
# -----------------------------------------------------------------------------
def test_physics_qubit_dim_2_and_t2_dephasing():
    # T2 < 2*T1 triggers pure dephasing collapse operator (gamma_phi > 0)
    cfg = SimulationConfig(
        qubits={"q0": QubitConfig(N_q=2, T1=10e-6, T2=5e-6)},
        resonators={"q0": ResonatorConfig(N_res=2, kappa=1e6)},
    )
    system = QuantumSystem(cfg)

    # Exercise strict 2-level Pauli matrices
    assert system.sx["q0"].shape == (4, 4)
    assert system.sy["q0"].shape == (4, 4)
    assert system.sz["q0"].shape == (4, 4)

    # Verify 3 collapse operators built: T1 decay, T2 dephasing, resonator kappa
    c_ops = system.build_collapse_operators()
    assert len(c_ops) == 3


# -----------------------------------------------------------------------------
# 4. engine.py: Quadrature Drives (Q-component and Resonator Drives)
# -----------------------------------------------------------------------------
def test_engine_quadrature_drives():
    cfg = SimulationConfig(
        qubits={"q0": QubitConfig(N_q=2)}, resonators={"q0": ResonatorConfig(N_res=2)}
    )
    system = QuantumSystem(cfg)
    engine = QuTiPEngine()

    t_list = np.linspace(0, 5e-9, 5)
    # Drives containing imaginary (Q) components
    drives = {
        "q0:mw": np.ones_like(t_list, dtype=complex) * 0.1j,
        "q0:res": np.ones_like(t_list, dtype=complex) * (0.05 + 0.05j),
    }
    init_state = system.get_default_initial_state()
    res = engine.run(
        system=system, drives=drives, t_list=t_list, initial_state=init_state
    )
    assert len(res.states) == 5


# -----------------------------------------------------------------------------
# 5. simulator.py: Result Validation Exceptions & Cluster Context Management
# -----------------------------------------------------------------------------
def test_simulation_result_expectation_exceptions():
    res_no_sys = SimulationResult(states=[], t_list=np.array([]), measurements=[])
    with pytest.raises(ValueError, match="No QuantumSystem attached"):
        res_no_sys.get_expectation("sz", "q0")

    cfg = SimulationConfig()
    sys = QuantumSystem(cfg)
    res_sys = SimulationResult(
        states=[], t_list=np.array([]), measurements=[], system=sys
    )

    with pytest.raises(ValueError, match="not found on QuantumSystem"):
        res_sys.get_expectation("non_existent_op", "q0")

    with pytest.raises(ValueError, match="not found in 'sz'"):
        res_sys.get_expectation("sz", "q99")


def test_q1simulator_context_manager():
    params = {"qubits": {"q0": {"N_q": 2}}, "resonators": {"q0": {"N_res": 2}}}
    with QbloxQ1Simulator(params) as sim:
        assert sim.cluster is not None
    assert sim.cluster is None


# -----------------------------------------------------------------------------
# 6. simulator.py: Uncompiled Schedule Fallback Compiler Path
# -----------------------------------------------------------------------------
def test_uncompiled_schedule_fallback_compiler(default_qubit_params):
    """Triggers _get_compiled_schedule exception block via PropertyMock."""
    sim = QbloxQutipSimulator(default_qubit_params)

    sched = Schedule("Uncompiled Sched")
    sched.add_resource(ClockResource(name="q0.ro", freq=6.0e9))
    sched.add(Measure("q0", clock="q0.ro"))

    # Use patch.object on PropertyMock to simulate uncompiled schedule exception cleanly
    with patch.object(
        Schedule,
        "timing_table",
        new_callable=PropertyMock,
        side_effect=AttributeError("Not compiled"),
    ):
        res = sim.simulate(sched)
        assert res is not None


# -----------------------------------------------------------------------------
# 7. simulator.py: Symbolic Value Resolution (_resolve_value)
# -----------------------------------------------------------------------------
class MockSubstituteVar:
    """Mock variable implementing .substitute()."""

    def __init__(self, name):
        self.name = name

    def substitute(self, mapping):
        return mapping.get(self.name, 0.5)


class MockNamedVar:
    """Mock variable with only a .name attribute (no .substitute())."""

    def __init__(self, name):
        self.name = name


def test_resolve_value_symbolic_objects(default_qubit_params):
    sim = QbloxQutipSimulator(default_qubit_params)
    var1 = MockSubstituteVar("var1")
    var2 = MockNamedVar("var2")

    mapping = {"var1": 0.1, "var2": 0.2}

    # Test substitute branch
    assert sim._resolve_value(var1, mapping) == 0.1

    # Test name mapping fallback branch
    assert sim._resolve_value(var2, mapping) == 0.2


# -----------------------------------------------------------------------------
# 8. simulator.py: Hardware Adapter Safe-Data Resampling & Measurement Window Tuple Formats
# -----------------------------------------------------------------------------
def test_q1simulator_interpolation_and_acq_tuple_formats(default_qubit_params):
    """Exercises np.interp resampling in safe_data and nested tuple formats in get_measurements."""
    with QbloxQ1Simulator(default_qubit_params) as sim:
        assert sim.cluster is not None

        # Mock hardware output with mismatched array length (10 points vs simulation grid)
        qcm = sim.cluster.get_connected_modules()[sim.drive_mod]
        qcm.out0 = MagicMock()
        qcm.out0.t_max = 100
        qcm.out0.t_min = 0
        qcm.out0.sample_rate = 1
        qcm.out0.data = np.ones(10)  # Shorter than default t_list length

        # Mock readout module with nested tuple acquisition windows: [((start, duration), ...)]
        qrm = sim.cluster.get_connected_modules()[sim.readout_mod]
        qrm.sequencers[sim.readout_seq].acq_windows = [((10, 50), 0)]

        res = sim.simulate()
        assert res is not None


# -----------------------------------------------------------------------------
# 9. simulator.py: Q1Simulator Solver Exception Path
# -----------------------------------------------------------------------------
def test_q1simulator_solver_failure_exception(default_qubit_params):
    """Triggers solver failure try-except block in QbloxQ1Simulator.simulate."""
    with QbloxQ1Simulator(default_qubit_params) as sim:
        # Mock engine.run to raise an exception
        sim.engine.run = MagicMock(
            side_effect=RuntimeError("Solver Convergence Failed")
        )

        # Run simulation; engine exception is caught, handled gracefully, and measurements extracted
        res = sim.simulate()
        assert res["result"].states == []
