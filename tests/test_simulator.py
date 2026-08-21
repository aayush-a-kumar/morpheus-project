# tests/test_simulator.py
# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from qblox_scheduler import Schedule
from qblox_scheduler.operations import SquarePulse
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import linspace
from qblox_scheduler.resources import ClockResource

from qblox_sim.physics import QuantumSystem
from qblox_sim.simulator import QbloxQ1Simulator, QbloxQutipSimulator, SimulationResult


@pytest.fixture
def base_simulator(default_qubit_params):
    """Provides a fast simulator instance with solver execution mocked out."""
    sim = QbloxQutipSimulator(default_qubit_params)
    sim.engine.run = MagicMock()
    return sim


@pytest.mark.unit
def test_loop_unrolling_exact_math(base_simulator):
    """Verify that schedule loops resolve symbolic variables into exact amplitude floats."""
    f_q = base_simulator.cfg.qubits["q0"].f_q
    sched = Schedule("Diagnostic Sweep")
    sched.add_resource(ClockResource(name="q0.01", freq=f_q))

    # 3-step sweep: 0.0, 0.25, 0.5
    amp_domain = linspace(0.0, 0.5, 3, dtype=DType.AMPLITUDE)

    with sched.loop(amp_domain) as amp:
        sched.add(SquarePulse(amp=amp, duration=50e-9, port="q0:mw", clock="q0.01"))

    # Mock the internal processed simulator to capture the unrolled DataFrames
    captured_pulses = []
    with patch.object(base_simulator, "_simulate_processed") as mock_proc:
        mock_proc.return_value = {
            "result": SimulationResult([], np.array([]), []),
            "t_list": np.array([]),
            "measurements": [],
        }
        base_simulator.simulate(sched)

        # Extract the DataFrame from every iteration shot passed to the solver
        for call_args in mock_proc.call_args_list:
            pulses_df = call_args.args[0]
            if not pulses_df.empty:
                captured_pulses.append(pulses_df.iloc[0]["amp"])

    # Strictly assert the loop unrolled the exact float domain (No Type 1 '< 1.0' bounds)
    np.testing.assert_allclose(captured_pulses, [0.0, 0.25, 0.5], atol=1e-12)


@pytest.mark.unit
def test_statistical_loop_bypass(base_simulator):
    """
    Verifies that a pure statistical loop bypasses shot-by-shot unrolling
    and instead passes the 'shots' parameter down exactly once.
    """
    pulses = pd.DataFrame(
        [{"abs_time": 0.0, "duration": 10e-9, "operation_hash": "p1"}]
    )
    acquisitions = pd.DataFrame([])

    # Directly mock the loop structure scanned by the simulator
    loops = [
        {
            "t_start": 0.0,
            "iteration_duration": 100e-9,
            "repetitions": 1000,
            "is_statistical": True,
            "domain": {},
        }
    ]

    with patch.object(base_simulator, "_simulate_processed") as mock_proc:
        mock_proc.return_value = {
            "result": SimulationResult([], np.array([]), []),
            "t_list": np.array([]),
            "measurements": [],
        }

        # Bypass the Schedule builder and inject directly into the execution step
        base_simulator._simulate_shot_sweep(
            pulses=pulses,
            acquisitions=acquisitions,
            loops=loops,
            operations_dict={},
            initial_state=None,
        )

        # Ensure we didn't unroll 1000 identical iterations
        assert mock_proc.call_count == 1

        # Ensure the 1000 shots were securely passed down to the statistical handlers
        _, kwargs = mock_proc.call_args
        assert kwargs["shots"] == 1000


@pytest.mark.edge_case
def test_simulation_result_expectation_exceptions():
    """Verify the result container fails safely when operators or systems are missing."""
    system = QuantumSystem(QbloxQutipSimulator({"dt": 1e-9}).cfg)
    res = SimulationResult(
        states=[], t_list=np.array([]), measurements=[], system=system
    )

    with pytest.raises(ValueError, match="not found on QuantumSystem"):
        res.get_expectation("non_existent_op", "q0")

    with pytest.raises(ValueError, match="not found in 'sz'"):
        res.get_expectation("sz", "invalid_qubit")

    res_no_sys = SimulationResult(states=[], t_list=np.array([]), measurements=[])
    with pytest.raises(ValueError, match="No QuantumSystem attached"):
        res_no_sys.get_expectation("sz", "q0")


@pytest.mark.edge_case
@patch("qblox_sim.simulator.Cluster")
def test_q1simulator_context_and_solver_failure(mock_cluster, default_qubit_params):
    """Verify hardware adapter handles ContextManager exit and internal solver crashes safely."""
    # Test Context Manager closure
    with QbloxQ1Simulator(default_qubit_params) as sim:
        assert sim.cluster is not None
    assert sim.cluster is None  # Ensures .close() was called

    # Test solver crash fallback
    with QbloxQ1Simulator(default_qubit_params) as sim_crash:
        sim_crash.engine.run = MagicMock(
            side_effect=RuntimeError("Solver Convergence Failed")
        )

        # Should catch the internal solver error, print to stdout, and return an empty result
        res = sim_crash.simulate()
        assert res["result"].states == []
        assert len(res["measurements"]) == 0
