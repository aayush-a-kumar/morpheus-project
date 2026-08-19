from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qblox_sim.simulator import QbloxQutipSimulator, SimulationResult


@pytest.fixture
def simulator():
    """Sets up a baseline simulator instance with dummy physics."""
    config_dict = {"dt": 1e-9}  # 1 ns grid
    sim = QbloxQutipSimulator(params=config_dict)
    sim.engine = MagicMock()
    # Mock the engine run to return a dummy result to prevent actual ODE solving
    dummy_res = MagicMock()
    dummy_res.states = [MagicMock()] * 1000
    sim.engine.run.return_value = dummy_res
    return sim


def test_grid_chunking_extraction(simulator):
    """
    Verifies that pulses and acquisitions are correctly merged into active chunks,
    leaving idle gaps in between.
    """
    # Create two pulses separated by a 50 ns gap
    # Pulse 1: 0 to 20 ns. Pulse 2: 70 to 100 ns.
    pulses = pd.DataFrame(
        [
            {"abs_time": 0.0, "duration": 20e-9, "operation_hash": "p1"},
            {"abs_time": 70e-9, "duration": 30e-9, "operation_hash": "p2"},
        ]
    )

    # Acquisition overlaps slightly with Pulse 2: 90 to 120 ns
    acquisitions = pd.DataFrame(
        [{"abs_time": 90e-9, "duration": 30e-9, "operation_hash": "a1"}]
    )

    # Execute processed simulation
    with patch.object(simulator, "_process_acquisitions", return_value=[]):
        simulator._simulate_processed(pulses, acquisitions)

    # Verify the chunks passed to the engine
    _, kwargs = simulator.engine.run.call_args
    chunks = kwargs.get("chunks")

    assert chunks is not None
    # Expected chunks based on 1 ns dt:
    # 1. Active: 0 to 20
    # 2. Idle: 20 to 70
    # 3. Active (Merged): 70 to 120
    assert chunks == [("active", 0, 20), ("idle", 20, 70), ("active", 70, 120)]


def test_statistical_loop_bypass(simulator):
    """
    Verifies that a pure statistical loop bypasses shot-by-shot unrolling
    and instead passes the 'shots' parameter down exactly once.
    """
    pulses = pd.DataFrame(
        [{"abs_time": 0.0, "duration": 10e-9, "operation_hash": "p1"}]
    )
    acquisitions = pd.DataFrame([])

    # Define a pure statistical loop (no sweep domain, 1000 repetitions)
    loops = [
        {
            "t_start": 0.0,
            "iteration_duration": 100e-9,
            "repetitions": 1000,
            "is_statistical": True,
            "domain": {},
        }
    ]

    # Track calls to _simulate_processed
    with patch.object(simulator, "_simulate_processed") as mock_sim_proc:
        mock_sim_proc.return_value = {
            "result": SimulationResult([], np.array([]), []),
            "t_list": np.array([]),
            "measurements": [],
        }

        simulator._simulate_shot_sweep(
            pulses=pulses,
            acquisitions=acquisitions,
            loops=loops,
            operations_dict={},
            initial_state=None,
        )

        # Assert the solver was only called exactly ONCE, not 1000 times
        assert mock_sim_proc.call_count == 1

        # Assert the 1000 shots were passed dynamically into the single call
        _, kwargs = mock_sim_proc.call_args
        assert kwargs["shots"] == 1000
