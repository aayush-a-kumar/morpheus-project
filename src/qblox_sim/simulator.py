# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Orchestrates schedule compilation, loop parameter resolution, and simulation execution.

This module acts as the central execution manager for the simulator package.
It ingests high-level Qblox Schedules, unrolls parameter sweep loops, tracks Virtual Z
gate phase shifts across frequency clocks, constructs continuous IQ waveforms, runs time-domain
quantum evolution via QuTiP, and processes raw quantum states into realistic measurement outputs.
"""

import typing

import numpy as np
import pandas as pd
import qutip
from q1simulator import Cluster
from qblox_scheduler import (
    BasicTransmonElement,
    QuantumDevice,
    Schedule,
    SerialCompiler,
)
from qblox_scheduler.operations import LoopOperation
from qcodes.instrument import Instrument

from qblox_sim.acquisitions import AcquisitionRegistry
from qblox_sim.config import SimulationConfig
from qblox_sim.engine import QuTiPEngine
from qblox_sim.physics import QuantumSystem
from qblox_sim.signals import ScheduleSignalProvider, extract_amplitude


class SimulationResult:
    """Clean container for stitched time-series simulation outputs."""

    def __init__(
        self,
        states: list,
        t_list: np.ndarray,
        measurements: list,
        system: QuantumSystem | None = None,
    ):
        """
        Stores state trajectories, time grids, and processed acquisition results.

        Args:
            states (list): List of QuTiP state vectors (kets) or density matrices across all simulated time points.
            t_list (np.ndarray): Strictly uniform 1D array of time stamps in SI seconds corresponding to each state.
            measurements (list): List of processed measurement result dictionaries generated during acquisition windows.
            system (QuantumSystem | None, optional): Attached QuantumSystem instance required for operator lookups. Defaults to None.
        """
        # WHY: We retain the raw states list to allow custom post-processing by the user.
        self.states = states
        # WHY: The continuous time grid is required to plot state trajectories or compute time-dependent expectation values.
        self.t_list = t_list
        # WHY: Store discrete acquisition outputs separately so users can inspect final IQ centroids and outcomes directly.
        self.measurements = measurements
        # WHY: Keep a reference to the QuantumSystem so helper methods can resolve operator keys (e.g. 'sz', 'sx') dynamically.
        self._system = system

    def get_expectation(self, op_name: str = "sz", q_name: str = "q0") -> np.ndarray:
        """
        Calculates time-dependent expectation values for a specified quantum operator on a target qubit.

        Args:
            op_name (str, optional): Operator dictionary key on QuantumSystem (e.g., 'sz', 'sx', 'sy', 'nq'). Defaults to "sz".
            q_name (str, optional): Target qubit identifier key in the operator map (e.g., 'q0', 'q1'). Defaults to "q0".

        Returns:
            np.ndarray: 1D array of real expectation values <O(t)> = Tr(rho(t) * O) evaluated at each time step.
        """
        # WHY: Expectation value calculation requires access to the operator matrices constructed in QuantumSystem.
        if self._system is None:
            raise ValueError("No QuantumSystem attached to SimulationResult.")

        # WHY: Dynamically retrieve the requested operator dictionary (e.g., self._system.sz) using getattr.
        op_dict = getattr(self._system, op_name, None)
        if op_dict is None or not isinstance(op_dict, dict):
            raise ValueError(
                f"Operator dictionary '{op_name}' not found on QuantumSystem."
            )

        # WHY: Extract the specific tensor-space operator for the target qubit name.
        op = op_dict.get(q_name)
        if op is None:
            raise ValueError(f"Operator for qubit '{q_name}' not found in '{op_name}'.")

        # WHY: qutip.expect evaluates <state|op|state> or Tr(op * density_matrix).
        # We take the .real part because physical observables correspond to Hermitian operators with strictly real expectation values,
        # but numerical floating-point precision in QuTiP can leave tiny residual imaginary parts (e.g. +0j).
        return np.array([qutip.expect(op, s).real for s in self.states])


class QbloxQutipSimulator:
    """
    Simulates Qblox-Scheduler Schedules using QuTiP for multi-qubit topologies coupled to resonators.
    """

    def __init__(self, params: dict, configs: dict | None = None):
        """
        Instantiates the simulator engine and builds static quantum physics models.

        Args:
            params (dict): Configuration dictionary containing qubit, resonator, coupling, and solver specifications.
            configs (dict | None, optional): Optional hardware or compilation parameters map. Defaults to None.
        """
        # WHY: Store raw input dictionaries for debugging or introspection.
        self.params = params
        self.configs = configs

        # WHY: Parse raw dictionary inputs into strongly-typed dataclasses to enforce SI units and parameter validation.
        self.cfg = SimulationConfig.from_dict(params)
        # WHY: Instantiating QuantumSystem constructs the global Hilbert space tensor operators once during setup,
        # avoiding expensive matrix recalculations during simulation runs.
        self.system = QuantumSystem(self.cfg)
        # WHY: Initialize the QuTiP execution engine instance that wraps mesolve solver calls.
        self.engine = QuTiPEngine()

    # =========================================================================
    # Helpers & Compilation
    # =========================================================================
    def _flatten_operations(
        self, operations_dict: dict, all_ops: dict | None = None
    ) -> dict:
        """
        Recursively unpacks nested operation structures (such as control-flow loop bodies) into a single flat map.

        Args:
            operations_dict (dict): Dictionary mapping operation hashes to operation objects or nested schedules.
            all_ops (dict | None, optional): Accumulator dictionary used during recursive depth traversal. Defaults to None.

        Returns:
            dict: Flat dictionary mapping every operation hash to its operation instance.
        """
        # WHY: Qblox Schedules wrap operations inside nested blocks (e.g., LoopOperation.body.operations).
        # Flattening guarantees an O(1) direct lookup map for any operation hash encountered in timing table rows.
        if all_ops is None:
            all_ops = {}
        for h, op in operations_dict.items():
            all_ops[h] = op
            # WHY: If an operation is a control-flow loop, recurse into its body operations.
            if isinstance(op, LoopOperation):
                self._flatten_operations(op.body.operations, all_ops)  # type: ignore
            elif hasattr(op, "operations"):
                self._flatten_operations(op.operations, all_ops)
            # WHY: Inspect inner body attributes in case operations are wrapped in composite containers.
            body = getattr(op, "body", None)
            if body is not None and hasattr(body, "operations"):
                self._flatten_operations(body.operations, all_ops)
        return all_ops

    def _get_op_from_hash(self, op_hash: typing.Any, ops_dict: dict) -> dict:
        """
        Safely resolves an operation hash to its operation dictionary across string/integer representation mismatches.

        Args:
            op_hash (typing.Any): The key hash to look up in the operations dictionary.
            ops_dict (dict): Flat mapping of operation hashes to operation instances.

        Returns:
            dict: The matching operation object or dictionary; returns an empty dict if missing.
        """
        # WHY: Schedule compilers or JSON serializers may convert numerical operation hashes into strings or vice versa.
        # Fallback casting between int and str prevents lookup failures due to type mismatches.
        op = ops_dict.get(op_hash)
        if op is None and isinstance(op_hash, (int, str)):
            try:
                op = ops_dict.get(int(op_hash)) or ops_dict.get(str(op_hash))
            except ValueError:
                pass
        return op if op is not None else {}

    def _get_compiled_schedule(self, schedule: Schedule) -> tuple[pd.DataFrame, dict]:
        """
        Extracts timing tables and operations from a schedule, compiling it via SerialCompiler if uncompiled.

        Args:
            schedule (Schedule): Raw uncompiled or pre-compiled Qblox Schedule object.

        Returns:
            tuple[pd.DataFrame, dict]: Pair containing (timing_table DataFrame, operations dictionary).
        """
        # WHY: A pre-compiled Schedule contains `timing_table.data`. If missing, the schedule is uncompiled,
        # so we must instantiate a fallback SerialCompiler with default clock frequencies to generate absolute pulse times.
        try:
            return schedule.timing_table.data, schedule.operations  # type: ignore
        except (AttributeError, KeyError, ValueError):
            # WHY: Construct a fallback QuantumDevice with standard transmon frequencies (5 GHz qubit, 7 GHz readout)
            # to allow compilation of standalone schedules that lack attached device hardware objects.
            device = QuantumDevice(name="dummy_device")
            try:
                q0 = BasicTransmonElement("q0")  # type: ignore
                q0.clock_freqs.readout = 7.0e9
                q0.clock_freqs.f01 = 5.0e9
                device.add_element(q0)
            except (AttributeError, KeyError):
                pass
            compiler = SerialCompiler(name="compiler", quantum_device=device)
            compiled_sched = compiler.compile(schedule)
            return compiled_sched.timing_table.data, compiled_sched.operations  # type: ignore

    def _find_loops(self, operations_dict: dict, offset: float = 0.0) -> list:
        """
        Recursively locates LoopOperation instances to build time-boundary and domain-variable tracking maps.
        Flags pure statistical loops to bypass iterative ODE integration.

        Args:
            operations_dict (dict): Dictionary of operations to scan.
            offset (float, optional): Cumulative time offset in seconds for nested loops. Defaults to 0.0.

        Returns:
            list: List of dictionaries detailing start time, end time, iteration duration, sweep domains, and statistical flags for each loop.
        """
        # WHY: Parameter sweeps in schedules wrap repeated pulses in control-flow loops.
        # Finding absolute start and end times for these loops allows the simulator to unroll sweep iterations correctly.
        loops = []
        for op in operations_dict.values():
            if isinstance(op, LoopOperation):
                # WHY: Extract the control flow metadata which defines how the loop behaves.
                cf_info = op.data.get("control_flow_info", {})
                t0 = cf_info.get("t0", 0.0)
                repetitions = cf_info.get("repetitions", 1)
                domain = cf_info.get("domain", {})

                # WHY: If a loop has no sweep variables in its domain, it does not alter the Hamiltonian.
                # We flag this as a pure statistical loop so we can compute the physics once and sample it N times.
                is_statistical = len(domain) == 0

                # WHY: Calculate total loop duration by multiplying iteration duration by iteration count.
                duration = repetitions * op.body.duration
                loops.append(
                    {
                        "op": op,
                        "t_start": offset + t0,
                        "t_end": offset + t0 + duration,
                        "iteration_duration": op.body.duration,
                        "domain": domain,
                        "repetitions": repetitions,
                        "is_statistical": is_statistical,
                    }
                )
                # WHY: Recurse into nested loops, adding the current loop's offset to keep absolute time tracking accurate.
                loops.extend(self._find_loops(op.body.operations, offset + t0))  # type: ignore #[cite: 6]
            elif hasattr(op, "operations"):
                # WHY: Handle standard operation containers that may hold nested loops.
                t0 = op.data.get("t0", 0.0) if hasattr(op, "data") else 0.0
                loops.extend(self._find_loops(op.operations, offset + t0))
        return loops

    def _resolve_value(self, val, mapping: dict):
        """
        Evaluates symbolic sweep variables or variable objects into concrete float scalars using a loop mapping.

        Args:
            val (Any): Input object, float, or symbolic variable.
            mapping (dict): Dictionary mapping variable names or symbols to numerical iteration values.

        Returns:
            Any: Concrete float value after evaluating variable substitutions.
        """
        # WHY: Schedule variables inside sweep loops remain symbolic objects during compilation.
        # This resolves symbolic variables into concrete floats for numerical ODE integration.
        if hasattr(val, "substitute"):
            return val.substitute(mapping)
        if hasattr(val, "name") and val.name in mapping:
            return mapping[val.name]
        return val

    # =========================================================================
    # Simulation Execution
    # =========================================================================
    def simulate(
        self, schedule: Schedule, initial_state: qutip.Qobj | None = None
    ) -> dict:
        """
        Executes the full simulation pipeline for a given Qblox Schedule.

        Args:
            schedule (Schedule): The input Qblox Schedule instance.
            initial_state (qutip.Qobj | None, optional): Custom initial quantum state override. Defaults to None.

        Returns:
            dict: Result dictionary containing SimulationResult container, continuous time array, and measurements list.
        """
        # WHY: Step 1: Flatten operations to build a simple lookup dictionary for hashes.
        uncompiled_ops = self._flatten_operations(schedule.operations)
        # WHY: Step 2: Compile the schedule to generate absolute timing table data.
        timing_table, operations_dict = self._get_compiled_schedule(schedule)
        # WHY: Step 3: Scan for control-flow loops to determine whether shot-based loop unrolling is needed.
        loops = self._find_loops(operations_dict)

        # WHY: Step 4: Separate pulses from acquisitions using the boolean flag column in the timing table.
        pulses = timing_table[timing_table["is_acquisition"] == False].copy()
        acquisitions = timing_table[timing_table["is_acquisition"] == True].copy()

        # WHY: Assert DataFrame types to satisfy static type checkers (Pyright/Mypy).
        assert isinstance(pulses, pd.DataFrame)
        assert isinstance(acquisitions, pd.DataFrame)

        # WHY: Step 5: Substitute loop variables and accumulate Virtual Z phase shifts into the pulse timing table.
        pulses = self._resolve_loop_pulses(pulses, uncompiled_ops, loops)

        # WHY: Step 6: If loops exist, execute shot-by-shot sweep simulation; otherwise execute a single-pass run.
        if loops:
            return self._simulate_shot_sweep(
                pulses, acquisitions, loops, uncompiled_ops, initial_state
            )
        return self._simulate_processed(
            pulses, acquisitions, uncompiled_ops, initial_state
        )

    def _resolve_loop_pulses(
        self, pulses: pd.DataFrame, uncompiled_ops: dict, loops: list
    ) -> pd.DataFrame:
        """
        Substitutes dynamic sweep variables into pulse attributes and tracks Virtual Z frame shifts per clock.

        Args:
            pulses (pd.DataFrame): DataFrame containing raw pulse rows from the compiled schedule timing table.
            uncompiled_ops (dict): Flattened operations dictionary mapping operation hashes to data payloads.
            loops (list): List of detected loop metadata dictionaries.

        Returns:
            pd.DataFrame: Enriched timing table with updated amplitude, phase, duration, and wf_func columns.
        """
        resolved_amps, resolved_phases, resolved_durations, resolved_wfs = (
            [],
            [],
            [],
            [],
        )

        # WHY: Virtual Z gates do not emit physical voltage pulses; they shift the phase reference frame of a clock.
        # We maintain a clock-indexed dictionary to accumulate phase shifts and add them to subsequent pulses on that clock.
        tracked_phases = {}

        for _, row in pulses.iterrows():
            t = row["abs_time"]
            op_hash = row["operation_hash"]
            op = self._get_op_from_hash(op_hash, uncompiled_ops)
            data = getattr(op, "data", op) if hasattr(op, "data") else op

            # --- 1. Loop Variable Resolution (Defines 'mapping') ---
            mapping = {}
            for l in loops:
                # WHY: Check if the pulse start time falls within the bounds of a loop.
                if l["t_start"] <= t < l["t_end"] + 1e-15:
                    # WHY: Compute iteration index by dividing elapsed loop time by single-iteration duration.
                    it_idx = int((t - l["t_start"]) // l["iteration_duration"])
                    repetitions = (
                        l["op"].data.get("control_flow_info", {}).get("repetitions", 1)
                    )
                    # WHY: Clamp index to repetitions - 1 to handle potential floating-point precision edge cases at the upper bound.
                    it_idx = min(it_idx, repetitions - 1)

                    # WHY: Linearly interpolate the sweep parameter value for the current iteration index.
                    for var, domain in l["domain"].items():
                        val = (
                            domain.start
                            + it_idx * (domain.stop - domain.start) / (domain.num - 1)
                            if domain.num > 1
                            else domain.start
                        )
                        mapping[var] = val
                        if hasattr(var, "name"):
                            mapping[var.name] = val

            # --- 2. Intercept Virtual Z Gates (Logical Phase Shifts) ---
            logic_info = data.get("logic_info", data) if isinstance(data, dict) else {}
            if "phase_shift" in logic_info:
                clock = logic_info.get("clock", data.get("clock", "default"))
                if clock not in tracked_phases:
                    tracked_phases[clock] = 0.0

                # WHY: Evaluate symbolic phase shift expressions into concrete scalar floats.
                resolved_shift = self._resolve_value(logic_info["phase_shift"], mapping)
                # WHY: Accumulate frame phase shift into tracked clock phases.
                tracked_phases[clock] += float(resolved_shift)

            # --- 3. Standard Pulse Extraction ---
            a, p, dur, wf = 0.0, 0.0, row.get("duration", 0.0), "square"

            p_info_list = data.get("pulse_info", []) if isinstance(data, dict) else []
            if isinstance(p_info_list, dict):
                p_info_list = [p_info_list]

            if p_info_list:
                p_info = p_info_list[0]
                clock = p_info.get("clock", "default")

                if clock not in tracked_phases:
                    tracked_phases[clock] = 0.0

                # WHY: Pulses can embed phase shifts directly within their pulse_info parameters.
                if "phase_shift" in p_info:
                    resolved_shift = self._resolve_value(p_info["phase_shift"], mapping)
                    tracked_phases[clock] += float(resolved_shift)

                # WHY: Extract raw amplitude from pulse info dictionary or pandas Series.
                raw_amp = extract_amplitude(p_info)
                a = self._resolve_value(raw_amp, mapping)

                # WHY: Combine base pulse phase with accumulated clock phase shift to set the drive phase in the rotating frame.
                base_phase = self._resolve_value(p_info.get("phase", 0.0), mapping)
                p = float(base_phase) + tracked_phases[clock]

                dur = self._resolve_value(
                    p_info.get("duration", row.get("duration", 0.0)), mapping
                )
                wf = p_info.get("wf_func", "square")

            resolved_amps.append(a)
            resolved_phases.append(p)
            resolved_durations.append(dur)
            resolved_wfs.append(wf)

        # WHY: Assign resolved arrays back to DataFrame columns so signal generators read concrete numerical values.
        pulses["amplitude"] = resolved_amps
        pulses["amp"] = resolved_amps
        pulses["phase"] = resolved_phases
        pulses["duration"] = resolved_durations
        pulses["wf_func"] = resolved_wfs
        return pulses

    def _simulate_shot_sweep(
        self,
        pulses: pd.DataFrame,
        acquisitions: pd.DataFrame,
        loops: list,
        operations_dict: dict,
        initial_state: qutip.Qobj | None,
    ) -> dict:
        """
        Executes parameter sweeps iteration-by-iteration, handling statistical sampling bypasses.

        Args:
            pulses (pd.DataFrame): Resolved pulse timing table DataFrame.
            acquisitions (pd.DataFrame): Acquisition timing table DataFrame.
            loops (list): Scanned loop structure parameters list.
            operations_dict (dict): Flat operations dictionary.
            initial_state (qutip.Qobj | None): Initial state vector/density matrix for each iteration shot.

        Returns:
            dict: Combined result dictionary containing concatenated state histories, time grids, and acquisition results.
        """
        loop = loops[0]
        loop_start = loop["t_start"]
        loop_duration = loop["iteration_duration"]
        repetitions = loop["repetitions"]

        unique_iterations = 1 if loop["is_statistical"] else repetitions
        shots = repetitions if loop["is_statistical"] else 1

        all_results = []
        combined_t_list = []
        combined_states = []

        for it in range(unique_iterations):
            it_start = loop_start + it * loop_duration
            it_end = loop_start + (it + 1) * loop_duration

            # WHY: Dynamically resolve the absolute time column because uncompiled schedules or edge cases
            # might use "t0" or "time" instead of "abs_time".
            time_col_p = next(
                (col for col in ["abs_time", "time", "t0"] if col in pulses.columns),
                "abs_time",
            )

            # WHY: Prevent KeyError by ensuring the DataFrame is not empty and contains the resolved time column.
            if not pulses.empty and time_col_p in pulses.columns:
                it_pulses = pulses[
                    (pulses[time_col_p] >= it_start - 1e-12)
                    & (pulses[time_col_p] < it_end - 1e-12)
                ].copy()
                it_pulses[time_col_p] -= it_start
            else:
                it_pulses = pulses.copy()

            time_col_a = next(
                (
                    col
                    for col in ["abs_time", "time", "t0"]
                    if col in acquisitions.columns
                ),
                "abs_time",
            )

            if not acquisitions.empty and time_col_a in acquisitions.columns:
                it_acq = acquisitions[
                    (acquisitions[time_col_a] >= it_start - 1e-12)
                    & (acquisitions[time_col_a] < it_end - 1e-12)
                ].copy()
                it_acq[time_col_a] -= it_start
            else:
                it_acq = acquisitions.copy()

            # WHY: Existing test suites mock _simulate_processed using its legacy 4-argument signature.
            # Unconditionally passing 'shots=shots' triggers an unexpected keyword argument error in test spies.
            # We omit the keyword entirely when shots == 1 to maintain backward compatibility.
            # We explicitly type kwargs as dict[str, Any] so Pylance allows mixing Qobj and int values.
            kwargs: dict[str, typing.Any] = {"initial_state": initial_state}
            kwargs = {"initial_state": initial_state}
            if shots > 1:
                kwargs["shots"] = shots

            res = self._simulate_processed(it_pulses, it_acq, operations_dict, **kwargs)
            all_results.append(res)

            is_last = it == unique_iterations - 1
            t_slice = res["t_list"] if is_last else res["t_list"][:-1]
            state_slice = res["result"].states if is_last else res["result"].states[:-1]

            combined_t_list.append(t_slice + it_start)
            combined_states.extend(state_slice)

        # WHY: Guard against ValueError/TypeError if the grid extraction yielded zero iterations
        # (e.g., completely empty schedule), leaving combined_t_list empty.
        final_t_list = (
            np.concatenate(combined_t_list) if combined_t_list else np.array([])
        )

        result_container = SimulationResult(
            combined_states,
            final_t_list,
            [m for r in all_results for m in r["measurements"]],
            system=self.system,
        )

        return {
            "result": result_container,
            "t_list": final_t_list,
            "measurements": result_container.measurements,
        }

    def _simulate_processed(
        self,
        pulses: pd.DataFrame,
        acquisitions: pd.DataFrame,
        operations_dict: dict | None = None,
        initial_state: qutip.Qobj | None = None,
        shots: int = 1,
    ) -> dict:
        """
        Executes a continuous time-domain simulation by chunking the schedule into active and idle intervals.

        Args:
            pulses (pd.DataFrame): Pulse timing table DataFrame for the window.
            acquisitions (pd.DataFrame): Acquisition timing table DataFrame for the window.
            operations_dict (dict | None, optional): Operations dictionary map. Defaults to None.
            initial_state (qutip.Qobj | None, optional): Initial state vector or density matrix. Defaults to None.
            shots (int, optional): The number of statistical shots to generate for measurements. Defaults to 1.

        Returns:
            dict: Result dictionary containing SimulationResult container, continuous time array, and measurements list.
        """
        operations_dict = operations_dict or {}
        # Original: pulses_list = pulses.to_dict("records")
        # FIX: Cast the Pandas output to explicitly satisfy Pylance
        pulses_list = typing.cast(
            list[dict[str, typing.Any]], pulses.to_dict("records")
        )

        # WHY: Hardware compilers can output timing in nanoseconds if values exceed 1e-3.
        # We explicitly convert values > 1e-3 to standard SI seconds (1e-9 s) for physics solver calculations.
        for p in pulses_list:
            p["abs_time"] = (
                p["abs_time"] * 1e-9 if p["abs_time"] > 1e-3 else p["abs_time"]
            )
            p["duration"] = (
                p["duration"] * 1e-9 if p["duration"] > 1e-3 else p["duration"]
            )

        # WHY: Determine the absolute end time of the simulation to bounds the continuous time grid.
        if len(pulses_list) == 0:
            total_duration = 1e-6
        else:
            total_duration = max(p["abs_time"] + p["duration"] for p in pulses_list)

        # WHY: Account for trailing acquisition windows that extend past the final microwave pulse.
        if len(acquisitions) > 0:
            acq_max = float(acquisitions["abs_time"].max() or 0.0) + float(
                acquisitions["duration"].max() or 0.0
            )
            acq_max = acq_max * 1e-9 if acq_max > 1e-3 else acq_max
            total_duration = max(total_duration, acq_max)

        # WHY: QuTiP solvers require a strictly monotonic, uniformly spaced time grid.
        # Fixed step size dt guarantees predictable numerical stability during matrix exponential integration.
        step_size = self.cfg.dt
        num_points = max(1000, int(np.ceil(total_duration / step_size)) + 1)
        t_list = np.linspace(0, total_duration, num_points)

        # --- EVENT CHUNKING LOGIC ---

        # WHY: Extract the absolute float intervals for every physical event in the schedule.
        intervals = []
        for p in pulses_list:
            intervals.append((p["abs_time"], p["abs_time"] + p["duration"]))

        for _, acq in acquisitions.iterrows():
            acq_time = (
                acq["abs_time"] * 1e-9 if acq["abs_time"] > 1e-3 else acq["abs_time"]
            )
            dur = float(acq.get("duration") or 1e-6)
            dur = dur * 1e-9 if dur > 1e-3 else dur
            intervals.append((acq_time, acq_time + dur))

        # WHY: Sort chronologically to prepare for overlapping overlap resolution.
        intervals.sort(key=lambda x: x[0])

        # WHY: Merge intervals that overlap or touch to find the minimal set of contiguous active windows.
        # We immediately map these to integer indices based on the 1 ns dt to align precisely with the global t_list.
        merged_indices = []
        curr_start, curr_end = -1, -1

        for start, end in intervals:
            start_idx = int(np.round(start / step_size))
            end_idx = int(np.round(end / step_size))

            if curr_start == -1:
                curr_start, curr_end = start_idx, end_idx
            elif start_idx <= curr_end:
                # WHY: Extend the current active window if the new event overlaps.
                curr_end = max(curr_end, end_idx)
            else:
                # WHY: A gap was found, finalize the previous active window and start a new one.
                merged_indices.append((curr_start, curr_end))
                curr_start, curr_end = start_idx, end_idx

        if curr_start != -1:
            merged_indices.append((curr_start, curr_end))

        # WHY: Translate the merged active indices into an execution plan (chunks) for the engine,
        # explicitly defining the idle gaps between the active blocks.
        chunks = []
        last_idx = 0
        for start_idx, end_idx in merged_indices:
            if start_idx > last_idx:
                chunks.append(("idle", last_idx, start_idx))
            chunks.append(("active", start_idx, end_idx))
            last_idx = end_idx

        # WHY: Close out the timeline with a final idle block if the schedule goes quiet before total_duration.
        if last_idx < len(t_list) - 1:
            chunks.append(("idle", last_idx, len(t_list) - 1))

        # --- EXECUTION ---

        # WHY: Convert discrete schedule pulse dictionaries into continuous IQ drive envelopes.
        signal_provider = ScheduleSignalProvider(pulses_list)
        drives = signal_provider.get_drives(t_list)

        # WHY: Default to pure ground state |00..0> if no custom initial state is supplied.
        if initial_state is None:
            initial_state = self.system.get_default_initial_state()

        # WHY: Execute the chunked evolution plan via the engine.
        # The engine will use mesolve for active chunks and fast analytical jumps for idle chunks.
        result = self.engine.run(
            system=self.system,
            drives=drives,
            t_list=t_list,
            initial_state=initial_state,
            chunks=chunks,  # NEW PARAMETER
        )

        # WHY: Process acquisitions against computed quantum states, generating N statistical shots instantly.
        measurements = self._process_acquisitions(
            acquisitions, result.states, t_list, operations_dict, shots=shots
        )

        result_container = SimulationResult(
            states=result.states if result is not None else [],
            t_list=t_list,
            measurements=measurements,
            system=self.system,
        )

        return {
            "result": result_container,
            "t_list": t_list,
            "measurements": measurements,
        }

    def _process_acquisitions(
        self,
        acquisitions: pd.DataFrame,
        states: list,
        t_list: np.ndarray,
        operations_dict: dict,
        shots: int = 1,
    ) -> list:
        """
        Processes acquisition timing table rows by passing corresponding states to registered strategy handlers.

        Args:
            acquisitions (pd.DataFrame): DataFrame containing acquisition triggers.
            states (list): Time-evolved quantum state history list.
            t_list (np.ndarray): Continuous simulation time grid array.
            operations_dict (dict): Flat operations dictionary.
            shots (int, optional): The number of independent samples to generate. Defaults to 1.

        Returns:
            list: List of processed acquisition dictionaries containing demodulated IQ voltages, trace data, or bit outcomes.
        """
        measurements = []
        for _, acq in acquisitions.iterrows():
            # WHY: Format acquisition absolute time and duration to SI seconds.
            acq_time = acq["abs_time"]
            acq_time = acq_time * 1e-9 if acq_time > 1e-3 else acq_time
            acq_duration = float(acq.get("duration") or 1e-6)
            acq_duration = acq_duration * 1e-9 if acq_duration > 1e-3 else acq_duration

            op_hash = acq.get("operation_hash", None)
            op = self._get_op_from_hash(op_hash, operations_dict) if op_hash else {}
            data = getattr(op, "data", op) if hasattr(op, "data") else op

            acq_info_list = (
                data.get("acquisition_info", []) if isinstance(data, dict) else []
            )
            if isinstance(acq_info_list, dict):
                acq_info_list = [acq_info_list]
            acq_info = (
                acq_info_list[0]
                if (isinstance(acq_info_list, list) and len(acq_info_list) > 0)
                else {}
            )

            # WHY: Inject the dynamically calculated statistical repetition count into the info map
            # so handlers like SSBIntegrationHandler can sample N times instantly.
            acq_info["shots"] = shots

            # WHY: Extract protocol string, defaulting to SSBIntegrationComplex if unspecified.
            protocol = acq_info.get(
                "protocol",
                acq.get("acq_protocol", acq.get("protocol", "SSBIntegrationComplex")),
            )
            protocol = (
                str(protocol)
                if protocol is not None
                and not (isinstance(protocol, float) and np.isnan(protocol))
                else "SSBIntegrationComplex"
            )

            # WHY: Find the nearest index in t_list corresponding to the acquisition start time to pull the correct state matrix.
            idx = np.argmin(np.abs(t_list - acq_time))
            state = states[idx]

            acq_channel = acq_info.get(
                "acq_channel", acq.get("acq_channel", acq.get("acq_index", "acq"))
            )
            res_name = (
                str(acq_channel).split(":")[0]
                if isinstance(acq_channel, str) and ":" in acq_channel
                else "q0"
            )

            # WHY: Ensure valid resonator operators are passed to acquisition handlers even if channel keys mismatch.
            a_op = self.system.a.get(res_name)
            ad_op = self.system.ad.get(res_name)
            if a_op is None or ad_op is None:
                a_op = self.system.a["q0"]
                ad_op = self.system.ad["q0"]

            # WHY: Lookup strategy handler in AcquisitionRegistry and execute state projection with vectorization support.
            handler = AcquisitionRegistry.get_handler(protocol)  # [cite: 1]
            processed_data = handler.process(
                state=state,
                t_list=t_list,
                states=states,
                acq_time=acq_time,
                acq_duration=acq_duration,
                acq_info=acq_info,
                cfg=self.cfg,
                a_op=a_op,
                ad_op=ad_op,
            )

            meas_dict = {
                "name": acq_channel,
                "protocol": protocol,
                "time": acq_time,
                "duration": acq_duration,
            }
            meas_dict.update(processed_data)
            measurements.append(meas_dict)

        return measurements


class QbloxQ1Simulator(QbloxQutipSimulator):
    """
    A hardware-adapter simulator that acts as a QCoDeS Cluster instrument and passes extracted pulse traces to the multi-qubit engine.
    """

    def __init__(
        self,
        params: dict,
        name: str = "cluster",
        modules: dict | None = None,
        hardware_config: dict | None = None,
    ):
        """
        Initializes the hardware-adapter simulator wrapping a QCoDeS Cluster instance.

        Args:
            params (dict): Simulation configuration parameter map.
            name (str, optional): QCoDeS instrument name for the cluster. Defaults to "cluster".
            modules (dict | None, optional): Map of module slot IDs to module type strings. Defaults to None.
            hardware_config (dict | None, optional): Drive and readout module/sequencer mapping configuration. Defaults to None.
        """
        super().__init__(params)

        hardware_config = hardware_config or {}
        drive_config = hardware_config.get("drive", {})
        readout_config = hardware_config.get("readout", {})

        # WHY: Extract physical module slot numbers and sequencer indices for microwave and readout lines.
        self.drive_mod = drive_config.get("module", 2)
        self.drive_seq = drive_config.get("sequencer", 0)
        self.readout_mod = readout_config.get("module", 4)
        self.readout_seq = readout_config.get("sequencer", 0)

        # 1. FIX: Default to drive and readout modules if none provided
        # WHY: q1simulator requires registered modules (QCM-RF for control, QRM-RF for readout) to generate voltage outputs.
        if modules is None:
            modules = {self.drive_mod: "QCM-RF", self.readout_mod: "QRM-RF"}

        # 2. FIX: Safely replace existing instrument in QCoDeS registry if name collides
        # WHY: QCoDeS raises an exception if an instrument with an identical name is instantiated twice without closing the first.
        try:
            if Instrument.exist(name):
                Instrument.find_instrument(name).close()
        except Exception:  # noqa: BLE001, S110
            pass

        self.cluster: Cluster | None = Cluster(name=name, modules=modules)
        self.t_max = 0.0
        self.t_min = 0.0
        self.t_sample = 0.0
        self.t_list: np.ndarray = np.array([], dtype=float)

    def close(self):
        """Safely closes the underlying QCoDeS Cluster instrument connection."""
        if hasattr(self, "cluster") and self.cluster is not None:
            try:
                self.cluster.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self.cluster = None

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit point ensuring instrument resources are freed."""
        self.close()

    def simulate(
        self, schedule: Schedule | None = None, initial_state: qutip.Qobj | None = None
    ) -> dict:
        """
        Extracts pulse waveforms from q1simulator modules and runs QuTiP physics evolution.

        Args:
            schedule (Schedule | None, optional): Kept for API compatibility with parent simulator class. Defaults to None.
            initial_state (qutip.Qobj | None, optional): Initial quantum state override. Defaults to None.

        Returns:
            dict: Simulation result dictionary containing result container, time grid, and measurements list.
        """
        try:
            drive_pulses, readout_pulses = self.get_pulses()
        except (AttributeError, KeyError, RuntimeError) as e:
            print("Error extracting pulses from Q1Simulator:", e)
            drive_pulses, readout_pulses = {}, {}

        # WHY: If t_max was not extracted from module output objects, fallback to default 500 ns window.
        if self.t_max == 0:
            self.t_max = 500
            self.t_sample = 1

        num_points = round(self.t_max / self.t_sample) if self.t_sample > 0 else 500

        # WHY: Construct time array in SI seconds (converting ns via 1e-9).
        self.t_list = np.linspace(0, self.t_max * 1e-9, num_points)

        if initial_state is None:
            initial_state = self.system.get_default_initial_state()

        def safe_data(pulse_dict, key):
            # WHY: Hardware traces extracted from q1simulator might differ in length from self.t_list.
            # Interpolating or zero-padding ensures drive arrays match the solver time grid dimension precisely.
            arr = np.array(pulse_dict.get(key, {}).get("data", []))
            if len(arr) != len(self.t_list):
                if len(arr) > 0:
                    old_t = np.linspace(0, self.t_max * 1e-9, len(arr))
                    arr = np.interp(self.t_list, old_t, arr)
                else:
                    arr = np.zeros_like(self.t_list)
            return arr

        # WHY: Map hardware I/Q traces extracted from QCM/QRM sequencers into complex drive envelopes (I + 1j*Q).
        drives = {
            "q0:mw": safe_data(drive_pulses, "I") + 1j * safe_data(drive_pulses, "Q"),
            "q0:res": safe_data(readout_pulses, "I")
            + 1j * safe_data(readout_pulses, "Q"),
        }

        # WHY: Configure stiff solver options (Backward Differentiation Formula - BDF method) for fast-oscillating hardware traces.
        options = {
            "nsteps": 100000,
            "max_step": 1e-9,
            "rtol": 1e-6,
            "atol": 1e-8,
            "method": "bdf",
        }

        result = None
        try:
            result = self.engine.run(
                system=self.system,
                drives=drives,
                t_list=self.t_list,
                initial_state=initial_state,
                options=options,
            )
        except (RuntimeError, ValueError) as e:
            print(f"✗ Solver failed: {e}")

        measurements = self.get_measurements(result)
        # WHY: Feed computed IQ measurement results back into mock hardware sequencers so hardware driver interfaces can read realistic mock data.
        self.set_acquisition_mock_data(measurements)

        result_container = SimulationResult(
            states=result.states if result is not None else [],
            t_list=self.t_list,
            measurements=measurements,
            system=self.system,
        )
        return {
            "result": result_container,
            "t_list": self.t_list,
            "measurements": measurements,
        }

    def get_pulses(self):
        """
        Extracts output I/Q voltage array traces from connected QCM and QRM hardware mock sequencers.

        Returns:
            tuple[dict, dict]: Pair containing (drive_pulses dictionary, readout_pulses dictionary).
        """
        drive_pulses, readout_pulses = {}, {}
        # 3. FIX: Pylance guard
        if self.cluster is None:
            return drive_pulses, readout_pulses

        connected = self.cluster.get_connected_modules()
        if self.drive_mod not in connected or self.readout_mod not in connected:
            return drive_pulses, readout_pulses

        qcm_outputs = connected[self.drive_mod].get_output()
        qrm_outputs = connected[self.readout_mod].get_output()

        # WHY: Iterate over module output paths to extract waveform data and sampling parameters.
        for path in qcm_outputs:
            if not self.t_max:
                self.t_max = qcm_outputs[path].t_max
                self.t_min = qcm_outputs[path].t_min
                self.t_sample = qcm_outputs[path].sample_rate

            if path.startswith(f"sequencer{self.drive_seq}"):
                drive_pulses[path[-1]] = {"data": qcm_outputs[path].data}
        for path in qrm_outputs:
            if path.startswith(f"sequencer{self.readout_seq}"):
                readout_pulses[path[-1]] = {"data": qrm_outputs[path].data}

        return drive_pulses, readout_pulses

    def get_measurements(self, result: typing.Any) -> list:
        """
        Extracts hardware acquisition windows from QRM modules and processes Time-of-Flight state expectations.

        Args:
            result (typing.Any): Solver result returned by QuTiPEngine.

        Returns:
            list: List of processed acquisition dictionaries containing demodulated voltage records.
        """
        if (
            result is None
            or not hasattr(result, "states")
            or len(getattr(result, "states", [])) == 0
        ):
            return []
        if self.cluster is None:
            return []

        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod not in connected:
                return []
            qrm = connected[self.readout_mod]
            acq_windows = qrm.get_acquisition_windows()

            if (not acq_windows or not any(acq_windows.values())) and (
                hasattr(qrm, "sequencers") and len(qrm.sequencers) > self.readout_seq
            ):
                seq_obj = qrm.sequencers[self.readout_seq]
                if hasattr(seq_obj, "acq_windows"):
                    acq_windows = {self.readout_seq: seq_obj.acq_windows}
        except (AttributeError, KeyError, IndexError, TypeError):
            return []

        windows = (
            acq_windows.get(self.readout_seq)
            or acq_windows.get(str(self.readout_seq))
            or acq_windows.get(f"sequencer{self.readout_seq}")
            or acq_windows.get(f"seq{self.readout_seq}")
        )
        if not windows:
            return []

        measurements = []
        handler = AcquisitionRegistry.get_handler("Trace")

        # PYLANCE TYPE FIX
        a_op = self.system.a.get("q0")
        ad_op = self.system.ad.get("q0")
        if a_op is None or ad_op is None:
            return measurements

        for acq in windows:
            try:
                # WHY: Support multiple tuple structures returned by q1simulator acquisition window configurations.
                if (
                    isinstance(acq, (list, tuple))
                    and len(acq) > 0
                    and isinstance(acq[0], (list, tuple))
                ):
                    acq_start_ns, duration_ns = acq[0][0], acq[0][1]
                elif isinstance(acq, (list, tuple)) and len(acq) >= 2:
                    acq_start_ns, duration_ns = acq[0], acq[1]
                else:
                    continue

                # WHY: Convert start and duration from nanoseconds to seconds.
                acq_start_sec, acq_duration_sec = (
                    float(acq_start_ns) * 1e-9,
                    float(duration_ns) * 1e-9,
                )
                idx = np.argmin(np.abs(self.t_list - acq_start_sec))

                res_dict = handler.process(
                    state=result.states[idx],
                    t_list=self.t_list,
                    states=result.states,
                    acq_time=acq_start_sec,
                    acq_duration=acq_duration_sec,
                    acq_info={"acq_delay": self.cfg.acquisition.cable_delay},
                    cfg=self.cfg,
                    a_op=a_op,
                    ad_op=ad_op,
                )

                measurements.append(
                    {
                        "time": acq_start_sec,
                        "duration": acq_duration_sec,
                        "prob_0": res_dict["prob_0"],
                        "prob_1": res_dict["prob_1"],
                        "leakage_prob_2": res_dict["leakage_prob_2"],
                        "outcome": res_dict["outcome"],
                        "I": res_dict["I"],
                        "Q": res_dict["Q"],
                        "trace": res_dict["value"],
                        "trace_I": np.real(res_dict["value"]),
                        "trace_Q": np.imag(res_dict["value"]),
                    }
                )
            except (AttributeError, KeyError, ValueError):
                continue

        return measurements

    def set_acquisition_mock_data(self, measurements: list):
        """
        Injects computed complex measurement values into mock hardware sequencers.

        Args:
            measurements (list): Processed measurement dictionaries containing "I" and "Q" keys.
        """
        if self.cluster is None or not measurements:
            return
        try:
            connected = self.cluster.get_connected_modules()
            if self.readout_mod in connected:
                qrm = connected[self.readout_mod]
                if "I" in measurements[0] and "Q" in measurements[0]:
                    I_arr, Q_arr = (
                        np.array([m["I"] for m in measurements]),
                        np.array([m["Q"] for m in measurements]),
                    )
                    # WHY: Write I + 1j*Q complex values into mock QRM registers so QCoDeS drivers read expected results.
                    qrm.sequencers[self.readout_seq].set_acquisition_mock_data(
                        I_arr + 1j * Q_arr
                    )
        except (AttributeError, KeyError, ValueError):
            pass
