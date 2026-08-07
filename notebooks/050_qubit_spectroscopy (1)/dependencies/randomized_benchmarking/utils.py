"""Utility functions for executing Schedules on Qblox hardware."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dependencies.randomized_benchmarking.clifford_group import CZ as CZ_PTM
from dependencies.randomized_benchmarking.clifford_group import ZX_01 as ZX_PTM
from dependencies.randomized_benchmarking.clifford_group import (
    Clifford,
    SingleQubitClifford,
    TwoQubitCliffordCZ,
    TwoQubitCliffordZX,
    common_cliffords,
)
from dependencies.randomized_benchmarking.randomized_benchmarking import (
    randomized_benchmarking_sequence,
)
from quantify_core.visualization.mpl_plotting import (
    set_suptitle_from_dataset,
)
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from qblox_scheduler import Schedule
from qblox_scheduler.analysis.single_qubit_timedomain import SingleQubitTimedomainAnalysis
from qblox_scheduler.backends.qblox.constants import MIN_TIME_BETWEEN_OPERATIONS
from qblox_scheduler.operations import CZ, X90, Y90, IdlePulse, Measure, Reset, Rxy, X, Y
from qblox_scheduler.operations.acquisition_library import BinMode
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

if TYPE_CHECKING:
    from collections.abc import Iterable

    from xarray import Dataset


def randomized_benchmarking_schedule(
    qubit_specifier: str | Iterable[str],
    lengths: Iterable[int],
    seeds: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    generator: type[Clifford] = SingleQubitClifford,
) -> Schedule:
    """
    Generate a randomized benchmarking schedule.

    All Clifford gates in the schedule are decomposed into products
    of the following unitary operations:

        {'CZ', 'I', 'Rx(pi)', 'Rx(pi/2)', 'Ry(pi)', 'Ry(pi/2)', 'Rx(-pi/2)', 'Ry(-pi/2)'}

    Parameters
    ----------
    qubit_specifier
        String or iterable of strings specifying which qubits to conduct the
        experiment on. If one name is specified, then single qubit randomized
        benchmarking is performed. If two names are specified, then two-qubit
        randomized benchmarking is performed.
    lengths
        Array of non-negative integers specifying how many Cliffords
        to apply before each recovery and measurement. If lengths is of size M
        then there will be M recoveries and M measurements in the schedule.
    desired_net_clifford_index
        Optional index specifying what the net Clifford gate should be. If None
        is specified, then no recovery Clifford is calculated. The default index
        is 0, which corresponds to the identity gate. For a map of common Clifford
        gates to Clifford indices, please see: two_qubit_clifford_group.common_cliffords
    seeds
        Optional random seeds to use for all lengths m. If a seed is None,
        then a new seed will be used for each length m. Values can be any integer
        between 0 and 2**32 - 1 inclusive.
    repetitions
        Optional positive integer specifying the amount of times the
        Schedule will be repeated. This corresponds to the number of averages
        for each measurement.
    generator
        Clifford decomposition.

    """
    # ---- Error handling and argument parsing ----#
    lengths = np.asarray(lengths, dtype=int)

    if isinstance(qubit_specifier, str):
        qubit_names = [qubit_specifier]
    else:
        qubit_names = [q for q in qubit_specifier]

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit randomized benchmarking supported.")

    # ---- Build RB schedule ----#
    sched = Schedule("Randomized benchmarking on " + " and ".join(qubit_names))

    # two-qubit RB needs buffer time for phase corrections on drive lines
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    # seeds and lengths both have length len(seed_setpoints)*len(length_setpoints)
    # or max_batch_size, whichever is smaller. If seed_setpoints is [1,2,3] and
    # length_setpoints is [4,5], then seeds will be [1,2,3,1,2,3] and lengths will
    # be [4,5,4,5,4,5]. This is why we iterate up to [:-2] for both. # FIXME: this seems fishy
    with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
        for seed in seeds:
            for m in lengths:
                sched.add(Reset(*qubit_names))

                # m-sized random sample of the single/two qubit Clifford group
                rb_sequence_m = randomized_benchmarking_sequence(
                    m,
                    number_of_qubits=n,
                    seed=seed,
                    desired_net_cl=desired_net_clifford_index,
                    generator=generator,
                )

                for clifford_gate_idx in rb_sequence_m:
                    gate_sched = index_to_operation(
                        qubit_names, operation_buffer_time, clifford_gate_idx
                    )
                    if gate_sched is not None:
                        sched.add(gate_sched)

                sched.add(
                    Measure(qubit_names[-1], coords={"seed": seed, "length": m}, acq_channel="S_21")
                )

            # Calibration points measured by preparing ground and excited states.
            sched.add(Reset(qubit_names[-1]))
            sched.add(Measure(qubit_names[-1], acq_channel="calibration"))
            reset_cal_1 = sched.add(Reset(qubit_names[-1]))
            for qubit_name in qubit_names:
                sched.add(X(qubit_name), ref_op=reset_cal_1, rel_time=0)
            sched.add(Measure(qubit_names[-1], acq_channel="calibration"))

    return sched


def interleaved_randomized_benchmarking_and_t1(
    qubit_name: str,
    length: int,
    seeds: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    generator: type[Clifford] = SingleQubitClifford,
    n_t1_points: int = 10,
    t1: float = None,
) -> Schedule:
    """
    Generate a randomized benchmarking schedule.

    All Clifford gates in the schedule are decomposed into products
    of the following unitary operations:

        {'CZ', 'I', 'Rx(pi)', 'Rx(pi/2)', 'Ry(pi)', 'Ry(pi/2)', 'Rx(-pi/2)', 'Ry(-pi/2)'}

    Parameters
    ----------
    qubit_name
        String specifying which qubit to conduct the experiment on.
    length
        Non-negative integer specifying how many Cliffords to apply.
    desired_net_clifford_index
        Optional index specifying what the net Clifford gate should be. If None
        is specified, then no recovery Clifford is calculated. The default index
        is 0, which corresponds to the identity gate. For a map of common Clifford
        gates to Clifford indices, please see: two_qubit_clifford_group.common_cliffords
    seeds
        Optional random seeds to use for all lengths m. If a seed is None,
        then a new seed will be used for each length m. Values can be any integer
        between 0 and 2**32 - 1 inclusive.
    repetitions
        Optional positive integer specifying the amount of times the
        Schedule will be repeated. This corresponds to the number of averages
        for each measurement.
    generator
        Clifford decomposition.
    n_t1_points
        Specifies how many T1 points to acquire per (seed, m) pair.
    t1
        Specifies the delay time for the T1 points.

    """
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][0]

    interleaved_rb_t1 = Schedule(name="interleaved_rb_t1_experiment")

    with interleaved_rb_t1.loop(arange(0, repetitions, 1, DType.NUMBER)):
        for seed in seeds:
            interleaved_rb_t1.add(Reset(qubit_name))

            # m-sized random sample of the single/two qubit Clifford group
            rb_sequence_m = randomized_benchmarking_sequence(
                length,
                number_of_qubits=1,
                seed=seed,
                desired_net_cl=desired_net_clifford_index,
                generator=generator,
            )

            for clifford_gate_idx in rb_sequence_m:
                gate_sched = index_to_operation(
                    [qubit_name], operation_buffer_time, clifford_gate_idx
                )
                if gate_sched is not None:
                    interleaved_rb_t1.add(gate_sched)

            interleaved_rb_t1.add(
                Measure(
                    qubit_name,
                    acq_protocol="ThresholdedAcquisition",
                    bin_mode=BinMode.APPEND,
                )
            )

            # T1 interleaved part
            with interleaved_rb_t1.loop(arange(0, n_t1_points, 1, DType.NUMBER)):
                interleaved_rb_t1.add(Reset(qubit_name))
                # Prepare |1>
                interleaved_rb_t1.add(X(qubit_name))
                # Measure after time T1
                interleaved_rb_t1.add(
                    Measure(
                        qubit_name,
                        acq_protocol="ThresholdedAcquisition",
                        bin_mode=BinMode.APPEND,
                    ),
                    ref_pt="start",
                    rel_time=round(t1, 9),
                )

    return interleaved_rb_t1


def index_to_operation(
    qubit_names: list[str],
    operation_buffer_time: float,
    clifford_gate_idx: int,
) -> Schedule | None:
    """
    Convert a Clifford gate index to a Quantify Schedule of physical operations.

    This function takes a list of qubit names, a buffer time between operations, and a Clifford gate index.
    It determines the appropriate Clifford class (single or two-qubit), obtains the gate decomposition for the
    specified Clifford index, and maps each gate in the decomposition to a Quantify operation using a predefined
    mapping. The resulting operations are assembled into a Quantify Schedule, with appropriate timing and referencing
    for single- and two-qubit gates. If the decomposition results in no physical operations, None is returned.

    Parameters
    ----------
    qubit_names : list[str]
        List of qubit names. Length 1 for single-qubit, 2 for two-qubit Clifford gates.
    operation_buffer_time : float
        Buffer time (in seconds) to insert between operations in the schedule.
    clifford_gate_idx : int
        Index of the Clifford gate to decompose and schedule.

    Returns
    -------
    Schedule | None
        A Quantify Schedule object containing the physical operations for the Clifford gate,
        or None if the decomposition results in no operations.

    Raises
    ------
    NotImplementedError
        If the number of qubits is not 1 or 2.

    """
    if len(qubit_names) == 1:
        clifford_class = SingleQubitClifford
    elif len(qubit_names) == 2:  # noqa: PLR2004
        clifford_class = TwoQubitCliffordCZ  # TwoQubitCliffordZX#
    else:
        raise NotImplementedError
    # ---- PycQED mappings ----#
    # map the pycqed qubit names to the ones used in quantify
    pycqed_qubit_map = {f"q{idx}": name for idx, name in enumerate(qubit_names)}
    # pycqed returns RB sequences as a list of strings. Map those to quantify operations
    pycqed_operation_map = {
        "I": lambda q: None,  # noqa: ARG005
        "X180": lambda q: X(pycqed_qubit_map[q[0]]),
        "X90": lambda q: X90(pycqed_qubit_map[q[0]]),
        "Y180": lambda q: Y(pycqed_qubit_map[q[0]]),
        "Y90": lambda q: Y90(pycqed_qubit_map[q[0]]),
        "mX90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=0.0, theta=-90.0),
        "mY90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=90.0, theta=-90.0),
        "CZ": lambda q: CZ(qC=pycqed_qubit_map[q[0]], qT=pycqed_qubit_map[q[1]]),
    }
    cl_decomp = clifford_class(clifford_gate_idx).gate_decomposition()
    gate_sched = Schedule("gate_sched")
    ref_op = gate_sched.add(IdlePulse(0.0))
    ref_ops = [ref_op, ref_op]

    for qubits, gates in cl_decomp:
        subsched = Schedule("subsched")
        subsched.add(IdlePulse(0.0))
        for gate in gates:
            op = pycqed_operation_map[gate](qubits)
            if op is not None:
                subsched.add(op, rel_time=operation_buffer_time)
        if len(subsched.operations) == 1:
            # no gates added, only the initial IdlePulse
            continue
        if qubits == ("q0",):
            schedulable = gate_sched.add(subsched)
            ref_ops[0] = schedulable
        elif qubits == ("q1",):
            # FIXME: this relies on the fact that single qubit Clifford are ALWAYS defined for both, and ALWAYS in the order q0, q1
            schedulable = gate_sched.add(subsched)
            ref_ops[1] = schedulable
        elif qubits in [("q0", "q1"), ("q1", "q0")]:
            schedulable = gate_sched.add(subsched)
            schedulable.add_timing_constraint(operation_buffer_time, ref_ops[1])
            ref_ops = [schedulable, schedulable]
    if not gate_sched.operations:
        return None
    return gate_sched


def test_rb_sequence(n_gates: int, generator: type[Clifford], n_qubits: int) -> None:
    clifford_idx = {
        "I": 0,
        "X90": 16,
        "Y90": 21,
        "mX90": 13,
        "mY90": 15,
        "mZ90": 23,
        "X180": 3,
        "Y180": 6,
        "CZ": TwoQubitCliffordCZ._get_clifford_id(CZ_PTM),
        "ZX": TwoQubitCliffordZX._get_clifford_id(ZX_PTM),
    }
    rb_sequence = randomized_benchmarking_sequence(
        n_gates, number_of_qubits=n_qubits, generator=generator
    )
    net_clifford = generator(0)
    for idx in rb_sequence:
        cl_decomp = generator(idx).gate_decomposition()
        for base_gate in cl_decomp:
            for native_gate in base_gate[1]:
                ci = clifford_idx[native_gate]
                if base_gate[0] == ("q1",):
                    ci *= 24
                net_clifford = generator(ci) * net_clifford
    assert net_clifford.idx == 0


# print("Testing decompositions.")
# test_rb_sequence(10001, TwoQubitCliffordCZ, 2)
# test_rb_sequence(10001, TwoQubitCliffordZX, 2)
# test_rb_sequence(10001, SingleQubitClifford, 1)
# print("Test passed.")


class RBAnalysis(SingleQubitTimedomainAnalysis):
    """
    Analysis class for the randomized benchmarking (RB) experiment.

    This class extends the SingleQubitTimedomainAnalysis class, which in turn extends the
    BaseAnalysis class:
    - BaseAnalysis.run() runs all steps in the AnalysisSteps class:
        1. process_data                  # Empty
        2. run_fitting                   # Empty
        3. analyze_fit_results           # Empty
        4. create_figures                # Empty
        5. adjust_figures                # Defined
        6. save_figures                  # Defined
        7. save_quantities_of_interest   # Defined
        8. save_processed_dataset        # Defined
        9. save_fit_results              # Defined
    - SingleQubitTimedomainAnalysis extends BaseAnalysis:
        - run() defines self.calibration_points
        - process_data() populates dataset_processed.S21 and dataset_processed.pop_exc
    - RBAnalysis extends SingleQubitTimedomainAnalysis:
        - process_data() is extended by calculating:
            - pop_exc
        - create_figures() is defined
    """

    def __init__(  # noqa: D107
        self,
        dataset: Dataset = None,
        tuid: str = None,
        label: str = "",
        settings_overwrite: dict = None,
        plot_figures: bool = True,
        repetitions: int = 1,
        n_qubits: int = 1,
        yscale: str = "lin",
    ) -> None:
        super().__init__(dataset, tuid, label, settings_overwrite, plot_figures)
        self.repetitions = repetitions
        self.n_qubits = n_qubits
        self.asymptote = 1 / (2**n_qubits)
        self.yscale = yscale

    def run(self):  # noqa: F811
        """
        Run the SingleQubitTimedomainAnalysis with calibration_points.

        This removes the calibration points (last two) and converts
        the rest of the IQ values to a population (pop_exc).
        """
        return super().run(calibration_points=False)

    def process_data(self) -> None:  # noqa: D102
        def _error_per_clifford(
            alpha: float,
            n_qubits: int = 1,
        ) -> float:
            """Error per Clifford as defined in eq.(1) of arxiv:1712.06550."""
            return (2**n_qubits - 1) / 2**n_qubits * (1 - alpha)

        def _rb_decay(
            m: int,
            alpha: float,
            prefactor: float,
        ) -> float:
            """Exponential decay consistent with eq.(1) of arxiv:1712.06550."""
            return prefactor * alpha**m + self.asymptote

        # The processed data set gives us the excited state population vs time.
        # From this, we can calculate the error rate and fidelity.
        super().process_data()

        # TODO: If we don't go back to the initial state, then 1-<1|\psi> is
        # no longer the right metric.
        overlap = 1 - self.dataset_processed["S21"].values
        overlap = overlap**self.n_qubits
        # Add the overlap to the dataset that's returned to the user
        self.dataset_processed["overlap"] = (["x0"], overlap)

        # m_values are the setpoints for the number of Cliffords per measurement
        # for this specific batch of measurements
        m_values = self.dataset_processed.x0.values

        # Fit exponential decay
        popt, pcov = curve_fit(
            _rb_decay,
            m_values,
            overlap,
            p0=(0.9, 1),  # Use alpha=0.9 and prefactor=1 as starting point
            bounds=([0, 0], [1, 1]),
        )
        (self.alpha, self.prefactor) = popt
        fit_errors = np.sqrt(np.diag(pcov))

        # Convert alpha to r as defined in eq.(1) of arxiv:1712.06550
        self.r = _error_per_clifford(alpha=self.alpha, n_qubits=self.n_qubits)
        # Store alpha, error per clifford and prefactor inside quantities of interest
        self.quantities_of_interest["alpha"] = self.alpha
        self.quantities_of_interest["error_per_clifford"] = self.r
        self.quantities_of_interest["error_per_clifford_error"] = fit_errors[0]
        self.quantities_of_interest["prefactor"] = self.prefactor
        self.quantities_of_interest["prefactor_error"] = fit_errors[1]

        # Since the measurement is 2D, seed setpoints [A,B] and m setpoints
        # [1,2,3] will give m_values for this batch of [1,2,3,1,2,3], but we
        # only want [1,2,3].
        unique_m = np.unique(m_values)
        # Add the unique m as a coordinate axis to the dataset
        self.dataset_processed = self.dataset_processed.assign_coords(
            unique_m=("unique_m", unique_m)
        )
        # Add the fit to the dataset
        self.dataset_processed["fitted_overlap"] = (["unique_m"], _rb_decay(unique_m, *popt))

    def create_figures(self) -> None:
        """Create simplified figure."""
        fig, ax = plt.subplots()

        ax.scatter(self.dataset_processed.x0, self.dataset_processed.overlap, label="data")
        ax.plot(self.dataset_processed.unique_m, self.dataset_processed.fitted_overlap, label="fit")
        ax.set_xlabel("Sequence length [#]")
        ax.set_ylabel(r"Population of |0$\rangle$")

        set_suptitle_from_dataset(fig, self.dataset)


class InterleavedRBT1Analysis(SingleQubitTimedomainAnalysis):
    """
    Analysis class for the interleaved fast-RB / fast-T1 experiment.

    Processes one or more pre-acquired datasets from an interleaved RB/T1 schedule,
    aggregating T1 and average gate fidelity estimates across iterations to produce
    time-series plots and a T1–fidelity scatter plot with Pearson correlation.

    The schedule acquires:
    - RB data: IQ measurements for each (seed, m) pair, on acq_channel="S_21".
    - T1 data: N_T1_points thresholded bit measurements per seed loop, via
      BinMode.APPEND on acq_channel="bits_T1".

    Quantities of interest (per iteration, stored as lists)
    -------------------------------------------------------
    mean_y_rb, mean_y_rb_error  : mean RB overlap and its std error
    p_rb, p_rb_error            : depolarizing parameter and its propagated error
    error_per_clifford, _error  : error per Clifford gate and its propagated error
    avg_gate_fidelity, _error   : average gate fidelity and its propagated error
    mean_p1_t1                  : mean T1 excited-state population
    t1_estimate, _error         : T1 estimate and its propagated binomial error
    pearson_r, pearson_pval     : Pearson correlation between T1 and fidelity
    """

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        tuid: str | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
        n_qubits: int = 1,
        repetitions: int = 1,
        seeds: Iterable[int] | None = None,
        n_t1_points: int = 10,
        tau: float | None = None,
        a: float = 1.0,
        m_length: int | None = None,
        t1_var: float | None = None,
        p01: float = 0.0,
        p10: float = 0.0,
    ) -> None:
        """
        Initialize the InterleavedRBT1Analysis.

        Parameters
        ----------
        dataset
            xarray Dataset containing the experimental data to be analyzed.
            If None, the analysis will attempt to load data based on `tuid`.
        tuid
            Unique experiment identifier used to retrieve datasets from storage when
            `datasets` is not provided.
        label
            Optional label for this analysis instance, used in figure titles and
            log messages.
        settings_overwrite
            Dictionary of analysis or fit settings that should override the default
            configuration.
        plot_figures
            If True, generate and store plots of the fitted RB and T1 curves as part
            of the analysis workflow.
        n_qubits
            Number of qubits included in the interleaved RB–T1 experiment (used for
            consistency checks and metadata, not for the fit model itself).
        repetitions
            Number of times each RB–T1 configuration (sequence length and T1 point)
            is repeated to improve statistics.
        seeds
            Iterable of random seeds used for the RB sequences
        n_t1_points
            Number of distinct T1 delay points used in the interleaved T1 schedule.
        tau
            Base time spacing of the T1 delays (e.g. in seconds); if None, it is
            inferred from the dataset metadata.
        a
            Amplitude scaling parameter for the RB–T1 decay model, related to SPAM contrast.
        m_length
            Reference RB sequence length used in the interleaved schedule; if None,
            a default or dataset‑dependent value is used.
        t1_var
            Optional parameter characterizing the variance of the T1 measurements.
        p01, p10
            Optional parameters characterizing state preparation and measurement (SPAM)

        """
        super().__init__(None, tuid, label, settings_overwrite, plot_figures)
        self.n_qubits = n_qubits
        self.repetitions = repetitions
        self.asymptote = 1 / (2**n_qubits)
        self.n_t1_points = n_t1_points
        self.tau = tau
        self.a = a
        self.m_length = m_length
        self.t1_var = t1_var
        self.p01 = p01
        self.p10 = p10

        self.dataset: xr.Dataset | None = dataset
        self.seeds = list(seeds)

        # Aggregated results across iterations
        self.t1_values: list[float] = []
        self.t1_errors: list[float] = []
        self.f_avg_values: list[float] = []
        self.f_avg_errors: list[float] = []

        self.t1_total_errors: list[float] = []
        self.f_avg_total_errors: list[float] = []

    def run(self) -> InterleavedRBT1Analysis:
        """Run the interleaved RB/T1 analysis on a single dataset with repetitions."""
        if self.dataset is None:
            raise ValueError("No dataset provided to analysis.")
        self.process_data()
        return self

    def process_data(self) -> None:  # noqa: PLR0915
        """
        Process a single interleaved RB/T1 dataset.

        Per repetition, the acquisition order in y0 is:
            [RB(seed_1), T1(seed_1, 1..n_t1_points),
            RB(seed_2), T1(seed_2, 1..n_t1_points),
            ...
            RB(seed_N), T1(seed_N, 1..n_t1_points),
            cal0, cal1]
        """
        # Step 1: Separate RB and T1 data, and calculate the IQ threshold from the calibration points

        y0 = np.asarray(self.dataset[0].values)

        shots_per_seed = self.n_t1_points + 1  # 1 RB shot + n_t1_points T1 shots per seed
        shots_per_rep = len(self.seeds) * shots_per_seed

        y0_rep = y0.reshape(self.repetitions, shots_per_rep)
        seed_block = y0_rep[:, : len(self.seeds) * shots_per_seed]
        seed_block = seed_block.reshape(self.repetitions, len(self.seeds), shots_per_seed)

        rb_bits = seed_block[:, :, 0]
        t1_bits = seed_block[:, :, 1:]

        # Step 2: Average gate fidelity
        overlap = rb_bits  # 1 if |0>, 0 if |1>

        mean_y_rep = overlap.mean(axis=1)
        sigma_mean_y_rep = np.std(overlap, axis=1) / np.sqrt(len(self.seeds))

        f_rep = np.empty(self.repetitions, dtype=float)
        f_err_rep = np.empty(self.repetitions, dtype=float)

        for i in range(self.repetitions):
            mean_y = mean_y_rep[i]
            sigma_mean_y = float(sigma_mean_y_rep[i])
            # The RB model is: overlap(m) = A * p^m + B, where B = asymptote = 1/2^n_qubits.
            ratio_raw = (mean_y - self.asymptote) / self.a
            if not (0.0 <= ratio_raw <= 1.0):
                ratio_raw = np.nan

            ratio = np.clip(
                ratio_raw, 1e-6, 1.0 - 1e-6
            )  # ratio is clipped to avoid numerical issues with log and power
            p_estimate = float(ratio ** (1.0 / self.m_length))

            # Propagate error from mean_y to p using the derivative dp/dy
            dp_dy = (1.0 / (self.m_length * self.a)) * ratio ** (1.0 / self.m_length - 1)
            sigma_p = float(abs(dp_dy) * sigma_mean_y)

            epc_prefactor = (2**self.n_qubits - 1) / (
                2**self.n_qubits
            )  # error per clifford prefactor (d-1)/d
            r_estimate = float(epc_prefactor * (1 - p_estimate))
            sigma_r = float(epc_prefactor * sigma_p)

            f_rep[i] = 1.0 - r_estimate
            f_err_rep[i] = sigma_r

        # Step 3: T1 estimate
        t1_bits_rep = t1_bits.reshape(self.repetitions, -1)

        t1_rep = np.empty(self.repetitions, dtype=float)
        t1_err_rep = np.empty(self.repetitions, dtype=float)

        for i in range(self.repetitions):
            bits = t1_bits_rep[i]
            mean_bits = float(bits.mean())
            # value is clipped to avoid numerical issues with log and power
            mean_bits_clipped = np.clip(mean_bits, 1e-10, 1.0 - 1e-10)
            # Correct for SPAM errors using the known error rates p01 and p10
            p = (mean_bits_clipped - self.p01) / (1 - self.p01 - self.p10)
            p_clipped = np.clip(p, 1e-10, 1.0 - 1e-10)
            # Estimate T1 from mean_bits using the model P(1|tau) = exp(-tau/T1)
            t1_estimate = float(-self.tau / np.log(p_clipped))

            # Binomial standard error is calculated for the T1 time trace
            sigma_mean_bits = np.sqrt(
                p_clipped * (1 - p_clipped) / (t1_bits.size * self.repetitions)
            )
            dt1_dp1 = self.tau / (p_clipped * np.log(p_clipped) ** 2)
            sigma_t1 = float(abs(dt1_dp1) * sigma_mean_bits)

            t1_rep[i] = t1_estimate
            t1_err_rep[i] = sigma_t1

        self.t1_values.extend(t1_rep.tolist())
        self.t1_errors.extend(t1_err_rep.tolist())
        self.f_avg_values.extend(f_rep.tolist())
        self.f_avg_errors.extend(f_err_rep.tolist())

        # Combine shot error with intrinsic error calculated from ProbeT1
        t1_total_err_rep = np.sqrt(t1_err_rep**2 + self.t1_var)
        f_total_err_rep = np.sqrt(f_err_rep**2)

        self.t1_total_errors.extend(t1_total_err_rep.tolist())
        self.f_avg_total_errors.extend(f_total_err_rep.tolist())

    def create_figures(self) -> None:
        """Create summary figures for interleaved RB/T1 analysis."""
        t1_vals = np.array(self.t1_values)
        t1_errs = np.array(self.t1_total_errors)
        f_vals = np.array(self.f_avg_values)
        f_errs = np.array(self.f_avg_total_errors)
        idx = np.arange(len(t1_vals))

        # Figure 1: Time traces
        fig1, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.errorbar(
            idx,
            t1_vals * 1e6,
            # yerr=t1_errs * 1e6,
            fmt="o-",
            color="tab:blue",
            capsize=3,
            label="T1",
        )
        ax2.errorbar(
            idx,
            f_vals,
            # yerr=f_errs,
            fmt="s-",
            color="tab:orange",
            capsize=3,
            label="Avg. gate fidelity",
        )

        ax1.set_xlabel("Measurement index")
        ax1.set_ylabel("T1 (µs)", color="tab:blue")
        ax2.set_ylabel("Avg. gate fidelity", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fig1.tight_layout()
        self.figs_mpl["T1_fidelity_timeseries"] = fig1
        self.axs_mpl["T1_timeseries"] = ax1
        self.axs_mpl["fidelity_timeseries"] = ax2

        # Figure 2: Scatter T1 vs fidelity with Pearson correlation
        fig2, ax3 = plt.subplots()

        ax3.errorbar(t1_vals * 1e6, f_vals, xerr=t1_errs * 1e6, yerr=f_errs, fmt="o", capsize=3)
        ax3.set_xlabel("T1 (µs)")
        ax3.set_ylabel("Avg. gate fidelity")

        # Calculate Pearson correlation, ignoring NaNs, and starting from the second point to avoid initialization transients
        MIN_POINTS_FOR_PEARSON = 2
        valid = ~(np.isnan(t1_vals) | np.isnan(f_vals))
        if valid.sum() >= MIN_POINTS_FOR_PEARSON:
            r, pval = pearsonr(t1_vals[valid], f_vals[valid])
            self.quantities_of_interest["pearson_r"] = float(r)
            self.quantities_of_interest["pearson_pval"] = float(pval)
            ax3.set_title(f"Pearson r = {r:.3f}")

        fig2.tight_layout()
        self.figs_mpl["T1_fidelity_scatter"] = fig2
        self.axs_mpl["T1_fidelity_scatter"] = ax3
