from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dependencies.randomized_benchmarking.utils import RBAnalysis as BaseRBAnalysis
from scipy.signal import fftconvolve, welch

from qblox_scheduler.analysis import Basic2DAnalysis, acq_coords_to_dims
from qblox_scheduler.analysis import RabiAnalysis as BaseRabiAnalysis
from qblox_scheduler.analysis import (
    ResonatorSpectroscopyAnalysis as BaseResonatorSpectroscopyAnalysis,
)
from qblox_scheduler.analysis import T1Analysis as BaseT1Analysis
from qblox_scheduler.analysis.readout_calibration_analysis import (
    ReadoutCalibrationAnalysis as BaseReadoutCalibrationAnalysis,
)
from qblox_scheduler.analysis.single_qubit_timedomain import (
    EchoAnalysis as BaseEchoAnalysis,
)
from qblox_scheduler.analysis.single_qubit_timedomain import (
    RamseyAnalysis as BaseRamseyAnalysis,
)
from qblox_scheduler.analysis.single_qubit_timedomain import SingleQubitTimedomainAnalysis
from qblox_scheduler.analysis.spectroscopy_analysis import (
    QubitSpectroscopyAnalysis as BaseQubitSpectroscopyAnalysis,
)
from qblox_scheduler.analysis.spectroscopy_analysis import (
    ResonatorFluxSpectroscopyAnalysis as BaseResonatorFluxSpectroscopyAnalysis,
)
from qblox_scheduler.analysis.time_of_flight_analysis import (
    TimeOfFlightAnalysis as BaseTimeOfFlightAnalysis,
)


def _create_analysis_dataset(
    dataset: xr.Dataset,
    coords: dict,
    data_type: str = "complex",  # 'complex', 'iq', 'magnitude_only'
    attrs: dict | None = None,
    adjust: bool = False,
) -> xr.Dataset:
    """Standardize dataset creation for analysis classes."""
    flat_data = dataset["S_21"].to_numpy().flatten()

    if data_type == "complex":
        data_vars = {
            "y0": (
                ("dim_0",),
                np.abs(flat_data),
                {"units": "V", "long_name": "Amplitude"},
            ),
            "y1": (
                ("dim_0",),
                np.angle(flat_data, deg=True),
                {"units": "deg", "long_name": "Phase"},
            ),
        }
    elif data_type == "iq":
        data_vars = {
            "y0": (
                ("dim_0",),
                np.real(flat_data),
                {"units": "V", "long_name": "I"},
            ),
            "y1": (
                ("dim_0",),
                np.imag(flat_data),
                {"units": "V", "long_name": "Q"},
            ),
        }
    elif data_type == "magnitude_only":
        avg = np.average(flat_data) if adjust else 0
        data_vars = {
            "y0": (
                ("dim_0",),
                np.abs(flat_data - avg),
                {"units": "V", "long_name": "Magnitude"},
            )
        }
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    new_dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            f"x{i}": (
                ("dim_0",),
                coord.values,
                {
                    "long_name": coord.attrs.get("long_name", ""),
                    "units": coord.attrs.get("units", ""),
                },
            )
            for i, coord in enumerate(coords.values())
        },
        attrs=attrs or {},
    )
    return new_dataset


class ResonatorSpectroscopyAnalysis(BaseResonatorSpectroscopyAnalysis):
    """Analysis for resonator spectroscopy data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the ResonatorSpectroscopyAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["frequency"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"]},
            attrs={"tuid": dataset.tuid, "name": "ResonatorSpectroscopy"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class ResonatorFluxSpectroscopyAnalysis(BaseResonatorFluxSpectroscopyAnalysis):
    """Analysis for resonator flux spectroscopy data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the ResonatorFluxSpectroscopyAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"], "amplitude": dataset["amplitude"]},
            attrs={"tuid": dataset.tuid, "name": "ResonatorFluxSpectroscopy"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class PunchoutAnalysis(Basic2DAnalysis):
    """Analysis for resonator punchout data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the PunchoutAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"], "amp": dataset["amp"]},
            attrs={"tuid": dataset.tuid, "name": "Punchout"},
        )

        def _normalize_data(ds_raw: xr.Dataset) -> xr.Dataset:
            ds_raw_copy = deepcopy(ds_raw)
            ds_raw_copy["y0"].values = (
                ds_raw["y0"].values.reshape(len(ds_raw["x1"]), -1)
                * 10 ** (ds_raw["x1"].values / 20).reshape(len(ds_raw["x1"]), 1)
            ).flatten()
            ds_raw_copy["y0"].attrs["long_name"] = "|S21|"
            return ds_raw_copy

        dataset = _normalize_data(dataset)
        super().__init__(dataset)


class RabiAnalysis(BaseRabiAnalysis):
    """Analysis for Rabi data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the RabiAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["amplitude"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"amplitude": dataset["amplitude"]},
            attrs={"tuid": dataset.tuid, "name": "Rabi"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class T1Analysis(BaseT1Analysis):
    """Analysis for T1 data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the T1Analysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["tau"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"tau": dataset["tau"]},
            attrs={"tuid": dataset.tuid, "name": "T1"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class QubitSpectroscopyAnalysis(BaseQubitSpectroscopyAnalysis):
    """Analysis for qubit spectroscopy data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the QubitSpectroscopyAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["frequency"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"]},
            attrs={"tuid": dataset.tuid, "name": "QubitSpectroscopy"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class RamseyAnalysis(BaseRamseyAnalysis):
    """Analysis for Ramsey data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the RamseyAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"tau": dataset["tau"]},
            attrs={"tuid": dataset.tuid, "name": "Ramsey"},
        )
        super().__init__(dataset)


class SSROAnalysis(BaseReadoutCalibrationAnalysis):
    """Analysis for single shot readout data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the SSROAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"state": dataset["state"]},
            data_type="iq",
            attrs={"tuid": dataset.tuid, "name": "SSRO"},
        )
        super().__init__(dataset)


class TimeOfFlightAnalysis(BaseTimeOfFlightAnalysis):
    """Analysis for time of flight data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the TimeOfFlightAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={},
            data_type="magnitude_only",
            adjust=True,
            attrs={"tuid": dataset.tuid, "name": "Time of Flight"},
        )
        super().__init__(dataset)


class EchoAnalysis(BaseEchoAnalysis):
    """Analysis for T2 echo data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the EchoAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["tau"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"tau": dataset["tau"]},
            attrs={"tuid": dataset.tuid, "name": "T2echo"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class RBAnalysis(BaseRBAnalysis):
    """Analysis for randomized benchmarking data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the EchoAnalysis with a dataset."""
        # Rotate the data based on the calibration points.
        calibrated_points = np.real(
            rotate_to_calibrated_axis(
                data=dataset.S_21,
                ref_val_0=dataset.calibration.values[0],
                ref_val_1=dataset.calibration.values[1],
            )
        )
        dataset.update({"S_21": calibrated_points})

        dataset = _create_analysis_dataset(
            dataset,
            coords={"length": dataset["length"], "seed": dataset["seed"]},
            attrs={"tuid": dataset.tuid, "name": "Randomized Benchmarking"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class BaseProbeT1Analysis(SingleQubitTimedomainAnalysis):
    """
    Fast T1 analysis based on thresholded single-shot bits.

    Time trace
    ----------
    - Uses a single moving-average window `trace_window` on the raw bits
      to produce a T1(t) trace with uncertainty.

    PSD
    ---
    - High-frequency PSD:
        * Computed from the time trace at full sampling rate `psd_fs`.
    - Low-frequency PSD:
        * The bitstring is split into chunks of `segment_length` bits.
        * Each chunk is treated as one T1 measurement:
              p_k  = mean(bits in chunk k)
              T1_k = -tau / ln(p_k)
        * The resulting coarse series T1_low[k] has sampling rate
              fs_low = psd_fs / segment_length.
        * Welch PSD is computed on T1_low at fs_low.
    - The last low-frequency PSD point and the first high-frequency PSD
      point are connected by a smooth, bent bridge in log-log space.

    Parameters
    ----------
    dataset:
        Dataset containing variable "bits_t1" with thresholded results (0/1).
    tau:
        Fixed wait time used in the T1 schedule (seconds).
    psd_fs:
        Sampling frequency of the T1 bits in Hz (≈ 1 / (reset + tau)).
    trace_window:
        Moving-average window length (samples) for the T1 time trace and
        high-frequency PSD. If None, a default is computed from psd_fs
        targeting a cutoff ~ 1 kHz.
    segment_length:
        Number of bits per chunk when forming T1_low. Each chunk yields
        one T1 measurement; fs_low = psd_fs / segment_length.
    nperseg:
        Welch PSD segment length.
    p01, p10:
        Readout error rates used to correct the T1 time trace. If both are zero,
        no correction is applied.
    bridge_bend:
        Multiplicative bend factor for the PSD bridge in log-log space.
        >1 pushes bridge above the straight line, <1 below.
    time_rb:
        Duration of interleaved randomized benchmarking sections in seconds.
        Used to compute a simple variance estimate from the time trace for comparison
        with the PSD variance.

    """

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        tuid: str | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
        tau: float | None = None,
        psd_fs: float = 1.0,
        trace_window: int | None = None,
        segment_length: int = 2**17,
        nperseg: int = 2**14,
        p01: float = 0.0,
        p10: float = 0.0,
        bridge_bend: float = 2.0,
        time_rb: float = 1e-3,
    ) -> None:
        """Initialize the BaseProbeT1Analysis with a dataset and analysis parameters."""
        super().__init__(dataset, tuid, label, settings_overwrite, plot_figures)

        self.tau = tau
        self.psd_fs = psd_fs
        self.segment_length = segment_length
        self.nperseg = nperseg
        self.p01 = p01
        self.p10 = p10
        self.bridge_bend = bridge_bend
        self.time_rb = time_rb

        # Default trace window from sampling rate if not given (target ~1 kHz cutoff)
        if trace_window is None:
            f_high_target = 1e3  # Hz
            w = max(2, int(psd_fs / f_high_target))
            self.trace_window = w
        else:
            self.trace_window = trace_window

        # Time trace
        self.t1_trace: np.ndarray | None = None
        self.t1_err: np.ndarray | None = None

        # PSD
        self.t1_low: np.ndarray | None = None
        self.f_low: np.ndarray | None = None
        self.p_low: np.ndarray | None = None
        self.f_high: np.ndarray | None = None
        self.p_high: np.ndarray | None = None

    def run(self) -> "ProbeT1Analysis":
        """Run the analysis."""
        # We do not use the parent's process_data() logic; we override it entirely.
        return super().run(calibration_points=False)

    def process_data(self) -> None:
        """Process the dataset to compute T1 time trace and PSDs."""
        if "bits_t1" not in self.dataset:
            raise KeyError("FastT1Analysis expects dataset to contain 'bits_t1' variable.")
        if self.tau is None:
            raise ValueError("tau must be set for FastT1Analysis (T1 wait time in seconds).")

        bits = self.dataset["bits_t1"].values.astype(float).ravel()

        # High-frequency T1 time trace (moving average at native sampling rate)
        self.t1_trace = self._moving_average(bits, self.trace_window)
        self.t1_err = self._error_moving_average(bits, self.trace_window)

        self.quantities_of_interest["T1_mean"] = float(np.nanmean(self.t1_trace))
        self.quantities_of_interest["T1_std"] = float(np.nanstd(self.t1_trace))

        # High-frequency PSD from T1_trace at fs = psd_fs
        t1hf_demeaned = self.t1_trace - np.nanmean(self.t1_trace)

        f_high, p_high = welch(
            t1hf_demeaned,
            fs=self.psd_fs,
            nperseg=self.nperseg,
        )

        self.f_high, self.p_high = f_high, p_high

        # Low-frequency T1 series & PSD (chunk-based, reduced sampling rate)
        segments = self._split_segments(bits, self.segment_length)

        t1_low = []
        for seg in segments:
            p_seg = np.clip(np.mean(seg), 1e-10, 1.0 - 1e-10)
            p_seg = (p_seg - self.p01) / (1 - self.p01 - self.p10)
            p_seg_clipped = np.clip(p_seg, 1e-10, 1.0 - 1e-10)
            t1_low.append(-self.tau / np.log(p_seg_clipped))
        self.t1_low = np.array(t1_low)

        fs_low = self.psd_fs / self.segment_length  # one T1 per segment
        t1lf_demeaned = self.t1_low - np.nanmean(self.t1_low)

        f_low, p_low = welch(
            t1lf_demeaned,
            fs=fs_low,
            nperseg=self.nperseg,
        )

        self.f_low, self.p_low = f_low, p_low

        # Calculate a simple variance estimate from the raw bits for comparison.
        interleaved_rb_duration = self.time_rb
        bits_per_section = int(interleaved_rb_duration * self.psd_fs)
        moving_variances = [
            np.var(self.t1_trace[i : i + bits_per_section])
            for i in range(len(self.t1_trace) - bits_per_section + 1)
        ]
        avg_variance = np.mean(moving_variances)
        self.quantities_of_interest["T1_avg_variance"] = avg_variance

    @staticmethod
    def _split_segments(bits: np.ndarray, segment_length: int) -> list[np.ndarray]:
        n_segments = len(bits) // segment_length
        if n_segments == 0:
            return [bits]
        return np.array_split(bits[: n_segments * segment_length], n_segments)

    def _moving_average(self, bits: np.ndarray, window: int) -> np.ndarray:
        kernel = np.ones(window) / window
        averages = fftconvolve(bits, kernel, mode="valid")
        averages_clipped = np.clip(averages, 1e-10, 1.0 - 1e-10)
        # Correct for SPAM errors using the known error rates p01 and p10
        p = (averages_clipped - self.p01) / (1 - self.p01 - self.p10)
        p_clipped = np.clip(p, 1e-10, 1.0 - 1e-10)
        return -self.tau / np.log(p_clipped)

    def _error_moving_average(
        self,
        bits: np.ndarray,
        window: int,
    ) -> np.ndarray:
        kernel = np.ones(window) / window
        b_meas = fftconvolve(bits, kernel, mode="valid")
        b_meas = np.clip(b_meas, 1e-10, 1.0 - 1e-10)

        N = window
        alpha = 1 - self.p01 - self.p10  # denominator of the linear inversion

        # Corrected population
        p = (b_meas - self.p01) / alpha
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        # Variance of the measured average (Bernoulli)
        var_b_meas = b_meas * (1 - b_meas) / N
        var_p = var_b_meas / alpha**2

        # Propagate through T1 = -tau / ln(p)
        var_t1 = self.tau * np.sqrt(var_p) / (p * np.abs(np.log(p)) ** 2)
        return var_t1

    def create_figures(self) -> None:
        """Create figures for the T1 time trace and PSDs."""
        self._plot_t1_trace()
        self._plot_psd()

    def _plot_t1_trace(self) -> None:
        if self.t1_trace is None or self.t1_err is None:
            return

        fig, ax = plt.subplots()

        t = np.arange(len(self.t1_trace)) / self.psd_fs  # seconds
        ax.plot(t, self.t1_trace * 1e6, color="tab:blue", label=f"T1 (w={self.trace_window})")
        ax.fill_between(
            t,
            (self.t1_trace - self.t1_err) * 1e6,
            (self.t1_trace + self.t1_err) * 1e6,
            alpha=0.3,
            color="tab:blue",
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("T1 (µs)")
        ax.legend(loc="best")
        ax.grid(True, ls=":")
        fig.tight_layout()

        self.figs_mpl["T1_trace"] = fig
        self.axs_mpl["T1_trace"] = ax

    def _plot_psd(self) -> None:
        if self.f_low is None or self.f_high is None:
            return

        # Drop DC bins
        f_low = self.f_low[1:]
        p_low = self.p_low[1:]
        f_high = self.f_high[1:]
        p_high = self.p_high[1:]

        # Endpoints from low and high PSDs
        f11, p11 = float(f_low[-1]), float(p_low[-1])
        f22, p22 = float(f_high[0]), float(p_high[0])

        # Control points for bridge in log-log
        f_mid = np.sqrt(f11 * f22)
        logf1, logf2 = np.log10(f11), np.log10(f22)
        logp1, logp2 = np.log10(p11), np.log10(p22)
        logf_mid = np.log10(f_mid)
        logp_mid_line = logp1 + (logp2 - logp1) * (logf_mid - logf1) / (logf2 - logf1)

        logp_mid = logp_mid_line + np.log10(self.bridge_bend)
        f_ctrl = np.array([f11, f_mid, f22])
        p_ctrl = np.array([p11, 10**logp_mid, p22])

        # Interpolate bridge in log-log
        n_gap = 20
        f_gap = np.logspace(np.log10(f11), np.log10(f22), n_gap)
        logf_ctrl = np.log10(f_ctrl)
        logp_ctrl = np.log10(p_ctrl)
        logp_gap = np.interp(np.log10(f_gap), logf_ctrl, logp_ctrl)
        p_gap = 10**logp_gap

        fig, ax = plt.subplots()
        ax.loglog(f_low, p_low, "o", label="low‑freq PSD")
        ax.loglog(f_high, p_high, "o", label="high‑freq PSD")
        ax.loglog(f_gap, p_gap, "-", label="bridge")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"T1 PSD (s$^2$/Hz)")
        ax.grid(True, which="both", ls=":")
        ax.legend(loc="best")
        fig.tight_layout()

        self.figs_mpl["T1_psd"] = fig
        self.axs_mpl["T1_psd"] = ax


class ProbeT1Analysis(BaseProbeT1Analysis):
    """Public wrapper for ProbeT1 data, mirroring T1Analysis(BaseT1Analysis)."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
        tau: float | None = None,
        psd_fs: float = 1.0,
        trace_window: int | None = None,
        segment_length: int = 2**17,
        nperseg: int = 2**14,
        p01: float = 0.0,
        p10: float = 0.0,
        bridge_bend: float = 2.0,
        time_rb: float = 1e-3,
    ) -> None:
        """Initialize the ProbeT1Analysis with a dataset."""
        super().__init__(
            dataset=dataset,
            tuid=getattr(dataset, "tuid", None),
            label=label,
            settings_overwrite=settings_overwrite,
            plot_figures=plot_figures,
            tau=tau,
            psd_fs=psd_fs,
            trace_window=trace_window,
            segment_length=segment_length,
            nperseg=nperseg,
            p01=p01,
            p10=p10,
            bridge_bend=bridge_bend,
            time_rb=time_rb,
        )


def rotate_to_calibrated_axis(
    data: np.ndarray, ref_val_0: complex, ref_val_1: complex
) -> np.ndarray:
    """
    Rotates, normalizes and offsets complex valued data based on calibration points.

    Parameters
    ----------
    data
        An array of complex valued data points.
    ref_val_0
        The reference value corresponding to the 0 state.
    ref_val_1
        The reference value corresponding to the 1 state.

    Returns
    -------
    :
        Calibrated array of complex data points.

    """
    rotation_anle = np.angle(ref_val_1 - ref_val_0)
    norm = np.abs(ref_val_1 - ref_val_0)
    offset = ref_val_0 * np.exp(-1j * rotation_anle) / norm

    corrected_data = data * np.exp(-1j * rotation_anle) / norm - offset

    return corrected_data
