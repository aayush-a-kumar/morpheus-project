# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Parses scheduling instructions into time-domain waveforms.

This module is responsible for interpreting Qblox Schedule timing tables
and converting parameterized pulse definitions (like Gauss or DRAG) into
continuous, complex-valued IQ envelope arrays that the QuTiP engine can process.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd


def extract_amplitude(source: Mapping[str, Any] | pd.Series, default: Any = 0.0) -> Any:
    """
    Safely extracts the amplitude from a pulse dictionary or pandas Series.

    Args:
        source (Mapping[str, Any] | pd.Series): The data source containing pulse parameters.
        default (Any, optional): Fallback value if no valid amplitude is found. Defaults to 0.0.

    Returns:
        Any: The extracted amplitude, which may be a float or a symbolic Variable object.
    """
    # WHY: During schedule compilation, loop sweep variables might remain as symbolic
    # Variable objects rather than floats. We must preserve them without triggering pandas NaN checks.
    d = source.to_dict() if isinstance(source, pd.Series) else source

    if isinstance(d, dict):
        for key in ("amplitude", "amp"):
            val = d.get(key)
            if val is not None:
                try:
                    if not pd.isna(val):
                        return val
                except (ValueError, TypeError, AttributeError):
                    return val
    return default


class SignalProvider(Protocol):
    """Abstract interface for drive signal providers."""

    def get_drives(self, t_list: np.ndarray) -> dict[str, np.ndarray]:
        """
        Returns a map of drive channels to complex envelope arrays.

        Args:
            t_list (np.ndarray): The simulation time grid.

        Returns:
            dict[str, np.ndarray]: Mapping of port names to time-series drive arrays.
        """
        ...


class ScheduleSignalProvider:
    """Parses Qblox Schedule timing tables into IQ time-series arrays."""

    def __init__(self, pulses_list: Sequence[Mapping[str, Any]]):
        """
        Args:
            pulses_list (Sequence[Mapping[str, Any]]): A list of dictionaries,
                where each dictionary represents a scheduled pulse event.
        """
        self.pulses_list = pulses_list

    def _pulse_envelope_vectorized(
        self, t_rel: np.ndarray, pulse_info: Mapping[str, Any]
    ) -> np.ndarray:
        """
        Generates a continuous time-domain voltage envelope for a given pulse schedule.

        Args:
            t_rel (np.ndarray): Relative time array covering the pulse duration.
            pulse_info (Mapping[str, Any]): Dictionary containing pulse parameters
                (duration, amplitude, phase, wf_func).

        Returns:
            np.ndarray: Complex-valued array representing the IQ waveform envelope.
        """
        duration = pulse_info["duration"]
        amp = extract_amplitude(pulse_info)
        phase_rad = np.deg2rad(pulse_info.get("phase", 0.0))

        wf_raw = pulse_info.get("wf_func")
        wf_func = str(wf_raw).lower() if wf_raw else "square"

        if "gauss" in wf_func:
            sigma = pulse_info.get("sigma", duration / 4)
            sigma = 1e-12 if (sigma is None or sigma == 0) else sigma
            t_mid = duration / 2
            # WHY: Standard Gaussian envelope for smooth, bandwidth-limited transitions.
            envelope = amp * np.exp(-((t_rel - t_mid) ** 2) / (2 * sigma**2))

        elif "drag" in wf_func:
            sigma = pulse_info.get("sigma", duration / 4)
            sigma = 1e-12 if (sigma is None or sigma == 0) else sigma
            beta = pulse_info.get("beta", 0.0)
            t_mid = duration / 2

            # WHY: DRAG (Derivative Removal by Adiabatic Gate) pulses use a Gaussian envelope
            # on the I-channel and its derivative on the Q-channel. This suppresses leakage
            # into the |2> state of the transmon by destructively interfering with the unwanted transition.
            envelope = amp * np.exp(-((t_rel - t_mid) ** 2) / (2 * sigma**2))
            envelope_dot = -(t_rel - t_mid) / (sigma**2) * envelope

            # Return early here to apply the beta derivative to the imaginary axis
            return (envelope + 1j * (-beta * envelope_dot / (2 * np.pi))) * np.exp(
                1j * phase_rad
            )
        else:
            # WHY: Default to a square (constant) pulse if no specific waveform is requested.
            envelope = np.full_like(t_rel, amp, dtype=complex)

        # WHY: Multiply by exp(1j * phase_rad) to set the initial pulse phase in the rotating frame.
        return envelope * np.exp(1j * phase_rad)

    def get_drives(self, t_list: np.ndarray) -> dict[str, np.ndarray]:
        """
        Maps the list of discrete pulse events onto the continuous simulation time grid.

        Args:
            t_list (np.ndarray): The uniform time array used by the QuTiP solver.

        Returns:
            dict[str, np.ndarray]: A dictionary mapping hardware ports to their aggregated
                time-series waveforms.
        """
        drives: dict[str, np.ndarray] = {}

        if len(t_list) <= 1:
            return drives

        dt_actual = t_list[1] - t_list[0]
        t_start_grid = t_list[0]

        for p in self.pulses_list:
            port = p.get("port")
            if not port:
                continue

            t_start = p["abs_time"]
            duration = p["duration"]
            t_end = t_start + duration

            # WHY: Map the continuous schedule time (abs_time) onto the discrete simulator
            # time grid (t_list). This ensures pulses align correctly with integration steps.
            idx_start = max(0, int(np.floor((t_start - t_start_grid) / dt_actual)))
            idx_end = min(
                len(t_list), int(np.ceil((t_end - t_start_grid) / dt_actual)) + 1
            )

            if idx_start >= len(t_list) or idx_end <= 0:
                continue

            # WHY: Create a relative time axis strictly bounded by the pulse duration
            # so the waveform generation functions calculate shapes accurately.
            t_slice = t_list[idx_start:idx_end]
            t_rel = t_slice - t_start
            mask = (t_rel >= 0) & (t_rel <= duration)

            if not np.any(mask):
                continue

            t_rel_valid = t_rel[mask]
            signal = self._pulse_envelope_vectorized(t_rel_valid, p)

            # Dynamically aggregate signals into their specific hardware port
            if port not in drives:
                drives[port] = np.zeros_like(t_list, dtype=complex)

            # WHY: Multiple pulses can occur on the same port simultaneously
            # (e.g., multiplexed readout tones). We dynamically add them together.
            drives[port][idx_start:idx_end][mask] += signal

        return drives
