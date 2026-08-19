# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
"""
Defines the static hardware configuration and simulation parameters.

This module uses dataclasses to represent the physical properties of the
quantum topology (qubits, readout resonators, and their couplings).
These parameters dictate the Hamiltonian construction and measurement noise models.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QubitConfig:
    """
    Physical parameters defining a tunable transmon qubit.
    """

    # WHY: Qubit bare frequency (f_q) and drive frequency (f_d).
    # The simulation operates in the drive frame, so the detuning is (f_q - f_d).
    f_q: float = 5.0e9
    f_d: float = 5.0e9

    # WHY: Transmons are slightly non-linear oscillators. Alpha is the negative anharmonicity
    # (the difference in frequency between the |0>->|1> and |1>->|2> transitions).
    alpha: float = -300.0e6

    # WHY: Number of energy levels to simulate. Set to 3 by default to capture
    # leakage into the non-computational |2> state during fast gates.
    N_q: int = 3

    # WHY: Maps the baseband voltage amplitude from the Qblox schedule to the
    # physical Rabi frequency (drive strength) experienced by the qubit.
    rabi_freq_per_volt: float = 10.0e6
    flux_freq_per_volt: float = 1.0e9

    # WHY: Phenomenological decoherence times. T1 is energy relaxation (amplitude damping),
    # and T2 is total dephasing. np.inf implies an ideal, lossless qubit.
    T1: float = np.inf
    T2: float = np.inf

    # --- New Non-Linear Flux Parameters ---
    # WHY: A tunable transmon uses a SQUID loop. Its frequency sweeps in an arc based on magnetic flux.
    # v_phi0 represents the voltage required to thread one magnetic flux quantum (Φ0) through the SQUID.
    v_phi0: float | None = None
    f_max: float | None = (
        None  # The maximum 'sweet spot' frequency where the qubit is least sensitive to flux noise.
    )


@dataclass(frozen=True)
class ResonatorConfig:
    """
    Physical parameters defining a readout cavity/resonator.
    """

    f_res: float = 6.0e9
    f_d_res: float = 6.0e9

    # WHY: Number of photon levels in the readout cavity.
    # Needs to be high enough to capture the photon population during readout drives.
    N_res: int = 5

    # WHY: Dispersive shift (chi). This is the shift in the resonator's frequency
    # depending on whether the coupled qubit is in |0> or |1>.
    chi: float = 1.0e6

    # WHY: Kappa is the resonator's photon decay rate (coupling to the feedline).
    # Dictates how fast information leaks out to the measurement chain.
    kappa: float = 1.0e6
    rabi_freq_res_per_volt: float = 10.0e6


@dataclass(frozen=True)
class CouplingConfig:
    """
    Defines the static exchange coupling between two qubits.

    WHY: Represents a capacitive coupling between two adjacent transmons,
    allowing them to exchange excitations (vital for two-qubit gates).
    """

    q1: str
    q2: str
    J: float = 0.0  # Coupling strength in Hz.


@dataclass(frozen=True)
class AcquisitionConfig:
    """
    Parameters for the simulated measurement chain and demodulation.
    """

    # WHY: Simulates the total noise temperature (e.g., from the HEMT and quantum limits)
    # added to the IQ voltage traces.
    noise_sigma: float = 0.02

    # WHY: Accounts for the physical time it takes the readout pulse to travel
    # through the cryostat cabling before hitting the digitizer.
    cable_delay: float = 120e-9

    # WHY: These are the calibrated IQ plane centroids for the ground and excited states
    # after signal demodulation and integration.
    v_0: complex = complex(0.05, 0.05)
    v_1: complex = complex(-0.05, -0.05)


@dataclass(frozen=True)
class SimulationConfig:
    """
    The root configuration object grouping all hardware components and solver settings.
    """

    qubits: dict[str, QubitConfig] = field(
        default_factory=lambda: {"q0": QubitConfig()}
    )
    resonators: dict[str, ResonatorConfig] = field(
        default_factory=lambda: {"q0": ResonatorConfig()}
    )
    couplings: list[CouplingConfig] = field(default_factory=list)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)

    # WHY: The discrete time step for the continuous QuTiP time grid.
    # 1.0 ns maps cleanly to the Qblox 1GSPS hardware sample rate.
    dt: float = 1.0e-9

    @property
    def qubit(self) -> QubitConfig:
        """
        Returns the primary 'q0' QubitConfig or first configured qubit.

        Returns:
            QubitConfig: The configuration of the primary qubit.
        """
        return self.qubits.get("q0", next(iter(self.qubits.values()), QubitConfig()))

    @property
    def resonator(self) -> ResonatorConfig:
        """
        Returns the primary 'q0' ResonatorConfig or first configured resonator.

        Returns:
            ResonatorConfig: The configuration of the primary resonator.
        """
        return self.resonators.get(
            "q0", next(iter(self.resonators.values()), ResonatorConfig())
        )

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "SimulationConfig":
        """
        Parses a multi-qubit topology configuration dictionary into typed dataclasses.

        Args:
            params (dict[str, Any]): The raw dictionary mapping component names to their parameters.

        Returns:
            SimulationConfig: The strongly-typed, instantiated configuration object.
        """
        # WHY: Recursively unpack inner dictionaries to ensure all sub-components
        # (qubits, resonators, couplings) are upgraded from raw dicts to proper dataclasses.
        qubits = {
            k: QubitConfig(**v) if isinstance(v, dict) else v
            for k, v in params.get("qubits", {}).items()
        }
        resonators = {
            k: ResonatorConfig(**v) if isinstance(v, dict) else v
            for k, v in params.get("resonators", {}).items()
        }
        couplings = [
            CouplingConfig(**c) if isinstance(c, dict) else c
            for c in params.get("couplings", [])
        ]
        acq = (
            AcquisitionConfig(**params["acquisition"])
            if "acquisition" in params
            else AcquisitionConfig()
        )

        return cls(
            qubits=qubits,
            resonators=resonators,
            couplings=couplings,
            acquisition=acq,
            dt=params.get("dt", 1.0e-9),
        )
