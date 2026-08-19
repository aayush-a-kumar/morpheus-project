# SPDX-FileCopyrightText: © 2026 Qblox <https://qblox.com>
# SPDX-License-Identifier: LicenseRef-Qblox
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QubitConfig:
    f_q: float = 5.0e9
    f_d: float = 5.0e9
    alpha: float = -300.0e6
    N_q: int = 3
    rabi_freq_per_volt: float = 10.0e6
    flux_freq_per_volt: float = 1.0e9
    T1: float = np.inf
    T2: float = np.inf
    # --- New Non-Linear Flux Parameters ---
    v_phi0: float | None = None  # Voltage required for a single flux quantum
    f_max: float | None = None  # The maximum 'sweet spot' frequency


@dataclass(frozen=True)
class ResonatorConfig:
    f_res: float = 6.0e9
    f_d_res: float = 6.0e9
    N_res: int = 5
    chi: float = 1.0e6
    kappa: float = 1.0e6
    rabi_freq_res_per_volt: float = 10.0e6


@dataclass(frozen=True)
class CouplingConfig:
    """Defines the static exchange coupling between two qubits."""

    q1: str
    q2: str
    J: float = 0.0


@dataclass(frozen=True)
class AcquisitionConfig:
    noise_sigma: float = 0.02
    cable_delay: float = 120e-9
    v_0: complex = complex(0.05, 0.05)
    v_1: complex = complex(-0.05, -0.05)


@dataclass(frozen=True)
class SimulationConfig:
    qubits: dict[str, QubitConfig] = field(
        default_factory=lambda: {"q0": QubitConfig()}
    )
    resonators: dict[str, ResonatorConfig] = field(
        default_factory=lambda: {"q0": ResonatorConfig()}
    )
    couplings: list[CouplingConfig] = field(default_factory=list)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    dt: float = 1.0e-9

    @property
    def qubit(self) -> QubitConfig:
        """Returns the primary 'q0' QubitConfig or first configured qubit."""
        return self.qubits.get("q0", next(iter(self.qubits.values()), QubitConfig()))

    @property
    def resonator(self) -> ResonatorConfig:
        """Returns the primary 'q0' ResonatorConfig or first configured resonator."""
        return self.resonators.get(
            "q0", next(iter(self.resonators.values()), ResonatorConfig())
        )

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "SimulationConfig":
        """Parses a multi-qubit topology configuration dictionary."""

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
