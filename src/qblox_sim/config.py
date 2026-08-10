import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass(frozen=True)
class QubitConfig:
    """Physical parameters and Hilbert space dimensions for the Transmon."""
    f_q: float = 5.0e9
    f_d: float = 5.0e9
    alpha: float = -300.0e6
    N_q: int = 3
    rabi_freq_per_volt: float = 10.0e6
    T1: float = np.inf
    T2: float = np.inf

@dataclass(frozen=True)
class ResonatorConfig:
    """Physical parameters and Hilbert space dimensions for the Readout Resonator."""
    f_res: float = 6.0e9
    f_d_res: float = 6.0e9
    N_res: int = 5
    chi: float = 1.0e6
    kappa: float = 1.0e6
    rabi_freq_res_per_volt: float = 10.0e6

@dataclass(frozen=True)
class AcquisitionConfig:
    """Parameters governing measurement delays, noise, and state mapping."""
    noise_sigma: float = 0.02
    cable_delay: float = 120e-9
    v_0: complex = complex(0.05, 0.05)
    v_1: complex = complex(-0.05, -0.05)

@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration tying together the quantum system and solver settings."""
    qubit: QubitConfig = field(default_factory=QubitConfig)
    resonator: ResonatorConfig = field(default_factory=ResonatorConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    dt: float = 1.0e-9

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "SimulationConfig":
        """
        Backward-compatibility bridge. Parses the legacy flat 'params' dictionary 
        into the structured configuration dataclasses.
        """
        # Resolve dynamic fallback defaults
        f_q = params.get('f_q', 5.0e9)
        f_d = params.get('f_d', f_q)
        f_res = params.get('f_res', 6.0e9)
        f_d_res = params.get('f_d_res', f_res)

        qubit = QubitConfig(
            f_q=f_q,
            f_d=f_d,
            alpha=params.get('alpha', -300.0e6),
            N_q=params.get('N_q', 3),
            rabi_freq_per_volt=params.get('rabi_freq_per_volt', 10.0e6),
            T1=params.get('T1', np.inf),
            T2=params.get('T2', np.inf),
        )

        resonator = ResonatorConfig(
            f_res=f_res,
            f_d_res=f_d_res,
            N_res=params.get('N_res', 5),
            chi=params.get('chi', 1.0e6),
            kappa=params.get('kappa', 1e6),
            rabi_freq_res_per_volt=params.get('rabi_freq_res_per_volt', 10.0e6),
        )

        acquisition = AcquisitionConfig(
            cable_delay=params.get('cable_delay', 120e-9),
            noise_sigma=params.get('noise_sigma', 0.02),
            v_0=params.get('v_0', complex(0.05, 0.05)),
            v_1=params.get('v_1', complex(-0.05, -0.05)),
        )

        return cls(
            qubit=qubit,
            resonator=resonator,
            acquisition=acquisition,
            dt=params.get('dt', 1.0e-9)
        )