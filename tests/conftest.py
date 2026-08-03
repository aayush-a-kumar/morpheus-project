import pytest

@pytest.fixture
def default_qubit_params():
    """Default physical parameters for single qubit coupled to resonator."""
    return {
        'f_q': 5.0e9,
        'f_d': 5.0e9,
        'f_res': 6.0e9,
        'chi': 2.0e6,
        'rabi_freq_per_volt': 50e6,
        'rabi_freq_res_per_volt': 20e6,
        'T1': 5.0e-6,
        'T2': 10.0e-6,
        'kappa': 2.0e6,
        'N_res': 3
    }
