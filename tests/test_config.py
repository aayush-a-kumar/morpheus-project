from qblox_sim.config import SimulationConfig

def test_config_defaults():
    cfg = SimulationConfig()
    assert cfg.qubit.f_q == 5.0e9
    assert cfg.qubit.N_q == 3
    assert cfg.resonator.N_res == 5

def test_config_from_legacy_dict():
    legacy_params = {'f_q': 6.2e9, 'N_q': 2, 'noise_sigma': 0.05}
    cfg = SimulationConfig.from_dict(legacy_params)
    
    # Custom values updated
    assert cfg.qubit.f_q == 6.2e9
    assert cfg.qubit.N_q == 2
    assert cfg.acquisition.noise_sigma == 0.05
    # Unspecified values fall back to defaults
    assert cfg.resonator.f_res == 6.0e9