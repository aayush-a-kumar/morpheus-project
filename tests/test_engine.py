import pytest
import numpy as np
import qutip
from qblox_sim.config import SimulationConfig
from qblox_sim.physics import QuantumSystem
from qblox_sim.engine import QuTiPEngine

@pytest.fixture
def default_system():
    """Provides a baseline QuantumSystem instance."""
    cfg = SimulationConfig()
    return QuantumSystem(cfg)

def test_qutip_engine_basic_execution(default_system):
    """Tests that the engine correctly integrates a zero-drive schedule."""
    engine = QuTiPEngine()
    
    # Create a short 10ns time grid
    t_list = np.linspace(0, 10e-9, 10)
    
    # Provide zero drives
    drives = {
        "q_drive": np.zeros_like(t_list, dtype=complex),
        "res_drive": np.zeros_like(t_list, dtype=complex)
    }
    
    initial_state = default_system.get_default_initial_state()
    
    # Run the solver
    result = engine.run(
        system=default_system, 
        drives=drives, 
        t_list=t_list, 
        initial_state=initial_state
    )
    
    # Verify the result object returned states for every time step
    assert result is not None
    assert hasattr(result, 'states')
    assert len(result.states) == len(t_list)
    
    # Since there are no drives and the system started in the ground state,
    # it should remain entirely in the ground state.
    final_state = result.states[-1]
    rho_q = final_state.ptrace(0) if final_state.type == 'oper' else qutip.ket2dm(final_state).ptrace(0)
    
    N_q = default_system.cfg.qubit.N_q
    prob_0 = float(np.real(qutip.expect(qutip.ket2dm(qutip.basis(N_q, 0)), rho_q)))
    
    np.testing.assert_allclose(prob_0, 1.0, atol=1e-5)

def test_qutip_engine_options_passthrough(default_system):
    """Tests that custom solver options are successfully passed to mesolve."""
    engine = QuTiPEngine()
    t_list = np.linspace(0, 1e-9, 2)
    drives = {
        "q_drive": np.zeros_like(t_list, dtype=complex),
        "res_drive": np.zeros_like(t_list, dtype=complex)
    }
    initial_state = default_system.get_default_initial_state()
    
    # Set a custom option that QuTiP will recognize
    custom_options = {"nsteps": 1000, "store_states": True}
    
    result = engine.run(
        system=default_system,
        drives=drives,
        t_list=t_list,
        initial_state=initial_state,
        options=custom_options
    )
    
    assert len(result.states) == 2