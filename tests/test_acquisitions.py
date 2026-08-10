import pytest
import numpy as np
import qutip
from qblox_sim.config import SimulationConfig
from qblox_sim.acquisitions import (
    SSBIntegrationHandler, 
    ThresholdedAcquisitionHandler, 
    AcquisitionRegistry
)

@pytest.fixture
def base_config():
    return SimulationConfig()

@pytest.fixture
def operators():
    a = qutip.tensor(qutip.identity(3), qutip.destroy(5))
    ad = a.dag()
    return a, ad

@pytest.fixture
def ground_state():
    return qutip.tensor(qutip.basis(3, 0), qutip.basis(5, 0))

def test_registry():
    handler = AcquisitionRegistry.get_handler('Thresholded')
    assert isinstance(handler, ThresholdedAcquisitionHandler)
    
    # Fallback to SSB
    handler = AcquisitionRegistry.get_handler('UnknownProtocol')
    assert isinstance(handler, SSBIntegrationHandler)

def test_ssb_integration(base_config, operators, ground_state):
    a, ad = operators
    handler = SSBIntegrationHandler()
    
    res = handler.process(
        state=ground_state, 
        t_list=np.array([0.0]), 
        states=[ground_state], 
        acq_time=0.0, 
        acq_duration=1e-6, 
        acq_info={}, 
        cfg=base_config, 
        a_op=a, 
        ad_op=ad
    )
    
    assert res['prob_0'] == 1.0
    assert res['prob_1'] == 0.0
    assert res['outcome'] == 0
    # The expected ground state voltage centroid is v_0 = 0.05 + 0.05j
    assert np.isclose(res['I'], 0.05, atol=0.1) # Account for noise

def test_thresholded_acquisition(base_config, operators, ground_state):
    a, ad = operators
    handler = ThresholdedAcquisitionHandler()
    
    # Ground state should fail a threshold check
    res = handler.process(
        state=ground_state, 
        t_list=np.array([0.0]), 
        states=[ground_state], 
        acq_time=0.0, 
        acq_duration=1e-6, 
        acq_info={'acq_rotation': 0, 'acq_threshold': 0.1}, 
        cfg=base_config, 
        a_op=a, 
        ad_op=ad
    )
    
    assert res['outcome'] == 0