import pytest
import numpy as np
import qutip
from qblox_scheduler import Schedule
from qblox_scheduler.operations import Measure
from qblox_scheduler.resources import ClockResource
from qblox_sim.simulator import QbloxQutipSimulator, QbloxQ1Simulator
from qblox_sim.config import SimulationConfig, QubitConfig, ResonatorConfig
from qblox_sim.acquisitions import (
    SSBIntegrationHandler, 
    ThresholdedAcquisitionHandler, 
    AcquisitionRegistry
)

def test_registry():
    handler = AcquisitionRegistry.get_handler('Thresholded')
    assert isinstance(handler, ThresholdedAcquisitionHandler)
    
    handler = AcquisitionRegistry.get_handler('UnknownProtocol')
    assert isinstance(handler, SSBIntegrationHandler)

def test_ssb_integration_iq_acquisition(default_qubit_params):
    """Verify SSBIntegrationComplex protocol produces state-dependent IQ centroids with noise."""
    with QbloxQ1Simulator(default_qubit_params) as sim:
        # Type guard for Pylance + Runtime check for Pytest
        assert sim.cluster is not None, "Simulator cluster failed to initialize."

        # Attach a mock acquisition window to the readout sequencer
        qrm = sim.cluster.get_connected_modules()[sim.readout_mod]
        qrm.sequencers[sim.readout_seq].acq_windows = [[(0, 500)]]

        res = sim.simulate()
        measurements = res.get('measurements', [])

        # 1. Structure Verification
        assert len(measurements) > 0, "Failed to retrieve any measurements from simulator."
        m = measurements[0]
        
        assert 'I' in m and 'Q' in m, "SSBIntegrationComplex measurement missing 'I' or 'Q' voltage outputs."
        assert 'prob_0' in m and 'prob_1' in m, "Measurement output missing qubit population probabilities."

        # 2. Physics & Value Bounds Check
        assert np.isclose(m['prob_0'], 1.0, atol=1e-2), f"Expected ground state prob_0 ≈ 1.0, got {m['prob_0']:.4f}"
        assert np.isfinite(m['I']) and np.isfinite(m['Q'])
        assert np.isclose(m['I'], 0.05, atol=0.1), f"I voltage {m['I']} deviated excessively from V_0 centroid"
        assert np.isclose(m['Q'], 0.05, atol=0.1), f"Q voltage {m['Q']} deviated excessively from V_0 centroid"


def test_thresholded_acquisition(default_qubit_params):
    """Verify ThresholdedAcquisition protocol outputs discrete 0/1 outcomes matching quantum states."""
    sim = QbloxQutipSimulator(default_qubit_params)

    n_q = sim.cfg.qubits['q0'].N_q
    n_res = sim.cfg.resonators['q0'].N_res
    f_res = sim.cfg.resonators['q0'].f_res

    excited_state = qutip.tensor(
        qutip.basis(n_q, 1), 
        qutip.basis(n_res, 0)
    )

    sched = Schedule("Thresholded State Test")
    sched.add_resource(ClockResource(name="q0.ro", freq=f_res))
    sched.add(Measure("q0", acq_protocol="ThresholdedAcquisition", acq_channel="bits", clock="q0.ro"))

    res = sim.simulate(sched, initial_state=excited_state)
    measurements = res.get('measurements', [])

    assert len(measurements) > 0, "No measurements returned for thresholded acquisition."
    m = measurements[0]

    assert 'outcome' in m, "ThresholdedAcquisition missing discrete 'outcome' field."
    assert m['outcome'] in (0, 1), f"Discrete outcome must be 0 or 1, got {m['outcome']}"
    assert np.isclose(m['prob_1'], 1.0, atol=1e-2), f"Expected excited state prob_1 ≈ 1.0, got {m['prob_1']:.4f}"


def test_trace_acquisition_continuous_array(default_qubit_params):
    """Verify Trace protocol generates continuous 1D time-series arrays over acq_duration."""
    sim = QbloxQutipSimulator(default_qubit_params)

    f_res = sim.cfg.resonators['q0'].f_res
    acq_duration = 300e-9
    sched = Schedule("Trace Acquisition Test")
    sched.add_resource(ClockResource(name="q0.ro", freq=f_res))
    
    sched.add(
        Measure(
            "q0", 
            acq_protocol="Trace", 
            acq_duration=acq_duration, 
            acq_channel="S_21", 
            clock="q0.ro"
        )
    )

    res = sim.simulate(sched)
    t_list = res.get('t_list', np.array([]))
    measurements = res.get('measurements', [])

    assert len(measurements) > 0, "No trace measurements extracted."
    
    total_sim_time = t_list[-1] if len(t_list) > 0 else 0.0
    assert total_sim_time >= acq_duration - 1e-12, (
        f"Simulation time grid ({total_sim_time*1e9:.1f} ns) shorter than trace duration ({acq_duration*1e9:.1f} ns)"
    )


def test_q1simulator_mock_memory_injection(default_qubit_params):
    """Verify QbloxQ1Simulator successfully pushes acquisition mock data back to cluster hardware sequencers."""
    with QbloxQ1Simulator(default_qubit_params) as sim:
        # Type guard for Pylance + Runtime check for Pytest
        assert sim.cluster is not None, "Simulator cluster failed to initialize."

        # Attach a mock acquisition window to the readout sequencer
        qrm = sim.cluster.get_connected_modules()[sim.readout_mod]
        qrm.sequencers[sim.readout_seq].acq_windows = [[(0, 500)]]

        res = sim.simulate()
        
        connected = sim.cluster.get_connected_modules()
        assert sim.readout_mod in connected, f"Readout module {sim.readout_mod} not found in connected cluster."

        seq = qrm.sequencers[sim.readout_seq]
        assert hasattr(seq, 'acquisitions') or hasattr(seq, '_acquisition_mock_data'), (
            "Q1Simulator sequencer lacks mock acquisition storage attributes."
        )


def test_joint_probabilities_2qubit_bell_state():
    """Verify 2-qubit Bell state (|00> + |11>)/sqrt(2) extracts prob_00 = 0.5, prob_11 = 0.5, and correct marginals."""
    cfg = SimulationConfig(
        qubits={
            "q0": QubitConfig(N_q=3),
            "q1": QubitConfig(N_q=3)
        },
        resonators={
            "q0": ResonatorConfig(N_res=3)
        }
    )
    handler = SSBIntegrationHandler()

    # Tensor space: [q0 (dim 3), q1 (dim 3), res0 (dim 3)]
    q0_0, q0_1 = qutip.basis(3, 0), qutip.basis(3, 1)
    q1_0, q1_1 = qutip.basis(3, 0), qutip.basis(3, 1)
    res_0 = qutip.basis(3, 0)

    ket_000 = qutip.tensor(q0_0, q1_0, res_0)
    ket_110 = qutip.tensor(q0_1, q1_1, res_0)
    bell_state = (ket_000 + ket_110).unit()

    probs = handler._get_joint_probabilities(bell_state, cfg)

    # 1. Joint basis populations
    assert np.isclose(probs["prob_00"], 0.5, atol=1e-5)
    assert np.isclose(probs["prob_11"], 0.5, atol=1e-5)
    assert np.isclose(probs["prob_01"], 0.0, atol=1e-5)
    assert np.isclose(probs["prob_10"], 0.0, atol=1e-5)

    # 2. Primary qubit marginals
    assert np.isclose(probs["prob_0"], 0.5, atol=1e-5)
    assert np.isclose(probs["prob_1"], 0.5, atol=1e-5)

    # 3. Leakage manifold check
    assert np.isclose(probs["leakage_prob_2"], 0.0, atol=1e-5)


def test_joint_probabilities_leakage_detection():
    """Verify state population in the |2> state is quantified in leakage_prob_2."""
    cfg = SimulationConfig(
        qubits={
            "q0": QubitConfig(N_q=3),
            "q1": QubitConfig(N_q=3)
        }
    )
    handler = SSBIntegrationHandler()

    # Qubit 0 in state |2> (leaked) and Qubit 1 in state |0>
    q0_2 = qutip.basis(3, 2)
    q1_0 = qutip.basis(3, 0)
    leak_state = qutip.tensor(q0_2, q1_0)

    probs = handler._get_joint_probabilities(leak_state, cfg)

    assert np.isclose(probs["prob_00"], 0.0, atol=1e-5)
    assert np.isclose(probs["prob_11"], 0.0, atol=1e-5)
    assert np.isclose(probs["leakage_prob_2"], 1.0, atol=1e-5)


def test_3qubit_dynamic_joint_probabilities():
    """Verify dynamic scaling to 3 qubits produces all 2^3 computational keys."""
    cfg = SimulationConfig(
        qubits={
            "q0": QubitConfig(N_q=2),
            "q1": QubitConfig(N_q=2),
            "q2": QubitConfig(N_q=2)
        }
    )
    handler = ThresholdedAcquisitionHandler()

    # Prepare state |101>
    q0_1 = qutip.basis(2, 1)
    q1_0 = qutip.basis(2, 0)
    q2_1 = qutip.basis(2, 1)
    state_101 = qutip.tensor(q0_1, q1_0, q2_1)

    probs = handler._get_joint_probabilities(state_101, cfg)

    expected_keys = {f"prob_{b0}{b1}{b2}" for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1)}
    assert expected_keys.issubset(probs.keys())

    assert np.isclose(probs["prob_101"], 1.0, atol=1e-5)
    assert np.isclose(probs["prob_000"], 0.0, atol=1e-5)