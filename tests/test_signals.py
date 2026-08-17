# SPDX-License-Identifier: LicenseRef-Proprietary
import numpy as np
from qblox_sim.signals import ScheduleSignalProvider, extract_amplitude


def test_extract_amplitude():
    assert extract_amplitude({'amp': 0.5}) == 0.5
    assert extract_amplitude({'amplitude': 0.75}) == 0.75
    assert extract_amplitude({'missing': 1.0}, default=0.1) == 0.1


def test_schedule_signal_provider_square_pulse():
    t_list = np.linspace(0, 100e-9, 100)
    pulses = [
        {'port': 'q0:mw', 'abs_time': 10e-9, 'duration': 20e-9, 'amp': 0.5, 'phase': 0, 'wf_func': 'square'}
    ]
    
    provider = ScheduleSignalProvider(pulses)
    drives = provider.get_drives(t_list)
    
    q_drive = drives.get("q0:mw", np.zeros_like(t_list))
    res_drive = drives.get("q0:res", np.zeros_like(t_list))
    
    assert len(q_drive) == 100
    assert np.max(np.real(q_drive)) == 0.5
    assert np.all(res_drive == 0.0)


def test_schedule_signal_provider_gauss_pulse():
    t_list = np.linspace(0, 100e-9, 100)
    pulses = [
        {'port': 'q0:res', 'abs_time': 0, 'duration': 40e-9, 'amp': 1.0, 'wf_func': 'gauss', 'sigma': 10e-9}
    ]
    
    provider = ScheduleSignalProvider(pulses)
    drives = provider.get_drives(t_list)
    
    res_drive = drives.get("q0:res", np.zeros_like(t_list))
    # Check that peak is near 1.0
    assert np.max(np.abs(res_drive)) > 0.9