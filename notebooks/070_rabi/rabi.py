# %% [markdown]
# <a href="dependencies/banner.jpeg">
#       <img src="dependencies/banner.jpeg" alt="image0" width="100%">
# </a>

# %% [markdown]
# # Rabi
# After determining the qubit's |0⟩ $\rightarrow$ |1⟩ transition frequency, a Rabi experiment
# is performed to calibrate the required microwave drive amplitude. The frequency and
# duration of the pulse are kept constant while the amplitude is swept, leading to oscillations
# in the qubit state. The power level that first fully inverts the qubit's population (a
# $\pi$-pulse) is then identified.

# %%
from dependencies.analysis_utils import RabiAnalysis
from xarray import open_dataset

from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.experiments import SetHardwareOption
from qblox_scheduler.operations import Measure, Reset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

# %% [markdown]
# ## Setup
# The hardware agent manages the connection to the instrument and ensures that pulses and acquisitions happen over the appropriate input and output channels of the Cluster.
# The cell below creates an instance of the `HardwareAgent` based on the hardware- and device-under-test configuration files in the `./dependencies/configs` folder, allowing us to start doing measurements.
# We also define some convenient aliases to use throughout our measurements.
# For a more thorough discussion of the hardware- and device-under-test configuration files, check out [this tutorial](000_transmon_setup.ipynb).

# %%
# Set up hardware agent, this automatically connects to the instrument
hw_agent = HardwareAgent(
    hardware_configuration="./dependencies/configs/hw_config.json",
    quantum_device_configuration="./dependencies/configs/dut_config.json",
)

# convenience aliases
q0 = hw_agent.quantum_device.get_element("q0")  # Qubits 0 and 2 are measured using QRM-RF + QCM-RF
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")  # Qubit 3 is measured using QRC

cluster = hw_agent.get_clusters()["cluster"]
hw_options = hw_agent.hardware_configuration.hardware_options
qubit = q0

# %% [markdown]
# ## Experiment settings

# %%
# Drive attenuation settings. Should be an even number <= 30
drive_att = 12  # dB

# Rabi settings
amp_start = -0.5  # a.u.
amp_stop = 0.5  # a.u.
amp_npoints = 100

repetitions = 1000

# %% [markdown]
# ## Experiment schedule

# %%
rabi_power_sched = Schedule("power_rabi")
rabi_power_sched.add(SetHardwareOption("output_att", drive_att, f"{qubit.name}:mw-{qubit.name}.01"))

with (
    rabi_power_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
    rabi_power_sched.loop(
        linspace(start=amp_start, stop=amp_stop, num=amp_npoints, dtype=DType.AMPLITUDE)
    ) as amp,
):
    rabi_power_sched.add(Reset(qubit.name))
    # Play pulse of varying amplitude
    rabi_power_sched.add(X(qubit=qubit.name, amp180=amp))
    rabi_power_sched.add(Measure(qubit.name, coords={"amplitude": amp}, acq_channel="S_21"))

# Execute the experiment
rabi_data = hw_agent.run(rabi_power_sched)
if cluster.is_dummy:
    example_data = open_dataset("./dependencies/datasets/rabi.hdf5", engine="h5netcdf")
    rabi_data = rabi_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment

# %%
rabi_analysis = RabiAnalysis(rabi_data).run()
rabi_analysis.display_figs_mpl()

# %% [markdown]
# ## Post-run

# %%
# Update device config
qubit.rxy.amp180 = rabi_analysis.quantities_of_interest["Pi-pulse amplitude"].nominal_value
hw_options.output_att[f"{qubit.name}:mw-{qubit.name}.01"] = drive_att

# %% [markdown]
# #### Update the device configuration file
# After measurement, we may store the measured device properties inside a new file to use in future experiments.
# The time-unique identifier ensures that it is easy to find back previously found measurement results.

# %%
hw_agent.quantum_device.to_json_file("./dependencies/configs", add_timestamp=True)


