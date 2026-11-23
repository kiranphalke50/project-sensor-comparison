import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import pearsonr

# read the evoked responses for TMS+Dipole and Dipole-Only conditions
tms_evoked = mne.read_evokeds(
    r"T:\project-sensor_comparison\derivatives\sub-dipole17Hz\evoked_response\sub-dipole17Hz_task-dipoleTMS_acq-nmor1_run-01_000_raw_evoked-ave.fif"
)
dipole_evoked = mne.read_evokeds(
    r"T:\project-sensor_comparison\derivatives\sub-dipole17Hz\evoked_response\sub-dipole17Hz_task-dipoleOnly_acq-nmor1_run-01_000_raw_evoked-ave.fif"
)
dipole_frequency = 17  # Hz
times = tms_evoked[0].times
sfreq = tms_evoked[0].info["sfreq"]

# Recoverability Metric: Amplitude Recovery Ratio
def bandpass(data, sfreq, low=8, high=10):
    nyq = sfreq / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


# Filter the evoked responses in the 8-10 Hz range
x_filt = bandpass(
    dipole_evoked[0].data[0], sfreq, low=dipole_frequency - 1, high=dipole_frequency + 1
)  # slightly wider band for TMS+Dipole
y_filt = bandpass(
    tms_evoked[0].data[0], sfreq, low=dipole_frequency - 1, high=dipole_frequency + 1
)  # slightly wider band for TMS+Dipole

# Compute the envelope of the filtered signals using the Hilbert transform
env_dipole = np.abs(hilbert(x_filt))
env_tms = np.abs(hilbert(y_filt))
recovery_ratio = env_tms / env_dipole

# Plot the recovery ratio over time
plt.figure(figsize=(8, 4))
plt.plot(times, recovery_ratio)
plt.axhline(1, color="k", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Recovery Ratio")
plt.title("Dipole Recovery After TMS")
plt.show()
