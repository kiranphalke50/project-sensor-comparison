import matplotlib.pyplot as plt
import mne
import numpy as np
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

# Recoverability Metric: Sliding-window Correlation with Dipole-Only Response

# constants
window_length = 0.2  # sec
step = 0.01

win_samples = int(window_length * sfreq)
step_samples = int(step * sfreq)

corr_times = []
corr_values = []

# x = dipole_evoked[0].data[0]
y = tms_evoked[0].data[0]

for start in range(0, len(times) - win_samples, step_samples):
    stop = start + win_samples

    # ideal 9 Hz dipole
    t_window = times[start:stop]
    x = np.sin(2 * np.pi * dipole_frequency * t_window)

    # r, _ = pearsonr(x[start:stop], y[start:stop])
    r, _ = pearsonr(x, y[start:stop])

    corr_values.append(r)
    corr_times.append(times[start + win_samples // 2])

plt.figure(figsize=(8, 4))
plt.plot(corr_times, corr_values)
plt.xlabel("Time (s)")
plt.ylabel("Correlation")
plt.title(
    f"Dipole+TMS vs Pure Dipole Correlation (Sliding Window {window_length}s and Step {step}s)"
)
plt.axvline(0, color="k", linestyle="--")
plt.show()
