# This script estimates the recoverability of a dipole response after TMS using a sliding-window SNR metric. 
# It reads evoked responses for TMS+Dipole and Dipole-Only conditions, computes the SNR in a specified frequency band, and plots the results for both conditions over time.
# The frequency band for signal is chosen as a 2Hz window centered at the dipole frequency.
# The noise is defined as the power in the frequency ranges outside this signal band but within a broader range (3-40Hz).
# The SNR is calculated in sliding windows of specific length and step size. Adjust these parameters as needed for your analysis.

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

# Constants for SNR calculation
FMIN = 4  # Minimum frequency for noise calculation
FMAX = 40  # Maximum frequency for noise calculation
WINDOW_LENGTH = 0.5
WINDOW_STEP = 0.01

def compute_snr(segment, sfreq, dipole_frequency):
    # Compute the Power Spectral Density (PSD) of the segment using Welch's method
    freqs, psd = welch(segment, fs=sfreq, nperseg=len(segment))

    print(f"Frequencies computed: {freqs}")

    # Define the signal and noise frequency bands
    signal_idx = (freqs >= dipole_frequency - 1) & (freqs <= dipole_frequency + 1)  # 2Hz window
    noise_idx = ((freqs >= FMIN) & (freqs < dipole_frequency - 1)) | ((freqs > dipole_frequency + 1) & (freqs <= FMAX))

    # Calculate the average power in the signal and noise bands
    signal_power = np.mean(psd[signal_idx])
    noise_power = np.mean(psd[noise_idx])

    print(f"Signal Power: {signal_power}, Noise Power: {noise_power}")

    return 10 * np.log10(signal_power / noise_power)
    # return signal_power


if __name__ == "__main__":
    print("Starting SNR computation...")

    # read the evoked responses for TMS+Dipole and Dipole-Only conditions
    tms_evoked = mne.read_evokeds(
        r"T:\project-sensor_comparison\derivatives\sub-dipole9Hz\evoked_response\sub-dipole9Hz_task-dipoleTMS_acq-nmor1_run-01_000_raw_evoked-ave.fif"
    )
    dipole_evoked = mne.read_evokeds(
        r"T:\project-sensor_comparison\derivatives\sub-dipole9Hz\evoked_response\sub-dipole9Hz_task-dipoleOnly_acq-nmor1_run-01_000_raw_evoked-ave.fif"
    )
    dipole_evoked_data = dipole_evoked[0].data[0]
    tms_evoked_data = tms_evoked[0].data[0]

    # constants for the SNR computation
    dipole_frequency = 9  # Hz
    times = tms_evoked[0].times
    sfreq = tms_evoked[0].info["sfreq"]
    win_samples = int(WINDOW_LENGTH * sfreq)
    step_samples = int(WINDOW_STEP * sfreq)

    # SNR containers
    snr_times = []
    snr_tms = []
    snr_dipole = []

    for start in range(0, len(dipole_evoked_data) - win_samples, step_samples):
        stop = start + win_samples
        print(f"Computing SNR for window: {start} to {stop}")

        snr_dipole.append(compute_snr(dipole_evoked_data[start:stop], sfreq, dipole_frequency))

        snr_tms.append(compute_snr(tms_evoked_data[start:stop], sfreq, dipole_frequency))

        snr_times.append(times[start + win_samples // 2])

    plt.figure(figsize=(8, 4))
    plt.plot(snr_times, snr_dipole, label="Dipole only")
    plt.plot(snr_times, snr_tms, label="Dipole + TMS")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("SNR (dB)")
    plt.title(
        f"Dipole Recovery After TMS (Expressed as SNR where signal is 2Hz window centered at {dipole_frequency}Hz)"
    )
    plt.grid()
    plt.show()
