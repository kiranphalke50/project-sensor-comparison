# script to process raw data collected on the NMOR-OPM sensor - in an empty room - without TMS clicks - without current dipole - thus no artifacts are expected
# Main goal is to compute the intrinsic noise floor of the NMOR-OPM gradiometer in an empty room
# Steps taken :
# 1. Load the raw data collected in an empty room
# 2. Create epochs of size 1 sec each
# 3. Compute the power spectral density (PSD) of the epochs : using Welch's method with a frequency range of 1-100Hz, using entire 1 sec epoch length for FFT computation
# 4. Convert the PSD to amplitude spectral density (ASD)
# 5. Convert the ASD to fT/sqrt(Hz)/cm using cell separation distance of 4 cm
# 6. Average the ASD across all epochs
# 7. Report the average noise floor across the frequency range of interest (2-48Hz) and (8-30Hz). Frequency range chosen for physiological relevance.

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

matplotlib.use("qtagg")

# constants
CELL_SEP = 5.0  # cm, distance between two cells in a gradiometer
EPOCH_LENGTH = 1.0  # sec, length of each epoch
FMIN = 1.0  # Hz, minimum frequency for PSD computation
FMAX = 100.0  # Hz, maximum frequency for PSD computation
FMIN_ASD_PLOT = 2.0  # Hz, minimum frequency for plotting ASD
FMAX_ASD_PLOT = 50.0  # Hz, maximum frequency for plotting ASD
FMIN_REPORTING_ASD = 2.0  # Hz, minimum frequency for reporting ASD
FMAX_REPORTING_ASD = 48.0  # Hz, maximum frequency for reporting ASD

# Load the TMS-OPM dataset (T:\project-sensor_comparison\pilot_data\sub-dipole9Hz_task-empty_acq-nmor11fs1_run-02_000\concat\mne_raw)
# project folder (check if the BLUEBEAR environment variable is set to 1, only set when calling the script from .sh script on BlueBEAR)
bluebear = os.environ.get("BLUEBEAR", "0") == "1"
base_folder = Path(r"/rds/projects/k/kowalcau-tms-opm-meg/project-sensor_comparison") if bluebear else Path(r"T:\project-sensor_comparison")

# subject list
subject_list = ["sub-all", "sub-dipole9Hz", "sub-dipole17Hz", "sub-dipole32Hz", "sub-dipole1kHz"]

# input/output paths (input : base/data/subject/meg/**emptyroom*_raw.fif, output : base/derivatives/subject)
for subject in subject_list:
    subject_meg_folder = base_folder / "data" / subject / "meg"
    subject_output_folder = base_folder / "derivatives" / subject
    os.makedirs(subject_output_folder, exist_ok=True)

    if not subject_meg_folder.exists():
        print(f"Subject meg folder does not exist: {subject_meg_folder}")
        continue

    # look for files containing "emptyroom" in the filename and ending with "_raw.fif"
    emptyroom_files = list(subject_meg_folder.glob("*emptyroom*_raw.fif"))

    if not emptyroom_files:
        print(f"No emptyroom files found for subject: {subject}")
        continue

    # Process all the empty room files for the subject
    for emptyroom_file in emptyroom_files:
        print(f"Processing emptyroom file: {emptyroom_file}")

        # extract the filename without extension to use for output plot filename
        emptyroom_filename = os.path.basename(emptyroom_file).split(".")[0]

        raw = mne.io.read_raw_fif(emptyroom_file, preload=True)

        # Create epochs of size 1 sec each
        events = mne.make_fixed_length_events(raw, duration=EPOCH_LENGTH)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_LENGTH, baseline=None, preload=True)

        # Compute PSD
        spectrum = epochs.compute_psd(
            method="welch", fmin=FMIN, fmax=FMAX, n_fft=int(raw.info["sfreq"]), picks="meg"
        )

        psd, freqs = spectrum.get_data(return_freqs=True)

        # PSD [T²/Hz] -> ASD [T/sqrt(Hz)]
        asd = np.sqrt(psd)

        # Convert to fT/sqrt(Hz)/cm
        asd *= 1e15 / CELL_SEP

        # Average ASD across all epochs
        asd = np.mean(asd, axis=0)

        # Average noise floor across the frequency range of interest (2-48Hz)
        average_noise_floor = np.mean(
            asd[0][(freqs >= FMIN_REPORTING_ASD) & (freqs <= FMAX_REPORTING_ASD)]
        )
        print(f"Average noise floor (2-48Hz): {average_noise_floor:.2f} fT/√Hz/cm")

        median_noise_floor = np.median(
            asd[0][(freqs >= FMIN_REPORTING_ASD) & (freqs <= FMAX_REPORTING_ASD)]
        )
        print(f"Median noise floor (2-48Hz): {median_noise_floor:.2f} fT/√Hz/cm")

        # Plot

        freq_range = (freqs >= FMIN_ASD_PLOT) & (freqs <= FMAX_ASD_PLOT)
        plt.figure(figsize=(8, 5))
        plt.semilogy(freqs[freq_range], asd[0][freq_range], lw=2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("ASD (fT/√Hz/cm)")
        plt.ylim(1e0, 1e3)
        plt.xlim(FMIN_ASD_PLOT, FMAX_ASD_PLOT)
        plt.title(
            f"Empty-room Sensor noise floor (2-48Hz in fT/√Hz/cm): mean={average_noise_floor:.2f}, median={median_noise_floor:.2f}"
        )
        plt.grid(True)

        # save the plot in the subject output folder
        plot_filename = subject_output_folder / f"{emptyroom_filename}_asd_plot.png"
        plt.savefig(plot_filename)
