# script to process raw data collected on the NMOR-OPM sensor - in an empty room - with TMS clicks
# contains only single channel OPM data, thus preprocessing includes:
# 1. load data
# 2. cant apply HFC because no location information is available for single channel OPM data
# 3. cant apply ICA because currently no human participant is involved in the experiment, thus no artifacts are expected, also a single sensor cannot be used for ICA decomposition
# 4. epoch data - around TMS clicks
# 5. TMS artefact handling - needs to be done in time domain, before any kind of brain-like preprocessing steps
#    Masking the pulse till 5-10ms (exact time to be found after several rounds) - could be as early as 8ms such that the TMS click has entered its decay phase
#    Fitting exponential decay function to the TMS artefact and subtracting it from the data
#    Applying polynomial detrending to the data to remove any slow drifts in the data
# 6. filter data - bandpass filter between 3-40Hz and notch filter at 50Hz and its harmonics to see how the TMS clicks look like in the frequency range of interest
# 7. After every step, plot the time-domain data and TFRs to see the effect of each step
# 8. Once recoverability metric script is ready - plot dipole recoverability after each step of preprocessing

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

# matplotlib.use("qtagg")

# constants
EPOCH_START = -0.1  # seconds
EPOCH_END = 2.0  # seconds
TMS_MASK_START = 0 # seconds - start of TMS artefact masking
TMS_MASK_END = 0.010 # seconds - end of TMS artefact masking
DETRENDING_POLY_ORDER = 4 # order of polynomial for detrending

# project folder
bluebear = True
base_folder = (
    Path(r"/rds/projects/k/kowalcau-tms-opm-meg/project-sensor_comparison")
    if bluebear
    else Path(r"T:\project-sensor_comparison")
)

# subject list
subject_list = ["sub-all", "sub-dipole9Hz", "sub-dipole17Hz", "sub-dipole32Hz", "sub-dipole1kHz"]

# input/output paths
# input : base/data/subject/meg/**dipoleOnly*_raw.fif or base/data/subject/meg/**dipoleTMS*_raw.fif
# output : base/derivatives/subject/evoked_response and base/derivatives/subject/figures
for subject in subject_list:
    subject_meg_folder = base_folder / "data" / subject / "meg"
    subject_output_folder = base_folder / "derivatives" / subject
    os.makedirs(subject_output_folder, exist_ok=True)

    if not subject_meg_folder.exists():
        print(f"Subject meg folder does not exist: {subject_meg_folder}")
        continue

    # look for files containing "dipoleOnly" or "dipoleTMS" in the filename and ending with "_raw.fif"
    meg_data_files = list(subject_meg_folder.glob("*dipoleOnly*_raw.fif")) + list(
        subject_meg_folder.glob("*dipoleTMS*_raw.fif")
    )

    if not meg_data_files:
        print(f"No dipoleOnly or dipoleTMS MEG files found for subject: {subject}")
        continue

    # process all the MEG files for the subject
    for meg_data_file in meg_data_files:
        print(f"Processing MEG data file: {meg_data_file}")

        # extract the filename without extension to use for output plot filename
        meg_data_filename = os.path.basename(meg_data_file).split(".")[0]

        # Create plot and results folders for the subject if they don't exist
        plot_folder = subject_output_folder / "figures"
        plot_folder.mkdir(parents=True, exist_ok=True)
        evoked_response_folder = subject_output_folder / "evoked_response"
        evoked_response_folder.mkdir(parents=True, exist_ok=True)

        raw = mne.io.read_raw_fif(meg_data_file, preload=True)
        print(raw.info)

        # Find events
        events = mne.find_events(raw, stim_channel="trigin1")
        event_dict = {"tms_pulse": 1}

        # Generate epochs from the raw data based on the events found, with a specified time window around each event
        # Baseline correction is applied using the time window from 1.5 to 2.0 seconds after the event (because pre-TMS period is contaminated by TMS recharge artefact)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=EPOCH_START,
            tmax=EPOCH_END,
            baseline=(1.5, 2.0),
            preload=True,
        )
        tms_epochs_orig = epochs["tms_pulse"]
        tms_evoked_orig = tms_epochs_orig.average()

        # save epochs and evoked to a file in the subject_output_folder/evoked_response folder
        tms_epochs_orig.save(evoked_response_folder / f"{meg_data_filename}_epochs_orig-epo.fif", overwrite=True)
        tms_evoked_orig.save(evoked_response_folder / f"{meg_data_filename}_evoked_orig-ave.fif", overwrite=True)

        # Plot all single epochs with the averaged epoch overlaid in bold
        single_epoch_data = tms_epochs_orig.get_data()[
            :, 0, :
        ]  # number of epochs x number of time points
        averaged_epoch_data = tms_evoked_orig.get_data()  # 1 x number of time points
        fig, ax = plt.subplots(figsize=(10, 6))
        for epoch in single_epoch_data:
            ax.plot(tms_epochs_orig.times, epoch * 1e15, color="lightgray", alpha=0.5)
        ax.plot(
            tms_epochs_orig.times,
            averaged_epoch_data[0] * 1e15,
            color="red",
            linewidth=2,
            label="Average",
        )
        ax.set_title("TMS Evoked Response - Single Epochs with Average")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (fT)")
        plt.legend()
        # plt.show()

        # filter the evoked response between 3-40Hz to see how the TMS clicks look like in the frequency range of interest
        # limit the epochs to tmin to tmax for filtering such that the filter does not introduce edge artifacts in the time window of interest
        fmin = 3
        fmax = 40
        tmin = 0.06
        tmax = 2.0
        tms_epochs_orig_cropped = tms_epochs_orig.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)
        tms_epochs_orig_filtered = tms_epochs_orig_cropped.copy().filter(
            l_freq=fmin, h_freq=fmax, fir_design="firwin"
        )
        tms_evoked_orig_filtered = tms_epochs_orig_filtered.average()

        # Compute required TFRs
        tfr_unfiltered = mne.time_frequency.tfr_morlet(
            tms_evoked_orig,
            freqs=np.arange(fmin, fmax, 1),
            return_itc=False,
            n_cycles=2,
            average=True,
        )
        tfr_cropped_filtered = mne.time_frequency.tfr_morlet(
            tms_evoked_orig_filtered,
            freqs=np.arange(fmin, fmax, 1),
            return_itc=False,
            n_cycles=2,
            average=True,
        )

        # Subplot the evoked response and the TFR - for the unfiltered, uncropped, and uncorrected averaged epoch
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(tms_evoked_orig.times, tms_evoked_orig.data.T * 1e15, color="red")
        axes[0, 0].set_title("TMS Evoked Response - Unfiltered, Uncropped, Uncorrected")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude (fT)")
        axes[0, 0].set_xlim([0, 2])
        axes[0, 0].set_ylim([-2000, 2000]) # in fT, to ensure that the TMS pulse and the dipole are visible in the plot
        axes[1, 0].pcolormesh(
            tfr_unfiltered.times,
            tfr_unfiltered.freqs,
            tfr_unfiltered.data[0],
            shading="auto",
            cmap="RdBu_r",
        )
        axes[1, 0].set_title("TMS Evoked Response - TFR - Unfiltered, Uncropped, Uncorrected")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Frequency (Hz)")
        axes[1, 0].set_xlim([0, 2])

        # Subplot the evoked response and the TFR - after cropping and filtering
        axes[0, 1].plot(
            tms_evoked_orig_filtered.times, tms_evoked_orig_filtered.data.T * 1e15, color="red"
        )
        axes[0, 1].set_title(f"TMS Evoked Response - {tmin}-{tmax}sec - {fmin}-{fmax}Hz")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Amplitude (fT)")
        axes[0, 1].set_ylim([-2000, 2000])
        axes[0, 1].set_xlim([0, 2])
        axes[1, 1].pcolormesh(
            tfr_cropped_filtered.times,
            tfr_cropped_filtered.freqs,
            tfr_cropped_filtered.data[0],
            shading="auto",
            cmap="RdBu_r",
        )
        axes[1, 1].set_title(f"TMS Evoked Response - TFR - {tmin}-{tmax}sec - {fmin}-{fmax}Hz")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Frequency (Hz)")
        axes[1, 1].set_xlim([-0.1, 2])
        plt.tight_layout()
        # plt.show()

        # save the plot in the subject output folder
        plt.savefig(plot_folder / f"{meg_data_filename}_evoked_response_plot.png")


def poly_detrend_epochs(epochs, order=4):
    """Detrend each epoch/channel with a polynomial fit of given order."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times
    t = times - times[0]  # make time start at 0 for numerical stability

    detrended = np.empty_like(data)

    for ep in range(data.shape[0]):
        for ch in range(data.shape[1]):
            y = data[ep, ch]
            # fit polynomial of given order
            coeffs = np.polyfit(t, y, order)
            trend = np.polyval(coeffs, t)
            detrended[ep, ch] = y - trend

    # put back into Epochs object
    epochs_detrended = epochs.copy()
    epochs_detrended._data = detrended
    return epochs_detrended
