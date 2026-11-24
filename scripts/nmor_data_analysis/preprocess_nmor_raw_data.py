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
#    Interpolating the data to fill in the masked TMS pulse region before proceeding to TFR analysis (not implemented yet)
# 6. filter data - bandpass filter between 3-40Hz and notch filter at 50Hz and its harmonics to see how the TMS clicks look like in the frequency range of interest
# 7. After every step, plot the time-domain data and TFRs to see the effect of each step
# 8. Once recoverability metric script is ready - plot dipole recoverability after each step of preprocessing

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.optimize import curve_fit

# constants
EPOCH_START = -0.1  # seconds
EPOCH_END = 2.0  # seconds
TMS_MASK_START = 0  # seconds - start of TMS artefact masking
TMS_MASK_END = 0.015  # seconds - end of TMS artefact masking
TMS_EXP_FIT_START = (
    TMS_MASK_END  # seconds - start of TMS artefact exponential decay fitting
)
TMS_EXP_FIT_END = TMS_EXP_FIT_START + 0.050  # seconds - end of TMS artefact exponential decay fitting
INTERPOLATION_START = (
    TMS_MASK_START - 0.05
)  # seconds - start of interpolation to fill in the masked TMS pulse region
INTERPOLATION_END = (
    TMS_MASK_END + 0.05
)  # seconds - end of interpolation to fill in the masked TMS pulse region
DETRENDING_POLY_ORDER = 2  # order of polynomial for detrending
FMIN_TFR = 3  # Hz - minimum frequency for TFR analysis
FMAX_TFR = 40  # Hz - maximum frequency for TFR analysis

# project folder
bluebear = True
base_folder = (
    Path(r"/rds/projects/k/kowalcau-tms-opm-meg/project-sensor_comparison")
    if bluebear
    else Path(r"T:\project-sensor_comparison")
)

# subject list
subject_list = [
    "sub-all",
    "sub-dipole9Hz",
    "sub-dipole17Hz",
    "sub-dipole32Hz",
    "sub-dipole1kHz",
]


def exp_func(t, a, b, c):
    return a * np.exp(-b * t) + c


def remove_tms_exp_decay(epochs, fit_window):
    """
    Fit and remove TMS exponential decay artefact.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs after TMS pulse masking.

    fit_window : tuple
        (start_time, end_time) in seconds relative to the TMS pulse.

    Returns
    -------
    epochs_corrected : mne.Epochs
        Epochs with fitted exponential decay removed.
    """

    epochs_corrected = epochs.copy()

    data = epochs_corrected.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs_corrected.times

    fit_idx = np.where((times >= fit_window[0]) & (times <= fit_window[1]))[0]

    t_fit = times[fit_idx]
    t_fit = t_fit - t_fit[0]

    exp_fit_r2 = np.zeros(data.shape[0])
    exp_fit_rmse = np.zeros(data.shape[0])
    exp_fit_tau = np.zeros(data.shape[0])
    n_success = 0

    for ep in range(data.shape[0]):
        y = data[ep, 0, fit_idx]

        try:
            p0 = (
                y[0] - y[-1],  # amplitude
                20,  # decay rate
                y[-1],  # asymptote
            )

            popt, _ = curve_fit(
                exp_func,
                t_fit,
                y,
                p0=p0,
                maxfev=10000,
            )

            fit_curve = exp_func(t_fit, *popt)

            # calculate fit quality metric
            residuals = y - fit_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # subtract the fitted exponential decay from the original data in the fit window
            data[ep, 0, fit_idx] -= fit_curve
            exp_fit_r2[ep] = r_squared
            exp_fit_rmse[ep] = np.sqrt(ss_res / len(y))
            exp_fit_tau[ep] = 1 / popt[1]
            n_success += 1

        except RuntimeError:
            continue

    epochs_corrected._data = data
    print(f"Exponential fit successful for {n_success}/{data.shape[0]} epochs")
    return epochs_corrected, exp_fit_r2, exp_fit_rmse, exp_fit_tau


def poly_detrend_epochs(epochs, order=2):
    """Detrend each epoch/channel with a polynomial fit of given order."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times
    t = times - times[0]  # make time start at 0 for numerical stability

    detrended = np.empty_like(data)

    for ep in range(data.shape[0]):
        y = data[ep, 0, :]  # assuming single channel data
        # fit polynomial of given order
        coeffs = np.polyfit(t, y, order)
        trend = np.polyval(coeffs, t)
        detrended[ep, 0, :] = y - trend

    # put back into Epochs object
    epochs_detrended = epochs.copy()
    epochs_detrended._data = detrended
    return epochs_detrended


def interpolate_epochs(pre_mask_data, post_mask_data):
    # Placeholder for interpolation logic
    pass

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

    # look for files containing "dipoleTMS" in the filename and ending with "_raw.fif"
    meg_data_files = list(subject_meg_folder.glob("*dipoleTMS*_raw.fif"))

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
        tms_epochs_orig.save(
            evoked_response_folder / f"{meg_data_filename}_orig-epo.fif",
            overwrite=True,
        )
        tms_evoked_orig.save(
            evoked_response_folder / f"{meg_data_filename}_orig-ave.fif",
            overwrite=True,
        )

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

        # save the plot in the subject output folder
        plt.savefig(plot_folder / f"{meg_data_filename}_single_epochs_with_evoked.png")
        plt.close()

        # Crop the epochs to mask the TMS pulse for further analysis
        tms_epochs_orig_masked = tms_epochs_orig.copy().crop(
            tmin=TMS_MASK_END, tmax=EPOCH_END, include_tmax=True
        )
        tms_evoked_orig_masked = tms_epochs_orig_masked.average()
        tms_epochs_orig_masked.save(
            evoked_response_folder / f"{meg_data_filename}_TMSmasked-epo.fif",
            overwrite=True,
        )
        tms_evoked_orig_masked.save(
            evoked_response_folder / f"{meg_data_filename}_TMSmasked-ave.fif",
            overwrite=True,
        )

        # fit exponential decay function to the TMS artefact and subtract it from the data
        tms_epochs_exp_corrected, exp_fit_r2, exp_fit_rmse, exp_fit_tau = (
            remove_tms_exp_decay(
                tms_epochs_orig_masked, fit_window=(TMS_EXP_FIT_START, TMS_EXP_FIT_END)
            )
        )
        tms_evoked_exp_corrected = tms_epochs_exp_corrected.average()
        tms_epochs_exp_corrected.save(
            evoked_response_folder
            / f"{meg_data_filename}_TMSmasked_expcorrected-epo.fif",
            overwrite=True,
        )
        tms_evoked_exp_corrected.save(
            evoked_response_folder
            / f"{meg_data_filename}_TMSmasked_expcorrected-ave.fif",
            overwrite=True,
        )

        # save the exponential fit quality metrics dictionary to a npz file in the subject_output_folder/evoked_response folder
        np.savez(
            evoked_response_folder
            / f"{meg_data_filename}_TMSmasked_expcorrected_fit_metrics.npz",
            r2=exp_fit_r2,
            rmse=exp_fit_rmse,
            tau=exp_fit_tau,
        )

        # plot a histogram of the exponential fit quality metrics for the current subject and save it to the subject_output_folder/figures folder
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(exp_fit_r2, bins=20, color="blue", alpha=0.7)
        axes[0].set_title("Exponential Fit R^2")
        axes[0].set_xlabel("R^2")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(exp_fit_rmse, bins=20, color="green", alpha=0.7)
        axes[1].set_title("Exponential Fit RMSE")
        axes[1].set_xlabel("RMSE")
        axes[1].set_ylabel("Frequency")

        axes[2].hist(exp_fit_tau, bins=20, color="red", alpha=0.7)
        axes[2].set_title("Exponential Fit Tau")
        axes[2].set_xlabel("Tau")
        axes[2].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(
            plot_folder
            / f"{meg_data_filename}_TMSmasked_expcorrected_fit_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # fit polynomial of order DETRENDING_POLY_ORDER to the data and subtract it from the data
        tms_epochs_detrended = poly_detrend_epochs(
            tms_epochs_exp_corrected, order=DETRENDING_POLY_ORDER
        )
        tms_evoked_detrended = tms_epochs_detrended.average()
        tms_epochs_detrended.save(
            evoked_response_folder
            / f"{meg_data_filename}_TMSmasked_expcorrected_detrended-epo.fif",
            overwrite=True,
        )
        tms_evoked_detrended.save(
            evoked_response_folder
            / f"{meg_data_filename}_TMSmasked_expcorrected_detrended-ave.fif",
            overwrite=True,
        )

        # # interpolate the data to fill in the masked TMS pulse region before proceeding to TFR analysis
        # interpolation_times_pre_tms_mask = (tms_epochs_orig.times >= INTERPOLATION_START) & (
        #     tms_epochs_orig.times < TMS_MASK_START
        # )
        # interpolation_times_post_tms_mask = (tms_epochs_orig.times > TMS_MASK_END) & (
        #     tms_epochs_orig.times <= INTERPOLATION_END
        # )
        # epoch_orig_pre_tms_mask = tms_epochs_orig.get_data()[:, 0, interpolation_times_pre_tms_mask]
        # epoch_orig_post_tms_mask = tms_epochs_detrended.get_data()[:, 0, interpolation_times_post_tms_mask]
        # tms_epochs_interpolated = tms_epochs_orig.copy()


        # filter the evoked response between 3-40Hz to see how the TMS clicks look like in the frequency range of interest
        # plot time domain and frequency domain responses for 3 cases - original, exponential fitted, and detrended
        tms_epochs_expcorrected_filtered = tms_epochs_exp_corrected.copy().filter(
            l_freq=FMIN_TFR, h_freq=FMAX_TFR, fir_design="firwin"
        )
        tms_evoked_expcorrected_filtered = tms_epochs_expcorrected_filtered.average()
        tms_epochs_filtered = tms_epochs_detrended.copy().filter(
            l_freq=FMIN_TFR, h_freq=FMAX_TFR, fir_design="firwin"
        )
        tms_evoked_filtered = tms_epochs_filtered.average()

        # Compute required TFRs
        tfr_unfiltered = mne.time_frequency.tfr_morlet(
            tms_evoked_orig,
            freqs=np.arange(FMIN_TFR, FMAX_TFR, 1),
            return_itc=False,
            n_cycles=2,
            average=True,
        )
        tfr_expcorrected_filtered = mne.time_frequency.tfr_morlet(
            tms_evoked_expcorrected_filtered,
            freqs=np.arange(FMIN_TFR, FMAX_TFR, 1),
            return_itc=False,
            n_cycles=2,
            average=True,
        )
        tfr_cropped_filtered = mne.time_frequency.tfr_morlet(
            tms_evoked_filtered,
            freqs=np.arange(FMIN_TFR, FMAX_TFR, 1),
            return_itc=False,
            n_cycles=2,
            average=True,
        )

        # Subplot the evoked response and the TFR - for the unfiltered, uncropped, and uncorrected averaged epoch
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].plot(
            tms_evoked_orig.times, tms_evoked_orig.data.T * 1e15, color="red"
        )
        axes[0, 0].set_title("TMS Evoked Response - Unfiltered, Uncropped, Uncorrected")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude (fT)")
        axes[0, 0].set_xlim([0, 2])
        axes[0, 0].set_ylim(
            [-2000, 2000]
        )  # in fT, to ensure that the TMS pulse and the dipole are visible in the plot
        axes[1, 0].pcolormesh(
            tfr_unfiltered.times,
            tfr_unfiltered.freqs,
            tfr_unfiltered.data[0],
            shading="auto",
            cmap="RdBu_r",
        )
        axes[1, 0].set_title(
            "TMS Evoked Response - TFR - Unfiltered, Uncropped, Uncorrected"
        )
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Frequency (Hz)")
        axes[1, 0].set_xlim([0, 2])

        # Subplot the evoked response and the TFR - after cropping and filtering but after exponential decay correction
        # Subplot the evoked response and the TFR - after cropping and filtering
        axes[0, 1].plot(
            tms_evoked_expcorrected_filtered.times,
            tms_evoked_expcorrected_filtered.data.T * 1e15,
            color="red",
        )
        axes[0, 1].set_title(
            f"TMS Evoked after expcorrection - {TMS_MASK_END}-{EPOCH_END}sec - {FMIN_TFR}-{FMAX_TFR}Hz"
        )
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Amplitude (fT)")
        axes[0, 1].set_ylim([-2000, 2000])
        axes[0, 1].set_xlim([0, 2])
        axes[1, 1].pcolormesh(
            tfr_expcorrected_filtered.times,
            tfr_expcorrected_filtered.freqs,
            tfr_expcorrected_filtered.data[0],
            shading="auto",
            cmap="RdBu_r",
        )
        axes[1, 1].set_title(
            f"TMS Evoked after expcorrection - TFR - {TMS_MASK_END}-{EPOCH_END}sec - {FMIN_TFR}-{FMAX_TFR}Hz"
        )
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Frequency (Hz)")
        axes[1, 1].set_xlim([-0.1, 2])

        # Subplot the evoked response and the TFR - after cropping and filtering and detrending
        axes[0, 2].plot(
            tms_evoked_filtered.times,
            tms_evoked_filtered.data.T * 1e15,
            color="red",
        )
        axes[0, 2].set_title(
            f"TMS Evoked after detrending - {TMS_MASK_END}-{EPOCH_END}sec - {FMIN_TFR}-{FMAX_TFR}Hz"
        )
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].set_ylabel("Amplitude (fT)")
        axes[0, 2].set_ylim([-2000, 2000])
        axes[0, 2].set_xlim([0, 2])
        axes[1, 2].pcolormesh(
            tfr_cropped_filtered.times,
            tfr_cropped_filtered.freqs,
            tfr_cropped_filtered.data[0],
            shading="auto",
            cmap="RdBu_r",
        )
        axes[1, 2].set_title(
            f"TMS Evoked after detrending - TFR - {TMS_MASK_END}-{EPOCH_END}sec - {FMIN_TFR}-{FMAX_TFR}Hz"
        )
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_ylabel("Frequency (Hz)")
        axes[1, 2].set_xlim([-0.1, 2])
        plt.tight_layout()
        # plt.show()

        # save the plot in the subject output folder
        plt.savefig(plot_folder / f"{meg_data_filename}_evoked_response_plot.png")
        plt.close()
