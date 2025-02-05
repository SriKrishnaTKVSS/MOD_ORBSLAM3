# #--------Filtering the data
# # Design the low-pass Butterworth filter
# cutoff = 5  # Cutoff frequency in Hz
# order = 5  # Order of the filter
# fs=30

# # Normalize the cutoff frequency by the Nyquist frequency (fs/2)
# nyquist = 0.5 * fs
# normal_cutoff = cutoff / nyquist


# b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

# # Apply the filter to the noisy signal (zero-phase filtering)
# filtered_signal_temp = signal.filtfilt(b, a, orbslam_data.T)
# filtered_signal=filtered_signal_temp.T

# filtered_signal = np.zeros_like(orbslam_data)
# for i in range(1,orbslam_data.shape[1]-1):  # Iterate over each column
#     filtered_signal[:, i] = signal.filtfilt(b, a, orbslam_data[:, i])

# orbslam_data=filtered_signal
# #--------------------------------------