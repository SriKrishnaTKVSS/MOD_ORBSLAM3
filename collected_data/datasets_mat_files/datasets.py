import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io

num_bins=16 # num_datasets+1
dataset_length=np.zeros((num_bins,))


for i in range(0,num_bins):
    loc_folder=f'{i+1}'
    folder_name='datasets_mat_files'+'/'+loc_folder
    current_path=os.getcwd()

    orbslam_data_path=current_path+'/'+folder_name+'/'+'orbslam_in_mocap_frame.mat'
    mocap_data_path=current_path+'/'+folder_name+'/'+'mocap_in_mocap_frame.mat'

    data_orb=scipy.io.loadmat(orbslam_data_path)
    data_mocap=scipy.io.loadmat(mocap_data_path)
    orb_timestamps=data_orb['timestamp']
    mocap_timestamps=data_mocap['timestamp']

    dataset_length[i]=orb_timestamps[-1,-1]


bins = np.arange(num_bins + 1)  # Define bin edges from 0 to the number of bins

# Plot the histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(bins[:-1], dataset_length, width=1, edgecolor="black", align="edge")

# Customize the plot
ax.set_xticks(bins)  # Ensure ticks align with bin edges
ax.set_xlabel("dataset")
ax.set_ylabel("time(s)")
ax.set_title("Histogram representing length of each dataset")

plt.show()