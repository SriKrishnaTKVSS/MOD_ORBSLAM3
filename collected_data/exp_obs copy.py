import numpy as np
import pandas as pd
import os
from Differentiators import differentiators
import matplotlib.pyplot as plt
import time
from scipy import signal
import scipy.io
from fourier_filter import fourier_fft

# -----------setting the global parameters-----------
# Adjust font sizes globally
plt.rcParams.update({
    'font.size': 28,          # Increase font size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 28,     # Axis labels font size
    'legend.fontsize': 24,    # Legend font size
    'xtick.labelsize': 24,    # X-axis tick labels font size
    'ytick.labelsize': 24,    # Y-axis tick labels font size
    'savefig.dpi': 600        # Set resolution for saved figures (DPI)
})

#----------------------------


#--------- Load the data from Mocap and orbslam---------
''' This data is treated as ground truth and velocity and acceleration data can be obtained.'''
''' Remember the data can be non uniform '''
for i in range(1,2):
    # base='191124'
    base='271124'
    loc_folder=f'{i}'
    folder_name=base+'/'+loc_folder
    current_path=os.getcwd()

    orbslam_data_path=current_path+'/'+folder_name+'/'+'orbslam_data_in_mocap.txt'
    mocap_data_path=current_path+'/'+folder_name+'/'+'mocap_data_in_mocap.txt'


    orbslam_data=np.loadtxt(orbslam_data_path)
    orbslam_data[:,1:4]=orbslam_data[:,1:4]
    mocap_data=np.loadtxt(mocap_data_path)
    mocap_data[:,3]=mocap_data[:,3]-1200 # This is because I had a mistake in logging where I have added 1200 to both mocap and orbslam

    #   Saving the whole data
    orbslam_data_dict={'timestamp':(orbslam_data[:,0]-orbslam_data[0,0])/1e9,'x':orbslam_data[:,1]/1e3,'y':orbslam_data[:,2]/1e3,'z':orbslam_data[:,3]/1e3,'roll':orbslam_data[:,4],'pitch':orbslam_data[:,5],'yaw':orbslam_data[:,6]}

    scipy.io.savemat(current_path+'/'+folder_name+'/'+'orbslam_in_mocap_frame.mat',orbslam_data_dict)

    mocap_data_dict={'timestamp':(mocap_data[:,0]-mocap_data[0,0])/1e9,'x':mocap_data[:,1]/1e3,'y':mocap_data[:,2]/1e3,'z':mocap_data[:,3]/1e3,'roll':mocap_data[:,4],'pitch':mocap_data[:,5],'yaw':mocap_data[:,6]}

    scipy.io.savemat(current_path+'/'+folder_name+'/'+'mocap_in_mocap_frame.mat',mocap_data_dict)



    # times of orbslam and mocap
    t_orb=(orbslam_data[:,0]-orbslam_data[0,0])/1e9
    t_mocap=(mocap_data[:,0]-mocap_data[0,0])/1e9


    ## Instantiating the differntiators class.
    diff=differentiators()


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

    #------Mocap data differentiation--------
    mocap_time_stamps_1,mocap_diff_1=diff.robust_differentiator(mocap_data,order=9)
    mocap_time_stamps_2,mocap_diff_2=diff.robust_differentiator(mocap_diff_1,order=9)

    #------Orbslam data differentiation-------
            
    orbslam_time_stamps_1,orbslam_diff_1=diff.robust_differentiator(orbslam_data,order=9)
    orbslam_time_stamps_2,orbslam_diff_2=diff.robust_differentiator(orbslam_diff_1,order=9)


    ##-----------plotting the derivatives----------

    # plot_index : In these processed data files, we have [0:timestamps,1:x,2:y,3:z,4:roll,5:pitch,6:yaw angles] 
    # for rotation matrices and quats..Look at the other files not these.
    # plot_index=3

    # plt.figure(1)
    # plt.subplot(2,1,1)
    # plt.plot(mocap_time_stamps_1,mocap_data[:,plot_index],label='signal')
    # # plt.plot(mocap_time_stamps_1,mocap_diff_1[:,plot_index],'r-',label='First derivative')
    # # plt.plot(mocap_time_stamps_2,mocap_diff_2[:,plot_index],'b-',label='second derivative')
    # plt.legend()
    # plt.xlabel('time(s)')
    # plt.ylabel('signal')
    # plt.title('MOCAP')
    # plt.grid(True)

    # plt.subplot(2,1,2)
    # plt.plot(orbslam_time_stamps_1,orbslam_data[:,plot_index],label='signal')
    # # plt.plot(orbslam_time_stamps_1,orbslam_diff_1[:,plot_index],'r-',label='First derivative')
    # # plt.plot(orbslam_time_stamps_2,orbslam_diff_2[:,plot_index],'b-',label='second derivative')
    # plt.legend()
    # plt.xlabel('time(s)')
    # plt.ylabel('signal')
    # plt.title('ORBSLAM')
    # plt.grid(True)
    # # plt.show()


    ##_------------ Exponential observer details ---------------
    data_size=len(mocap_time_stamps_1)

    delete_factor=0.15 # in percentage of total
    num_drop_indices_=int(delete_factor*data_size)

    #-------- data processing for the observer------------------

    dir_index=3 #{x:1,y:2,z:3}
    _dir=['x','y','z']

    timestamps_mod=orbslam_time_stamps_1[num_drop_indices_:-num_drop_indices_]-orbslam_time_stamps_1[num_drop_indices_]
    dt_timestamps_mod=np.mean(timestamps_mod[1:]-timestamps_mod[:-1])

    _mocap=mocap_data[num_drop_indices_:-num_drop_indices_,dir_index]/1000
    vel_mocap=mocap_diff_1[num_drop_indices_:-num_drop_indices_,dir_index]/1000
    acc_mocap=mocap_diff_2[num_drop_indices_:-num_drop_indices_,dir_index]/1000

    _orbslam=orbslam_data[num_drop_indices_:-num_drop_indices_,dir_index]/1000
    vel_orbslam=orbslam_diff_1[num_drop_indices_:-num_drop_indices_,dir_index]/1000
    acc_orbslam=orbslam_diff_2[num_drop_indices_:-num_drop_indices_,dir_index]/1000

    data_dict={'orb_pos':_orbslam,'orb_vel':vel_orbslam,'orb_acc':acc_orbslam,'mocap_pos':_mocap,'mocap_vel':vel_mocap,'mocap_acc':acc_mocap}
    scipy.io.savemat('data_dict.mat',data_dict)

    ##---------- Observer initial conditions



    scaled_orb_pose=np.zeros_like(timestamps_mod)#--------scale*orb_pose
    X=np.zeros_like(timestamps_mod)
    eta=np.zeros_like(timestamps_mod)
    mu_hat=np.zeros_like(timestamps_mod)
    mu_hat[0]=1
    s1=np.zeros_like(timestamps_mod)

    #----------gains---------------
    beta_1=4
    beta_2=5
    xi=np.zeros_like(timestamps_mod)


    loop_steps=len(timestamps_mod)-1

    for i in range(0,loop_steps):
        print(f"{i}\n")
        # eta[i]=-beta_1*s1[i]
        # mu_hat[i]=beta_2*acc_mocap[i]*s1[i]
        X[i]=vel_orbslam[i]-(dt_timestamps_mod*np.trapz(acc_mocap[:i]*mu_hat[:i])+acc_mocap[0]*mu_hat[0])
        
        s1[i]=X[i]+eta[i]
        print(f"String s1[i]: {s1[i]}, X[i]: {X[i]}, eta[i]: {eta[i]}, mu_hat[i]: {mu_hat[i]}\n")

        # eta[i+1]=dt_timestamps_mod*(-beta_1*s1[i])+eta[i]
        # mu_hat[i+1]=dt_timestamps_mod*(-beta_2*acc_mocap[i]*s1[i])+mu_hat[i]

        eta[i+1]=dt_timestamps_mod*(np.trapz(-beta_1*s1[:i]))+eta[0]
        mu_hat[i+1]=dt_timestamps_mod*(np.trapz(beta_2*acc_mocap[:i]*s1[:i]))+mu_hat[0]
        xi[i]=(acc_orbslam[i]/acc_mocap[i])-mu_hat[i]
        
    _scaled_orb_pose=_orbslam/mu_hat


    fig2=plt.figure(2)
    plt.suptitle(f"Comparison of Mocap and orbslam signals for $\\beta_1$: {beta_1}, $\\beta_2:$ {beta_2}")
    plt.subplot(3,1,1)
    plt.plot(timestamps_mod,_mocap,'r-',label='mocap_signal')
    plt.plot(timestamps_mod,_orbslam,'b-',label='original_orbslam_signal')
    plt.plot(timestamps_mod,_scaled_orb_pose,'g-',label='scaled_orbslam_signal')

    plt.legend(bbox_to_anchor=(0.5, 2.68),loc='upper center',ncols=3,mode="expand",borderpad=0,facecolor='none',borderaxespad=10,columnspacing=1,edgecolor='none')
    # plt.xlabel('time(s)')
    plt.ylabel(f'{_dir[dir_index-1]} (m)')

    plt.xlim([0,timestamps_mod[loop_steps-1]])
    plt.ylim([np.min(_mocap)-0.25,np.max(_mocap)+0.25])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()

    plt.subplot(3,1,2)
    plt.plot(timestamps_mod,vel_mocap,'r-',label='mocap_velocity')
    plt.plot(timestamps_mod,vel_orbslam,'b-',label='orbslam_velocity')

    # plt.legend()
    # plt.xlabel('time(s)')
    plt.ylabel(f"${{\\dot{{{_dir[dir_index-1]}}}}}$  (m/s)")

    plt.xlim([0,timestamps_mod[loop_steps-1]])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()



    plt.subplot(3,1,3)
    plt.plot(timestamps_mod,acc_mocap,'r-',label='mocap_acceleration')
    plt.plot(timestamps_mod,acc_orbslam,'b-',label='orbslam_acceleration')
    # plt.legend()
    plt.xlabel('time(s)')
    plt.ylabel(f"${{\\ddot{{{_dir[dir_index-1]}}}}}$  (m/s2)")

    plt.xlim([0,timestamps_mod[loop_steps-1]])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()
    plt.get_current_fig_manager().window.showMaximized()

    fig3=plt.figure(3)
    plt.subplot(3,2,1)
    plt.plot(timestamps_mod,mu_hat,'b-',label="$\\hat{\\mu}$")
    # plt.xlabel('t(s)')
    plt.ylabel('scale ($\\hat{\\mu}$)')
    plt.legend()
    plt.xlim([0,timestamps_mod[loop_steps-1]])
    # plt.ylim([-0.005,0.05])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()

    plt.subplot(3,2,2)
    plt.plot(timestamps_mod,s1,'b-',label='s')
    # plt.xlabel('t(s)')
    plt.ylabel('s')
    plt.legend()
    plt.xlim([0,timestamps_mod[loop_steps-1]])
    # plt.ylim([-1000,1000])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()


    plt.subplot(3,2,3)
    plt.plot(timestamps_mod,eta,'b-',label="$\\eta$")
    # plt.xlabel('t(s)')
    plt.ylabel("$\\eta$")
    plt.legend()
    plt.xlim([0,timestamps_mod[loop_steps-1]])
    # plt.ylim([-0.001,0.001])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()

    plt.subplot(3,2,4)
    plt.plot(timestamps_mod,X,'b-',label='X')
    plt.xlabel('t(s)')
    plt.ylabel('X')
    plt.legend()
    plt.xlim([0,timestamps_mod[loop_steps-1]])
    # plt.ylim([-1000,1000])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()

    plt.subplot(3,2,5)
    plt.plot(timestamps_mod,xi,'b-',label="$\\xi$")
    plt.xlabel('t(s)')
    plt.ylabel('$\\xi$')
    plt.legend()
    plt.xlim([0,timestamps_mod[loop_steps-1]])
    plt.ylim([-120,120])
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
    plt.minorticks_on()
    plt.get_current_fig_manager().window.showMaximized()

    plt.show()

    #--------- Saving the maximized plots ----------
    results_folder=os.path.join(current_path+'/'+folder_name+'/','results')

    # Create the 'results' folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    fig2.savefig(results_folder+f'/comps_{_dir[dir_index-1]}_{base}_{loc_folder}.pdf')
    fig3.savefig(results_folder+f'/gains_{_dir[dir_index-1]}_{base}_{loc_folder}.pdf')



    # time.sleep(3)
    # Mean of orbslam/mocap value

    mean=np.mean((_orbslam[:]/_mocap[:]))
    print(f"Mean of orbslam/mocap : {mean}")