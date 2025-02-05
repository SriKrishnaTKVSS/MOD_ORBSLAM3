## ----------- Importing the libraries -------------------
import numpy as np
import pandas as pd
import os
from Differentiators import differentiators
import matplotlib.pyplot as plt
import time
from scipy import signal
import scipy.io
from fourier_filter import fourier_fft

#---------------------------------------------------------
# 1D moving average filter
def moving_average_fil(array,window_size):
    filtered_signal=np.zeros_like(array)
    N=len(filtered_signal)

    for i in range(0,N):
        if i<window_size:
            filtered_signal[i]=array[i]
        else:
            filtered_signal[i]=np.sum(array[i-window_size:i])/window_size
    return filtered_signal




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

# ----------------------------


#--------- Load the data from Mocap and orbslam---------
''' This Mocap position data is treated as ground truth. Further, velocity and acceleration data can be obtained.'''
''' Remember the data can be non uniform '''

## ------- Setting up the path for reading the data -----

# base='191124' # Datasets folder with 20 seconds duration
base='271124' # Datasets folder with 140 seconds duration
loc_folder='1' # Sub folder
folder_name=base+'/'+loc_folder
current_path=os.getcwd()
orbslam_data_path=current_path+'/'+folder_name+'/'+'orbslam_data_in_mocap.txt'
mocap_data_path=current_path+'/'+folder_name+'/'+'mocap_data_in_mocap.txt'

#------------------Rough-----------------
# Checking the untransformed orbslam vel/pos
orb_untransform_path=current_path+'/'+ folder_name+'/' + 'CameraTrajectory.txt'
orbs_untr_data=np.loadtxt(orb_untransform_path)



orbslam_data=np.loadtxt(orbslam_data_path)
orbslam_data[:,1:4]=orbslam_data[:,1:4]
mocap_data=np.loadtxt(mocap_data_path)
mocap_data[:,3]=mocap_data[:,3]-1200 # This 1200 is because I had a mistake in logging where I have added 1200 to both mocap (we do not need to add to Mocap data) and orbslam
#----------------------------------------------------

 



# --------- For Data export as .mat files------------

# #   Saving the whole data
# orbslam_data_dict={'timestamp':orbslam_data[:,0]/1e9,'x':orbslam_data[:,1]/1e3,'y':orbslam_data[:,2]/1e3,'z':orbslam_data[:,3]/1e3,'roll':orbslam_data[:,4],'pitch':orbslam_data[:,5],'yaw':orbslam_data[:,6]}

# scipy.io.savemat(current_path+'/'+folder_name+'/'+'orbslam_in_mocap_frame.mat',orbslam_data_dict)

# mocap_data_dict={'timestamp':mocap_data[:,0]/1e9,'x':mocap_data[:,1]/1e3,'y':mocap_data[:,2]/1e3,'z':mocap_data[:,3]/1e3,'roll':mocap_data[:,4],'pitch':mocap_data[:,5],'yaw':mocap_data[:,6]}

# scipy.io.savemat(current_path+'/'+folder_name+'/'+'mocap_in_mocap_frame.mat',mocap_data_dict)
#-------------------------------------------------------


# times of orbslam and mocap
t_orb=(orbslam_data[:,0]-orbslam_data[0,0])/1e9
t_mocap=(mocap_data[:,0]-mocap_data[0,0])/1e9


## Instantiating the differntiators class from the other class.
diff=differentiators()

# ------Mocap data differentiation--------
mocap_time_stamps_1,mocap_diff_1=diff.robust_differentiator(mocap_data,order=9)
mocap_time_stamps_2,mocap_diff_2=diff.robust_differentiator(mocap_diff_1,order=9)

# ------Orbslam data differentiation-------
        
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

# Preprocessing the data by removing first and last few derivatives as well as corresponding pose data.

data_size=len(mocap_time_stamps_1)   

delete_factor=0.01 # in percentage of total
num_drop_indices_=int(delete_factor*data_size)
# num_drop_indices_=1

#-------- data processing for the observer------------------

dir_index=3 #{x:1,y:2,z:3}
_dir=['x','y','z']

timestamps_mod=orbslam_time_stamps_1[num_drop_indices_:-num_drop_indices_]-orbslam_time_stamps_1[num_drop_indices_]
dt_timestamps_mod=np.mean(timestamps_mod[1:]-timestamps_mod[:-1])

_mocap=mocap_data[num_drop_indices_:-num_drop_indices_,dir_index]/1000
vel_mocap_raw=mocap_diff_1[num_drop_indices_:-num_drop_indices_,dir_index]/1000
acc_mocap_raw=mocap_diff_2[num_drop_indices_:-num_drop_indices_,dir_index]/1000


_orbslam=orbslam_data[num_drop_indices_:-num_drop_indices_,dir_index]/1000
vel_orbslam_raw=orbslam_diff_1[num_drop_indices_:-num_drop_indices_,dir_index]/1000
acc_orbslam_raw=orbslam_diff_2[num_drop_indices_:-num_drop_indices_,dir_index]/1000

#------------------- filtered signals of derivatives-----------
window_size=30 # if given window size 1, there won't be any filtering effect.
vel_mocap=moving_average_fil(vel_mocap_raw,window_size=5)
acc_mocap=moving_average_fil(acc_mocap_raw,window_size=5)

vel_orbslam=moving_average_fil(vel_orbslam_raw,window_size=5)
acc_orbslam=moving_average_fil(acc_orbslam_raw, window_size=5)

plt.figure()
plt.subplot(2,2,1)
plt.plot(timestamps_mod,vel_mocap_raw,label='Raw velocity mocap')
plt.plot(timestamps_mod,vel_mocap,label='Filtered velocity mocap')

plt.subplot(2,2,2)
plt.plot(timestamps_mod,acc_mocap_raw,label='Raw acc mocap')
plt.plot(timestamps_mod,acc_mocap,label='Filtered acc mocap')

plt.subplot(2,2,3)
plt.plot(timestamps_mod,vel_orbslam_raw,label='Raw velocity orb')
plt.plot(timestamps_mod,vel_orbslam,label='Filtered velocity orb')

plt.subplot(2,2,4)
plt.plot(timestamps_mod,acc_orbslam_raw,label='Raw acc orb')
plt.plot(timestamps_mod,acc_orbslam,label='Filtered acc orb')

plt.show()
#--------------------------

data_dict={'orb_pos':_orbslam,'orb_vel':vel_orbslam,'orb_acc':acc_orbslam,'mocap_pos':_mocap,'mocap_vel':vel_mocap,'mocap_acc':acc_mocap}
scipy.io.savemat('data_dict.mat',data_dict)

## ---------- Observer initial conditions -----------------

scaled_orb_pose=np.zeros_like(timestamps_mod)#--------scale*orb_pose
X=np.zeros_like(timestamps_mod)
eta=np.zeros_like(timestamps_mod)
mu_hat=np.zeros_like(timestamps_mod)
mu_hat[0]=1
s1=np.zeros_like(timestamps_mod)

# ---------------------------------------------------------


#----------gains---------------

beta_1=1.0
beta_2=5
xi=np.zeros_like(timestamps_mod)


loop_steps=len(timestamps_mod)-1

for i in range(0,loop_steps):
    print(f"{i}\n")

    X[i]=vel_orbslam[i]-(dt_timestamps_mod*np.trapz(acc_mocap[:i]*mu_hat[:i])+acc_mocap[0]*mu_hat[0])
    
    s1[i]=X[i]+eta[i]
    print(f"String s1[i]: {s1[i]}, X[i]: {X[i]}, eta[i]: {eta[i]}, mu_hat[i]: {mu_hat[i]}\n")

    eta[i+1]=dt_timestamps_mod*(np.trapz(-beta_1*s1[:i]))+eta[0]
    mu_hat[i+1]=dt_timestamps_mod*(np.trapz(beta_2*acc_mocap[:i]*s1[:i]))+mu_hat[0]
    xi[i]=(acc_orbslam[i]/acc_mocap[i])-mu_hat[i]
    
_scaled_orb_pose=_orbslam/mu_hat


# ---------------- Plotting ------------------------

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

plt.ylabel(f"${{\\dot{{{_dir[dir_index-1]}}}}}$  (m/s)")
plt.xlim([0,timestamps_mod[loop_steps-1]])
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
plt.minorticks_on()



plt.subplot(3,1,3)
plt.plot(timestamps_mod,acc_mocap,'r-',label='mocap_acceleration')
plt.plot(timestamps_mod,acc_orbslam,'b-',label='orbslam_acceleration')
plt.xlabel('time(s)')
plt.ylabel(f"${{\\ddot{{{_dir[dir_index-1]}}}}}$  (m/s2)")
plt.xlim([0,timestamps_mod[loop_steps-1]])
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
plt.minorticks_on()
plt.get_current_fig_manager().window.showMaximized()

fig3=plt.figure(3)
plt.suptitle(f"Comparison of Mocap and orbslam signals for $\\beta_1$: {beta_1}, $\\beta_2:$ {beta_2}")
plt.subplot(3,2,1)
plt.plot(timestamps_mod,mu_hat,'b-',label="$\\hat{\\mu}$")
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




#----------Computation of vz/z-----------------

of_mocap=vel_mocap/_mocap
of_orbslam=vel_orbslam/_orbslam

plt.figure(4)
plt.subplot(1,2,1)
plt.plot(timestamps_mod,of_mocap,'r-',label='vz/z mocap')
plt.plot(timestamps_mod,of_orbslam,'b-',label='vz/z orbslam')

plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
plt.xlabel('t(s)')
plt.ylabel('OF (1/s)')

plt.xlim([0,timestamps_mod[loop_steps-1]])
plt.legend()

plt.subplot(1,2,2)
plt.plot(timestamps_mod,of_mocap/of_orbslam, label='OF mocap/orbslam')
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  
plt.xlabel('t(s)')
plt.ylabel('OF ratio')
plt.ylim([-100,100])

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



