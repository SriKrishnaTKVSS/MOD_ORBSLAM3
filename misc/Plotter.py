# This script plots the cameratrajectory from CameraTrajectory.txt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


def plot_(data):
  
    t=(data[:,0]-data[0,0])/1e9
    quaternions=data[:,4:]
    plt.figure(1)
    # Positions
    pose_x=data[:,1]
    pose_y=data[:,2]
    pose_z=data[:,3]

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(t,pose_x)
    plt.ylabel('x (m)')
    plt.grid(True)
    plt.title('Position')

    plt.subplot(3,1,2)
    plt.plot(t,pose_y)
    plt.ylabel('y (m)')
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.plot(t,pose_z)
    plt.ylabel('z (m)')
    plt.xlabel('t')
    plt.grid(True)
    """Extracts Euler angles from quaternions and plots them.

    Args:
        quaternions: A numpy array of quaternions in the format (x, y, z, w).
    """

    # Convert quaternions to Euler angles
    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler('xyz', degrees=True)

    # Extract individual Euler angles
    roll_angles = euler_angles[:, 0]
    pitch_angles = euler_angles[:, 1]
    yaw_angles = euler_angles[:, 2]

    # Plot Euler angles
    plt.figure(2)
    

    plt.subplot(3, 1, 1)
    plt.plot(t,roll_angles)
    plt.ylabel('Roll Angle (degrees)')
    plt.grid(True)
    plt.title('Orientation')

    plt.subplot(3, 1, 2)
    plt.plot(t,pitch_angles)
    plt.ylabel('Pitch Angle (degrees)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t,yaw_angles)
    plt.ylabel('Yaw Angle (degrees)')
    plt.grid(True)

    # plt.tight_layout()
    plt.show()



data=np.loadtxt('CameraTrajectory.txt')
plot_(data)
# time



# plt.show()