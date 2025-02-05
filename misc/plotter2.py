import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_trajectory_and_orientations(quaternions, trajectory):
  """Plots 3D trajectory and orientations from quaternions and trajectory data.

  Args:
    quaternions: A numpy array of quaternions in the format (w, x, y, z).
    trajectory: A numpy array of 3D coordinates representing the trajectory.
  """

  # Convert quaternions to rotation matrices
  rotations = Rotation.from_quat(quaternions)
  rotation_matrices = rotations.as_matrix()

  # Create 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Plot trajectory
  ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

  # Plot rotation frames
  for rotation_matrix, position in zip(rotation_matrices, trajectory):
    # Create frame vertices
    frame_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    transformed_frame = np.dot(rotation_matrix, frame_vertices.T).T + position

    # Create Poly3DCollection for frame
    frame = Poly3DCollection([transformed_frame])
    frame.set_edgecolor('blue')
    frame.set_facecolor('none')
    ax.add_collection(frame)

  # Add labels and title
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D Trajectory and Orientations')

  plt.show()


data=np.loadtxt('CameraTrajectory.txt')
plot_trajectory_and_orientations(data[:,4:],data[:,1:4])