from ORB_SLAM3_python import System, eSensor

# Initialize the ORB-SLAM3 system with the vocabulary and settings files
slam_system = System("path_to_vocabulary", "path_to_settings", eSensor.MONOCULAR, True)

# Track an image (assuming you have a cv2 Mat image)
slam_system.TrackMonocular(image, timestamp)

# Save trajectory to TUM format
slam_system.SaveTrajectoryTUM("trajectory.txt")

# Shutdown the SLAM system
slam_system.Shutdown()
