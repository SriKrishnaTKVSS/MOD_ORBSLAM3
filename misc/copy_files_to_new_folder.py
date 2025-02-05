import os
import shutil
import sys

def copy_files_to_new_folder():
    # List of file names
    file_names = ["CameraTrajectory.txt", "figure.pdf", "Mocap_data.txt","KeyFrameTrajectory.txt","mocap_data_in_mocap.txt","Mocap_data_processed.txt","orbslam_data_in_mocap.txt"]  # Replace with your file names

    # Prompt for the folder name
    folder_name = input("Enter the name of the new folder: ")

    # Base path where the folder will be created
    base_path = "/home/srikrishna/Desktop/Projects/scale_est/Krishna_pybind_orsbslam3/src/ORB_SLAM3/collected_data/271124/"  # Replace with your desired path

    # Full path of the new folder
    new_folder_path = os.path.join(base_path, folder_name)

    # Check if the folder already exists
    if os.path.exists(new_folder_path):
        print(f"Error: A directory with the name '{folder_name}' already exists at {base_path}.")
        sys.exit(1)  # Exit the script

    # Create the folder
    os.makedirs(new_folder_path)
    print(f"Folder created: {new_folder_path}")

    # Copy each file to the new folder
    for file_name in file_names:
        source_path = os.path.join(os.getcwd(), file_name)  # Current directory
        destination_path = os.path.join(new_folder_path, file_name)
        
        # Check if the file exists before copying
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            print(f"Copied: {file_name} to {new_folder_path}")
        else:
            print(f"File not found: {file_name}")

    print(f"All files copied to {new_folder_path}")

# Run the function
copy_files_to_new_folder()
