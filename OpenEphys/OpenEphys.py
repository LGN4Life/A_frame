
import numpy as np
import re
import os

def find_recording_folders(filepath):

    # List to store folder info dictionaries
    folder_info_list = []

    # Regular expression to match experiment and recording folder names
    experiment_re = re.compile(r'^experiment(\d+)$')
    recording_re = re.compile(r'^recording(\d+)$')

    # Iterate over each folder in the base path to find valid experiment folders
    for folder_name in os.listdir(filepath):
        experiment_match = experiment_re.match(folder_name)
        if experiment_match:
            experiment_number = int(experiment_match.group(1))  # Extract and convert the number to int
            experiment_folder_path = os.path.join(filepath, folder_name)

            # Iterate over each folder in a valid experiment folder to find valid recording folders
            for subfolder_name in os.listdir(experiment_folder_path):
                recording_match = recording_re.match(subfolder_name)
                if recording_match:
                    recording_number = int(recording_match.group(1))  # Extract and convert the number to int
                    recording_folder_path = os.path.join(experiment_folder_path, subfolder_name)

                    # Create a dictionary containing the relevant info and append to list
                    folder_info = {
                        'experiment_number': experiment_number,
                        'recording_number': recording_number,
                        'file_path': recording_folder_path
                    }
                    folder_info_list.append(folder_info)
    
    # Print the collected info
    for info in folder_info_list:
        print(info)

    return folder_info_list



def get_recording_params(filename):
    data = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip().strip("'")  # Remove surrounding single quotes if present
                data[key] = value

    data['sample_rate'] = np.float64(data['sample_rate'])
    return data

def get_synced_stim_info(filename):
    s = np.load(filename)
    stim_times = s[:, 0:2]
    IV = s[:, 2]
    return stim_times, IV

