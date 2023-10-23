
import numpy as np
import re
import os
import json


def fix_key(key):
    illegal_indices = [m.start() for m in re.finditer(r"[\s(),%#-/:?]", key)]
    illegal_indices = np.array(illegal_indices)
    for index in illegal_indices:
        if index < len(key)-1:
            key = key[:index+1] + key[index+1].upper() + key[index + 2:]
    legal_characters = re.findall(r"[^\s(),%#-/:?]", key)
    key = "".join(legal_characters)
    return key


def load_par_file(par_file, save_path):
    par = {}
    current_heading = 'Non Specific'
    if os.path.isfile(par_file):
        with open(par_file, 'r') as file:
            raw_parameters = file.readlines()
        for line_index in range(len(raw_parameters)):
            new_heading = re.match("-{5,}", raw_parameters[line_index])
            colon_indices = [m.start() for m in re.finditer(":", raw_parameters[line_index])]
            if new_heading is not None:
                current_heading = raw_parameters[line_index - 1]
                current_heading = current_heading.replace(" ", "_")
                current_heading = re.sub(r"\n", "", current_heading)
                par[current_heading] = {}

            if current_heading != "Non Specific":
                if current_heading == "Targets":
                    print("add code for star stim")
                elif current_heading == "Stimulus Locations":
                    print("add code for gregg task")
                elif current_heading == 'TrialByTrialOrientations':
                    print("add code for gregg task")
                else:
                    if len(colon_indices) == 1:
                        key, value = raw_parameters[line_index].split(':')
                        # remove illegal characters from key
                        key = fix_key(key)
                        legal_characters = re.findall(r"[^\s]", value)
                        value = "".join(legal_characters)
                        # print(f"key = {key}, value = {value}")
                        par[current_heading][key] = value
                    elif len(colon_indices) == 2:
                        current_line = raw_parameters[line_index]
                        current_line = current_line.replace(":", " ", 1)
                        key, value = current_line.split(':')
                        # remove illegal characters from key
                        key = fix_key(key)
                        legal_characters = re.findall(r"[^\s]", value)
                        value = "".join(legal_characters)
                        par[current_heading][key] = value

    # Save the dictionary to a JSON file
    with open(save_path, "w") as file:
        json.dump(par, file)

    return par
