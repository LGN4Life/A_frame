import os

def folder_check(folder_path):

    # check for folder
    if not os.path.exists(folder_path):
        # if folder does not exist, create it
        os.makedirs(folder_path)
    
    return

    