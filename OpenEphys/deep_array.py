
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt

def remap_to_linear_depth(default_map_directory):
    """
    Generate linear depth order for LFP channels. By deafult the channels of deep_aray are not in order of depth. Kilosort puts them in order to sort spikes. We
    should save the LFP channels using linear depth file_names (e.g., the deepest LFP channel = channel_128, the shallowest = channel_1)
    most of this code probably serves no purpose, but leave it for now
    input: 
        default_map_directory = location of default channel map used by kilosort
        save_directory: directory to save new channel map for matlab (not needed)
    output:
        open_ephys_default_index: default index values for each channel. (index 0 == the first row of continuous.dat). 
            the row number indicates the physical position on the probe (linear depth). For example, open_ephys_default_index[1] = 127
            This means row 127 of continuous.dat is channel_1 in linear depth (channels start at channel_0)
    """

    # import channel map that Scottie created in matlab
    # file_name =  "D:\\AwakeData\\Python\\230811\\tmj_230811\\channel_map\\deep_array_channel_map.mat"
    mat_data = loadmat(default_map_directory)
    
    # list of electrode index (1:N) in order of depth (e.g., 1,128,2,etc. This means, that in order of depth, channel 128 is second from the 
    # top)
    open_ephys_default_index =  mat_data['channel_map'][0,0][0].flatten()-1


   


    return open_ephys_default_index