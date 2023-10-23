
import smr
import scipy.io
import h5py
import numpy as np
import OpenEphys
import pickle

rec_par_file = 'C:\\Henry\\CurrentAwakeData\\Phy\\temp\\params.py'
recording_params = OpenEphys.get_recording_params(rec_par_file)

dir_name = 'D:\\AwakeData\\Deep Array\\230413\\Record Node 101\\experiment1\\recording4\\smr_triggers\\'
file_name = 'oe_tmj_230413_ori_003.csv'
file_name = dir_name + file_name
print(file_name)
smr_triggers = smr.load_smr_triggers(file_name)

# get spike times from kilosort
spike_times = np.load('C:\\Henry\\CurrentAwakeData\\Phy\\deep_array\\230413\\recording_4\\spike_times.npy')
spike_times = spike_times / recording_params['sample_rate']
spike_clusters = np.load('C:\\Henry\\CurrentAwakeData\\Phy\\deep_array\\230413\\recording_4\\spike_clusters.npy')
# mat_file = h5py.File(spike_times_file, 'r')
# spike_info = mat_file['spike_times'][:]

# unit number
cn = 1
# get spikes for current unit
valid_trials = smr_triggers.TrialSuccess > -1
trial_index = np.where(valid_trials)[0]


for index in trial_index:
    breakpoint()
    current_spikes = spike_info[0, :] > smr_triggers.StimOn[index]#  & spike_times < smr_triggers.StimOff[trial_index]
    print(index)
    print('hello world')
    breakpoint()



