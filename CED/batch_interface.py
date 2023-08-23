import sys
import CED.load as load
import CED.ced as ced
import ctypes as ct
import struct
# from matplotlib import pyplot as plt

temp = struct.calcsize("P") * 8
print(f"size =  {temp}")

file_name = 'C:\\Henry\\PythonProjects\\CED_interface\\data\\cnk_221012_tun_000'
# params for loading spike data
spike_info = {}
spike_info['channel'] = 5
spike_info['wave_mark']  = 1
# params for lfp data
lfp_channel = 3

spike_times, fhand, ced_lib, lfp_data, lfp_time, fixation_times, fixation_levels = load.load_file(file_name , spike_info, lfp_channel)
print(f"spike_times = {spike_times[0:10]}, n = {spike_times.shape[0]}")

# plt.plot(lfp_time, lfp_data)
# plt.show()



