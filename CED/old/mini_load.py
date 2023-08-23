import sys
import CED.load as load
import CED.ced as ced
import ctypes as ct
import struct
import numpy as np


file_name = 'C:\\Henry\\PythonProjects\\CED_interface\\data\\cnk_230628_area_000.smrx'
# file_name = 'C:\\Henry\\PythonProjects\\CED_interface\\data\\tmj_230731_tun_000.smrx' 
# params for loading spike data
spike_info = {}
spike_info['channel'] = 3
spike_info['wave_mark']  = 1

# load library
ced_lib = ced.createWrapper()
print(f"Successfully loaded ced library: type {type(ced_lib)}")


# make sure all previous files were closed
ced.closeFile(ced_lib=ced_lib)

# open file
fhand = ced.open_file(file_name, ced_lib)
if fhand > 0:
    print(f"file opened successfully: {file_name}")
else:
    print(f"file could not be opened: {file_name}")


# set mask for spikes
mask_handle = 1
mask_mode = 0
max_events = 1000

# set mask mode
mask_flag = ced_lib.S64SetMaskMode(mask_handle, mask_mode)

print(f"mask mode attempt   = {mask_flag}")

mask = np.zeros((256, 4)).astype('int8')
mask[spike_info['wave_mark'] , 0] = 1
# mask[0:spike_info['wave_mark'] , 0] = 0
# mask[spike_info['wave_mark']  + 1:, 0] = 0
mask = mask.reshape(-1, 1)
mask = mask.ctypes.data_as(ct.POINTER(ct.c_int))
ced_lib.S64SetMaskCodes(mask_handle, mask)

# get spike times
 # as far as I can tell mask_handle is always 1 and mask mode is always 0
    

# load spikes

start_tick = 0
spike_times = []
exit_flag = False
while not exit_flag:
    print(f"exit flag = {exit_flag}")
    outevpointer = (ct.c_longlong * max_events)()
    iRead = ced_lib.S64ReadEvents(fhand, spike_info['channel'], outevpointer, max_events, start_tick, -1, mask_handle)
    current_spike_times = list(outevpointer)[:iRead]
    spike_times.extend(current_spike_times)
    if iRead < max_events:
        exit_flag = True
    else:
        start_tick = current_spike_times[-1] + 1
   


# convert spike times
# get base time
time_base = ced_lib.S64GetTimeBase(fhand)
spike_times = np.array(spike_times)
spike_times = spike_times * time_base



# close file
# close_flg = ced.closeFile(fhand, ced_lib=ced_lib)
close_flag = ced_lib.S64Close(fhand)
print(f"close flag = {close_flag}")
print(spike_times[0:30])
close_flag = ced_lib.S64CloseAll()
print(f"close flag = {close_flag}")



