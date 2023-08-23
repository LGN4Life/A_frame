import main
import pdb
import importlib



file_name = b'C:/Henry/PythonProjects/CED_interface/data/rub_190719_con_023.smr'
print(f"file_name = {file_name}")
chan_num = 1

# params for loading spike data
spike_channel = 3
wave_mark = 1
mask_handle = 1

ced_lib = main.createWrapper()
print(f"Successfully loaded ced library: type {type(ced_lib)}")


# open file
f_handle = main.openFile(file_name, ced_lib)
if f_handle > 0:
    print(f"Successfully opened file: type {f_handle}")
else:
    print(f"Filed to open file: type {f_handle}")

# load channel title

chan_title = main.getChanTitle(chan_num, f_handle, ced_lib)
print(f"Successfully retrieved chan {chan_num} title : {chan_title}")

# load spike data
spike_times = main.loadWavemark(f_handle, spike_channel, wave_mark, mask_handle, ced_lib)


print(type(spike_times))


