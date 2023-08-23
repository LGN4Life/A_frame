import main
import pdb
import importlib

# importlib.reload(main)

file_name = b'C:/Henry/PythonProjects/CED_interface/data/rub_190719_con_023.smr'
chan_num = 1

# params for loading spike data
spike_channel = 3
wave_mark = 1
mask_handle = 1

ced_lib = main.createWrapper()
s = f"Successfully loaded ced library: type {type(ced_lib)}"
print(s)

# open file
f_handle = main.openFile(file_name, ced_lib)
if f_handle > 0:
    s = f"Successfully opened file: type {f_handle}"
else:
    s = f"Filed to open file: type {f_handle}"
print(s)


# load channel title

chan_title = main.getChanTitle(chan_num, f_handle, ced_lib)
s = f"Successfully retrieved chan {chan_num} title : {chan_title}"
print(s)

# load spike data
spike_times = main.loadWavemark(f_handle, spike_channel, wave_mark, mask_handle, ced_lib)


print(type(spike_times))
# if r == 0:
#     s = f"Successfully closed file: type {r}"
# else:
#     s = f"Filed to close file: type {r}"

# close file
r = main.closeFile(f_handle, ced_lib)

if r == 0:
    s = f"Successfully closed file: type {r}"
else:
    s = f"Filed to close file: type {r}"

print(s)
