
from ctypes import *
import os
import numpy as np
# import

print(os.getcwd())
# lib = WinDLL("C:\\Henry\\PythonProjects\\CED\\CEDS64ML\\x64\\ceds64int.dll")
ced_lib = LibraryLoader(ctypes.WinDLL).LoadLibrary("C:\\Henry\\PythonProjects\\CED\\CEDS64ML\\x86\\ceds64int.dll")

print(type(ced_lib))
# s = "Hello, World"
# c_s = c_char_p(s)
print(ced_lib)
print(ced_lib.S64Open.argtypes)  # this function opens .smr(x) files
ced_lib.S64Open.argtypes = [POINTER(c_char)]
ced_lib.S64Close.argtypes = [c_int]
print(ced_lib.S64Open.argtypes)  # this function opens .smr(x) files
file_name = b"rub_190719_con_023.smr"
print(c_char_p(file_name))
fhand = ced_lib.S64Open(file_name)
ced_lib.S64MaxChans.argtypes = [c_int]
new_title = b"place holder"

print('process start')
ced_lib.S64SetChanTitle.argtypes = [c_int, c_int,POINTER(c_char)]
chan_num = 1

chan_buffer = ced_lib.S64GetChanTitle(fhand, chan_num, new_title, -1)
arr = bytes(chan_buffer)
print('arr  = ' + str(arr))
print('type arr is ' + (str(type(arr))))
print('type new_title is ' + (str(type(new_title))))
get_title_result = ced_lib.S64GetChanTitle(fhand, chan_num, arr, chan_buffer)
print('process finished')
print("result of SetChan title  = " + str(arr))
print("get_title_result = " + str(get_title_result))

current_title = arr.decode("utf-8")
print('new string  = ' + current_title)


# load spike times
# set mask mode
wave_mark = 1
mask_handle = 1
ced_lib.S64GetMaskMode.argtypes = [c_int, c_int]

iOk = ced_lib.S64GetMaskMode(mask_handle,0)

print('mask mode attempt   = ' + str(iOk))
mask = np.ones((256, 4)).astype('int8')
# mask = np.ones((256, 4))
print('mask type is ' + str(type(mask)))
mask[0:wave_mark, 0] = 0
mask[wave_mark+1:, 0] = 0
print('mask type is ' + str(type(mask)))
mask = mask.ctypes.data_as(POINTER(c_int))
ced_lib.S64SetMaskCodes.argtypes = [c_int, POINTER(c_int)]
ced_lib.S64SetMaskCodes(mask_handle, mask)

max_events = 50

spike_data = -1*np.ones((max_events,), dtype=np.longlong)
spike_data = spike_data.ctypes.data_as(POINTER(c_longlong))
ced_lib.S64ReadEvents.argtypes = [c_int, c_int, POINTER(c_longlong), c_int, c_longlong, c_longlong, c_int]
spike_channel = 3
# start_tick = np.array([0], dtype=np.longlong)
start_tick = c_longlong(0)
end_tick = c_longlong(-1)
# start_tick = start_tick.ctypes.data_as(POINTER(c_longlong))
# end_tick = np.array([300000], dtype=np.longlong)
# end_tick = end_tick.ctypes.data_as(POINTER(c_longlong))
print('load spike times')

ced_lib.S64ReadEvents(fhand, spike_channel, spike_data, max_events, start_tick, end_tick, mask_handle)
# print(str(spike_data.data))
print('spike times loaded')

print('spike data  =  ' + str(type(spike_data)))

result = ced_lib.S64Close(fhand)



