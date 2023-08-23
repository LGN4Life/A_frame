import ctypes as ct
import numpy as np
import pdb


def createWrapper():
    import numpy as np

    # load CED lib
    ced_lib = ct.LibraryLoader(ct.WinDLL).LoadLibrary(
        "C:\\Henry\\PythonProjects\\CED_interface\\CEDS64ML\\x64\\ceds64int.dll")
    return ced_lib


def openFile(file_name, ced_lib):
    # openfile: opens .smr(x) files
    # inputs:
    # file_name: name of file (with directory)
    # ced_lib: created with createWrapper()
    ced_lib.S64Open.argtypes = [ct.POINTER(ct.c_char)]
    fhand = ced_lib.S64Open(file_name)
    return fhand


def closeFile(f_handle, ced_lib):
    # closeFile: closes .smr(x) files
    # inputs:
    # f_handle: handle to open file
    ced_lib.S64Close.argtypes = [ct.c_int]
    close_flag = ced_lib.S64Close(f_handle)
    return close_flag


def loadWavemark(f_hand, spike_channel, wave_mark, mask_handle, ced_lib):
    # loads spike times

    max_events = 50

    # set mask mode
    ced_lib.S64GetMaskMode.argtypes = [ct.c_int, ct.c_int]
    iok = ced_lib.S64GetMaskMode(mask_handle, 0)

    print(f"mask mode attempt   = {iok}")

    mask = np.ones((256, 4)).astype('int8')
    mask[0:wave_mark, 0] = 0
    mask[wave_mark + 1:, 0] = 0
    mask = mask.ctypes.data_as(ct.POINTER(ct.c_int))
    ced_lib.S64SetMaskCodes.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
    ced_lib.S64SetMaskCodes(mask_handle, mask)

    # load spikes
    spike_data = -1 * np.ones((max_events,), dtype=np.longlong)
    spike_data = spike_data.ctypes.data_as(ct.POINTER(ct.c_longlong))
    ced_lib.S64ReadEvents.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_longlong), ct.c_int, ct.c_longlong,
                                      ct.c_longlong, ct.c_int]

    start_tick = ct.c_longlong(0)
    end_tick = ct.c_longlong(-1)

    # MATINT_API int S64ReadEvents(const int nFid, const int nChan,  long long *pData, int nMax,
    #     const long long tFrom, const long long tTo, const int nMask);



    error_flag = ced_lib.S64ReadEvents(f_hand, spike_channel, spike_data, max_events,
                                       start_tick, end_tick, mask_handle)
    pdb.set_trace()
    return spike_data


def load_continuous():
    print('in progress')


def getChanTitle(chan_num, fhand, ced_lib):
    # changes the title of a channel in open .smr(x) file
    # inputs:
    #   new_title = string
    #   chan_num = channel number to change title
    #   f_hand =  handle to open file
    #   ced_lib: created with createWrapper
    # Returns result of attempt to set title

    # dummy string before the size of the title is known
    dummy_string = b' '

    ced_lib.S64MaxChans.argtypes = [ct.c_int]
    ced_lib.S64SetChanTitle.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_char)]

    iSize = ced_lib.S64GetChanTitle(fhand, chan_num, dummy_string, -1)
    chan_title = ' ' * (iSize + 1)
    chan_title = str.encode(chan_title)
    r2 = ced_lib.S64GetChanTitle(fhand, chan_num, chan_title, 0)

    return chan_title


def setChanTitle(new_title, chan_num, fhand, ced_lib):
    # changes the title of a channel in open .smr(x) file
    # inputs:
    #   new_title = string
    #   chan_num = channel number to change title
    #   f_hand =  handle to open file
    #   ced_lib: created with createWrapper
    # Returns result of attempt to set title

    ced_lib.S64MaxChans.argtypes = [ct.c_int]
    ced_lib.S64SetChanTitle.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_char)]
    r = ced_lib.S64GetChanTitle(fhand, chan_num, new_title, -1)

    arr = bytes(chan_buffer)
    print('type new_title is ' + (str(type(new_title))))
    return r
