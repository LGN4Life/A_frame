import ctypes as ct
import numpy as np
import os

class channel_info:
    def __init__(self, **kwargs):
        self.number = kwargs.get('number', -99)
        self.type = kwargs.get('type', -99)
        self.label = kwargs.get('label', ' ')

def createWrapper():
    # load CED lib
    # ced_lib = ct.LibraryLoader(ct.WinDLL).LoadLibrary(
    #     "C:\\Henry\\PythonProjects\\CED_interface\\CEDS64ML\\x64\\ceds64int.dll")
    # ced_lib = ct.LibraryLoader(ct.WinDLL).LoadLibrary(
    #       "C:\\CEDMATLAB\\CEDS64ML\\x86\\ceds64int.dll")
    
    ced_lib = ct.LibraryLoader(ct.WinDLL).LoadLibrary(
          "C:\\CEDMATLAB\\CEDS64ML\\x64\\ceds64int.dll")
    
    # open file
    ced_lib.S64Open.argtypes = [ct.POINTER(ct.c_char)]

    # close file
    ced_lib.S64Close.restype = ct.c_int
    ced_lib.S64Close.argtypes = [ct.c_int]
    ced_lib.S64CloseAll.restypes = [ct.c_int]

    # get base time
    ced_lib.S64GetTimeBase.restype = ct.c_double
    ced_lib.S64GetTimeBase.argtypes = [ct.c_int]

    # get max time
    ced_lib.S64ChanMaxTime.restype = ct.c_longlong
    ced_lib.S64ChanMaxTime.argtypes = [ct.c_int, ct.c_int]

    # get chan sampling rate
    ced_lib.S64ChanDivide.restype = ct.c_int
    ced_lib.S64ChanDivide.argtypes = [ct.c_int, ct.c_int]


    # max_channel
    ced_lib.S64MaxChans.restype = ct.c_int
    ced_lib.S64MaxChans.argtypes = [ct.c_int]

    # get channel title
    ced_lib.S64GetChanTitle.restype = ct.c_int
    ced_lib.S64GetChanTitle.argtypes = [ct.c_int, ct.c_int, ct.c_char_p, ct.c_int]

    # channel type
    ced_lib.S64ChanType.restype = ct.c_int
    ced_lib.S64ChanType.argtypes = [ct.c_int, ct.c_int]

    # mask mode
    ced_lib.S64GetMaskMode.restype = ct.c_int
    ced_lib.S64GetMaskMode.argtypes = [ct.c_int]

    ced_lib.S64SetMaskMode.restype = ct.c_int
    ced_lib.S64SetMaskMode.argtypes = [ct.c_int, ct.c_int]

    ced_lib.S64SetMaskCodes.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]

    # Event Data
    ced_lib.S64ReadEvents.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_longlong), ct.c_int, ct.c_longlong,
                                      ct.c_longlong, ct.c_int]
    
    # spike data
    ced_lib.S64TicksToSecs.restype = ct.c_double
    ced_lib.S64TicksToSecs.argtypes = [ct.c_int, ct.c_longlong]
    # CED.ced.LP_c_longlong
   

    # continuous data
    ced_lib.S64ReadWaveF.restype = ct.c_int
    ced_lib.S64ReadWaveF.argtypes = [
    ct.c_int,       # nFid
    ct.c_int,       # nChan
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # pData (float*)
    ct.c_int,       # nMax
    ct.c_longlong,  # tFrom
    ct.c_longlong,  # tUpto
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=2),  # tFirst (long long*)
    ct.c_int        # nMask
    ]

    # marker data
    ced_lib.S64ReadMarkers.restype = ct.c_int
    ced_lib.S64ReadMarkers.argtypes = argtypes = [
        ct.c_int,                             # nFid
        ct.c_int,                             # nChan
        ct.POINTER(S64Marker),                # pData (S64Marker*)
        ct.c_int,                             # nMax
        ct.c_longlong,                        # tFrom
        ct.c_longlong,                        # tUpto
        ct.c_int                             # nMask
    ]
    # MATINT_API int S64GetExtMarkInfo(const int nFid, const int nChan, int* nRows, int* nCols);
    # marker info
    ced_lib.S64GetExtMarkInfo.restype = ct.c_int
    ced_lib.S64GetExtMarkInfo.argtypes = [
        ct.c_int, 
        ct.c_int, 
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # nRows (int*)
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # nCols (int*)
        # ct.POINTER(ct.c_int), 
        # ct.POINTER(ct.c_int)
    ]

    # text marker data
    ced_lib.S64Read1TextMark.restype = ct.c_int
    ced_lib.S64Read1TextMark.argtypes = [
        ct.c_int,                             # nFid
        ct.c_int,                             # nChan
        ct.POINTER(S64Marker),                # pData (S64Marker*)
        ct.c_char_p,                          # text (char*)
        ct.c_longlong,                        # tFrom
        ct.c_longlong,                        # tUpto
        ct.c_int                              # nMask
    ]

    # level data
    ced_lib.S64ReadLevels.restype = ct.c_int
    ced_lib.S64ReadLevels.argtypes = [
        ct.c_int,          # nFid
        ct.c_int,          # nChan
        ct.POINTER(ct.c_longlong),  # pData (long long*)
        ct.c_int,          # nMax
        ct.c_longlong,     # tFrom
        ct.c_longlong,     # tTo
        ct.POINTER(ct.c_int),  # nLevel
    ]
    
    return ced_lib

class S64Marker(ct.Structure):
    _fields_ = [
        ('m_Time', ct.c_longlong), 
        ('m_Code1', ct.c_uint8),   
        ('m_Code2', ct.c_uint8),   
        ('m_Code3', ct.c_uint8),
        ('m_Code4', ct.c_uint8) 
    ]
    

def check_for_file(file_name):
    # check for *.smr
    flag = os.path.exists(file_name + '.smr')
    if flag:
        file_name =  file_name + '.smr'
    else:
        # if smr was not found, check for *.smrx
        flag = os.path.exists(file_name + '.smrx')
        if flag:
            file_name =  file_name + '.smrx'


    return flag, file_name

def open_file(file_name, ced_lib):

    
    file_name= file_name.encode('utf-8')
    fhand = ced_lib.S64Open(file_name)

    return fhand

def closeFile(f_handle = -99, ced_lib = None):
    # closeFile: closes .smr(x) files
    # inputs:
    # f_handle: handle to open file
    if f_handle == -99:
        close_flag = ced_lib.S64CloseAll()
    else:
        close_flag = ced_lib.S64Close(f_handle)
    return close_flag


def get_channel_info(fhand, ced_lib, chan_num= -99):
    all_channels =[]
    if chan_num == -99:
        # no channel number specified, get labels for all the channels
        max_chan_count = ced_lib.S64MaxChans(fhand)
        chan_numbers = [chan_num+1 for chan_num in range(max_chan_count)]
    else:
        chan_numbers = [chan_num]
        
    for chan_num in chan_numbers:
        # determine length of channel label
        # Create a buffer to hold the title string (adjust the buffer size as needed)
        dummy_string = b' '
        title_length = ced_lib.S64GetChanTitle(fhand, chan_num, dummy_string, -1)
        if title_length > 0:
            current_channel = channel_info()
            chan_title = ' ' * (title_length-1)
            chan_title = str.encode(chan_title)
            title_length = ced_lib.S64GetChanTitle(fhand, chan_num, chan_title, 0)
            current_channel.label = chan_title.decode('utf-8')
            current_channel.type  = ced_lib.S64ChanType(fhand, chan_num)
            current_channel.number = chan_num 
            all_channels.append(current_channel)
    
    return all_channels

def loadWavemark(fhand, spike_channel, wave_mark, ced_lib):
    # load spike times

    # set mask mode for current wavemark
    mask_handle = waveform_mask(ced_lib, wave_mark)

    # load spikes
    max_events = 1000

    start_tick = 0
    spike_times = []
    exit_flag = False
    while not exit_flag:
        outevpointer = (ct.c_longlong * max_events)()
        iRead = ced_lib.S64ReadEvents(fhand, spike_channel, outevpointer, max_events, start_tick, -1, mask_handle)
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
    return spike_times

def waveform_mask(ced_lib, wave_mark):
    # set mask mode
    mask_handle = 1
    mask_mode = 0
    mask_flag = ced_lib.S64SetMaskMode(mask_handle, mask_mode)
    print(f"mask mode attempt   = {mask_flag}")
    mask = np.zeros((256, 4)).astype('int8')
    mask[wave_mark , 0] = 1
    mask = mask.reshape(-1, 1)
    mask = mask.ctypes.data_as(ct.POINTER(ct.c_int))
    ced_lib.S64SetMaskCodes(mask_handle, mask)
    return mask_flag

def find_channel(all_channels, target_label):
    channel_index = None
    for obj in all_channels:
        if obj.label == target_label:
            channel_index = obj.number
            break
    return channel_index

def load_marker(fhand, channel_number, ced_lib, to_chr=True):

    max_events = 1000
    pData_array = (S64Marker * max_events)()
    maker_info =[]
    exit_flag = False
    start_tick = 0
    time_base = ced_lib.S64GetTimeBase(fhand)
    while not exit_flag:
        n = ced_lib.S64ReadMarkers(fhand, channel_number, pData_array, max_events, start_tick, -1, -1)
        for index in range(n):
            current_marker ={}
            # get time of each marker
            current_marker['time'] = pData_array[index].m_Time * time_base
            # get code
            if to_chr:
                current_marker['code'] = chr(pData_array[index].m_Code1)
            else:
                current_marker['code'] = pData_array[index].m_Code1
            maker_info.append(current_marker)
        if n < max_events:
            exit_flag = True
        else:
            start_tick = pData_array[-1].m_Time + 1
        # breakpoint()
    return maker_info

def load_text(fhand, channel_number, ced_lib):
    max_events = 1000
    pData_array = S64Marker()
    maxTimeTicks = ced_lib.S64ChanMaxTime(fhand, channel_number)
    # Create a buffer to hold the text (adjust the buffer size as necessary)
    
    text_info =[]
    exit_flag = False
    start_tick = 0
    time_base = ced_lib.S64GetTimeBase(fhand)
    # determine size of the text markers
    nRows = np.zeros(1, dtype=np.int32)  # Create a NumPy int32 array to hold nRows
    nCols = np.zeros(1, dtype=np.int32)  # Create a NumPy int32 array to hold nCols
    
    # Call the C function with the correct pointers using ctypes.addressof()
    result = ced_lib.S64GetExtMarkInfo(fhand, channel_number, nRows, nCols)
    print(f"result = {result}")
    
    while not exit_flag:
        text_buffer = ' ' * (nRows[0]+1)
        text_buffer = str.encode(text_buffer)
        
        n = ced_lib.S64Read1TextMark(fhand, channel_number, pData_array,  text_buffer, start_tick, -1, -1)
        
        current_marker ={}
        # get time of each marker
        current_marker['time'] = pData_array.m_Time * time_base

        # get the text
        current_marker['text'] = text_buffer.decode('utf-8').rstrip('\x00')
        text_info.append(current_marker)
        # print(f"start tick = {start_tick}, max tick  = {maxTimeTicks}, text = {current_marker['text']}")
        if n < 0:
            exit_flag = True
        else:
            start_tick = pData_array.m_Time + 1
            if start_tick >= maxTimeTicks:
                exit_flag = True
    return text_info

def load_level_data(fhand, channel_number, ced_lib):
    #S64ReadLevels
    time_base = ced_lib.S64GetTimeBase(fhand)
    max_events = 10000
    exit_flag= False
    start_tick=0
    all_times = []
    all_levels = []
    first_level = ct.c_int()
    initial_level = None
    while not exit_flag:
        times = (ct.c_longlong * max_events)() 
        n = ced_lib.S64ReadLevels(fhand, channel_number, times, max_events, start_tick, -1, ct.byref(first_level))
        all_times.extend(times[0:n])
        if n < max_events:
            exit_flag = True
        else:
            start_tick = times(-1)+1
        if start_tick == 0:
            initial_level = first_level
    all_times = np.array(all_times)
    all_times = all_times * time_base

    # given that the initial value of the level channel = initial_value, then all the even index locations [0, 2, 4, etc]
    # will != initial_value. For whatever reason, (high to low  = 1) and (low to high = 0). We want the opposite
    n = all_times.shape[0]
    all_levels = np.ones(n, dtype=int)
    even_values = np.arange(0,n,2)
    odd_values = np.arange(1,n,2)
    if initial_level.value == 1:
        all_levels[even_values] = 0
        all_levels[odd_values] = 1
    else:
        all_levels[even_values] = 1
        all_levels[odd_values] = 0
    return all_times, all_levels
    # MATINT_API int S64ReadLevels(const int nFid, const int nChan, long long *pData, int nMax,
    #     const long long tFrom, const long long tTo, int* nLevel);

def load_continuous(fhand, channel_number, ced_lib):
    maxTimeTicks = ced_lib.S64ChanMaxTime(fhand, channel_number) # total clock ticks
    n =  np.round( maxTimeTicks/ced_lib.S64ChanDivide(fhand, channel_number)).astype('int') # total samples for lfp channel
    continuous_data = np.zeros(n, dtype=np.float32)
    
    lpf_sampling_interval = ced_lib.S64ChanDivide(fhand, channel_number)*ced_lib.S64GetTimeBase(fhand)
    lfp_Fs = 1.0/lpf_sampling_interval
    continuous_time = np.arange(0, n)*lpf_sampling_interval 

    
    outtimepointer = np.zeros((1, 1), dtype=np.int64)
    maskcode = -1
    try:
        read_flg = ced_lib.S64ReadWaveF(fhand, channel_number, continuous_data, maxTimeTicks, 0, maxTimeTicks, outtimepointer, maskcode)
        continuous_time = continuous_time[0:read_flg]
        continuous_data = continuous_data[0:read_flg]
    except:
        print(f"could not read continuous data from channel {channel_number}, maxTimeticks = {maxTimeTicks} ")
    
    return continuous_data, continuous_time