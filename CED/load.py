import CED.ced as ced
import numpy as np
import ctypes as ct
import CED.eye_tracker as eyd
import os
def load_file(file_name, spike_info, lfp_channel, ced_lib):

    

    # make sure all previous files were closed
    ced.closeFile(ced_lib=ced_lib)
    # check for pupil file
    pupil_file = file_name + '.eyd'
    pupil_flag = os.path.exists(pupil_file)

    # check for file and open file (if it exists)
    flag, file_name = ced.check_for_file(file_name)

    if flag:
        fhand = ced.open_file(file_name, ced_lib)
        if fhand > 0:
            print(f"file opened successfully: {file_name}")
        else:
            print(f"file could not be opened: {file_name}")
    else:
        print(f"file not found: {file_name}")


    all_channels = ced.get_channel_info(fhand, ced_lib)


    print(f"{len(all_channels)} channels processed")

    # load ready channel
    ready_channel = ced.find_channel(all_channels, 'Ready')
    if ready_channel != None:
        ready_times, ready_levels = ced.load_level_data(fhand, ready_channel, ced_lib)
    else:
        ready_times = []
        ready_levels = []


    # load spike channel 
    spike_times = ced.loadWavemark(fhand, spike_info['channel'], spike_info['wave_mark'], ced_lib)

    print(f"spikes_loaded: n = {spike_times.shape[0]}")
    
    

    # load continuous channel

    lfp_data, lfp_time = ced.load_continuous(fhand, lfp_channel, ced_lib)

    # load fixation data. depending on when/how the  data was collected, the fixation channel may have various names

    # check for Fixpt
    fixation_channel = ced.find_channel(all_channels, 'Fixpt')
    if fixation_channel == None:
        fixation_channel = ced.find_channel(all_channels, 'FixPt')
        if fixation_channel == None:
            fixation_channel = ced.find_channel(all_channels, 'Fix')
    if fixation_channel != None:
        fixation_times, fixation_levels = ced.load_level_data(fhand, fixation_channel, ced_lib)
    else:
        fixation_times = []
        fixation_levels = []

    # load stim channel

    stim_channel = ced.find_channel(all_channels, 'Stim')
    if ready_channel != None:
        stim_times, stim_levels = ced.load_level_data(fhand, stim_channel, ced_lib)
    else:
        stim_times = []
        stim_levels = []

    # load frame channel

    frame_channel = ced.find_channel(all_channels, 'Frame')
    if frame_channel != None:
        frame_times, frame_levels = ced.load_level_data(fhand, frame_channel, ced_lib)
    else:
        frame_times = []
        frame_levels = []

    # load intan triggers

    intan_channel = ced.find_channel(all_channels, 'INTAN-T')
    if frame_channel != None:
        intan_times, intan_levels = ced.load_level_data(fhand, intan_channel, ced_lib)
    else:
        intan_times = []
        intan_levels = []

    # load intan markers
    intan_marker_channel = ced.find_channel(all_channels, 'INTAN')
    if intan_marker_channel != None:
        intan_marker_data =  ced.load_marker(fhand, intan_marker_channel, ced_lib)
    else:
        intan_marker_data = []

    

    # load keyboard channel
    keyboard_channel = ced.find_channel(all_channels, 'Keyboard')
    
    if keyboard_channel != None:
       keyboard_data =  ced.load_marker(fhand, keyboard_channel, ced_lib)
    else:
        keyboard_data = []
    
    # load Textmark channel
    textmark_channel = ced.find_channel(all_channels, 'TextMark')
    if textmark_channel != None:
        textmark_data =  ced.load_text(fhand, textmark_channel, ced_lib)
    else:
        textmark_data = []
    
    # load Laser On
    laser_channel = ced.find_channel(all_channels, 'Laser On')
    if laser_channel == None:
        laser_channel = ced.find_channel(all_channels, 'DigMark')
        if laser_channel == None:
            laser_channel = ced.find_channel(all_channels, 'EyeFilePu')
    if laser_channel != None:
        laser_data = ced.load_marker(fhand, laser_channel, ced_lib,to_chr= False)
    else:
        laser_data = []
    
    # load opto signal
    opto_channel = ced.find_channel(all_channels, 'Optic')
    if opto_channel == None:
        opto_channel = ced.find_channel(all_channels, 'OptoWF')
    if opto_channel != None:
        opto_data =  ced.load_marker(fhand, opto_channel, ced_lib)
    else:
        opto_data = []

    # load eye tracker data
    x_channel = ced.find_channel(all_channels, 'Eye X')
    y_channel = ced.find_channel(all_channels, 'Eye Y')

    if x_channel != None:
        x_data, x_time = ced.load_continuous(fhand, x_channel, ced_lib)
        y_data, y_time = ced.load_continuous(fhand, y_channel, ced_lib)
    else:
        x_data = [], x_time =[]
        y_data = [], y_time =[]
    
    
    
    # load pupil data
    
    if pupil_flag:
        # load data from ASL
        eyd_data = eyd.eydexp(pupil_file)
        # load pupil data from Spike2
        pupil_channel = ced.find_channel(all_channels, 'EyeFilePu')
        if pupil_channel !=None:
            pupil_data =  ced.load_marker(fhand, pupil_channel, ced_lib, to_chr=False)
        else:
            pupil_data = []
    else:
        print(f"pupil file {pupil_file} not found")

    eyd_data = eyd.synch_pupil_data(eyd_data, pupil_data)
    close_flg = ced.closeFile(f_handle=fhand, ced_lib=ced_lib)
    print(f"close flag = {close_flg}")
    return spike_times, lfp_data, lfp_time, fixation_levels, fixation_times, textmark_data


