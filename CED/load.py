from sys import exception
import CED.ced as ced
import CED.eye_tracker as eyd
import spikes.spikes as spikes
import lfp.lfp as lfp
import utility.sys as system_uti
import numpy as np
import ctypes as ct
import pickle
import os
import CED.calculate_trial_info as calculate_trial_info
import CED.calculate_intan_info as calculate_intan_info
import re
import free_view.load as free_view_load
import tuning.tuning as tuning
import fixation 
import matplotlib.pyplot as plt

def load_file(file_name, spike_info, lfp_channel, ced_lib, data_path, load_flag):

    
    parameters = {}
    # make sure all previous files were closed
    ced.closeFile(ced_lib=ced_lib)
    # check for pupil file
    pupil_file = file_name + '.eyd'
    pupil_flag = os.path.exists(pupil_file)


    par_file_name =  file_name + ".par"

    # check for file and open file (if it exists)
    flag, file_name = ced.check_for_file(file_name)

    # load par file and save info to file
    
    par_file_info =  load_par_file(par_file_name)

    with open(data_path + '/par_file_info.pkl', 'wb') as file:
            pickle.dump(par_file_info, file)
    
    if flag:
        fhand = ced.open_file(file_name, ced_lib)
        if fhand > 0:
            print(f"file opened successfully: {file_name}")
        else:
            print(f"file could not be opened: {file_name}")
            return
    else:
        print(f"file not found: {file_name}")
        fhand = None
        return

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
    if spike_info['channel'] > 0:
        spike_times = ced.loadWavemark(fhand, spike_info['channel'], spike_info['wave_mark'], ced_lib)
        spike_data = spikes.spike_data(spike_times, file_name, spike_info['channel'], spike_info['wave_mark'])
    else:
        spike_times = None
        spike_data = None
    

    # load continuous channel
    if lfp_channel > 0:
        lfp_y, lfp_x, fs = ced.load_continuous(fhand, lfp_channel, ced_lib)
        lfp_data = lfp.lfp(lfp_y, fs, file_name, lfp_channel)
    else:
        lfp_y = None
        lfp_x = None
        fs = None
        lfp_data = None
    

    
    
    # load fixation data. depending on when/how the  data was collected, the fixation channel may have various names

    # check for Fixpt
    fixation_channel = ced.find_channel(all_channels, 'Fixpt')
    if fixation_channel == None:
        fixation_channel = ced.find_channel(all_channels, 'FixPt')
        if fixation_channel == None:
            fixation_channel = ced.find_channel(all_channels, 'Fix')
    if fixation_channel != None:
        parameters['fixation']  = ced.load_level_data(fhand, fixation_channel, ced_lib)
    else:
        parameters['fixation'] = []
    
    # load stim channel

    stim_channel = ced.find_channel(all_channels, 'Stim')
    if ready_channel != None:
        parameters['stimulus'] = ced.load_level_data(fhand, stim_channel, ced_lib)
    else:
        parameters['stimulus'] = []

    # load frame channel

    frame_channel = ced.find_channel(all_channels, 'Frame')
    if frame_channel != None:
        parameters['frame']  = ced.load_level_data(fhand, frame_channel, ced_lib)
    else:
        parameters['frame']  = []

    # load intan triggers

    intan_channel = ced.find_channel(all_channels, 'INTAN-T')
    if frame_channel != None:
        parameters['intan']  = ced.load_level_data(fhand, intan_channel, ced_lib)
    else:
        parameters['intan'] = []

    # load intan markers
    intan_marker_channel = ced.find_channel(all_channels, 'INTAN')
    if intan_marker_channel != None:
        parameters['intan_markers'] =  ced.load_marker(fhand, intan_marker_channel, ced_lib)
    else:
        parameters['intan_markers'] = []



    # load keyboard channel
    keyboard_channel = ced.find_channel(all_channels, 'Keyboard')
    
    if keyboard_channel != None:
        parameters['keyboard'] =  ced.load_marker(fhand, keyboard_channel, ced_lib)
    else:
        parameters['keyboard'] = []
    
    # load Textmark channel
    textmark_channel = ced.find_channel(all_channels, 'TextMark')
    if textmark_channel != None:
        parameters['textmark']  =  ced.load_text(fhand, textmark_channel, ced_lib)
    else:
        parameters['textmark']  = []
    

    # load Laser On
    laser_channel = ced.find_channel(all_channels, 'Laser On')
    if laser_channel == None:
        laser_channel = ced.find_channel(all_channels, 'DigMark')
        if laser_channel == None:
            laser_channel = ced.find_channel(all_channels, 'EyeFilePu')
    if laser_channel != None:
        parameters['laser']  = ced.load_marker(fhand, laser_channel, ced_lib,to_chr= False)
    else:
        parameters['laser']  =  []
    
    # load opto signal
    opto_channel = ced.find_channel(all_channels, 'Optic')
    if opto_channel == None:
        opto_channel = ced.find_channel(all_channels, 'OptoWF')
    if opto_channel != None:
        parameters['opto_marker']  =   ced.load_marker(fhand, opto_channel, ced_lib)
    else:
        parameters['opto_marker']  = []
    
    
    if load_flag['eye_tracker']:
        # load eye tracker data
        x_channel = ced.find_channel(all_channels, 'Eye X')
        y_channel = ced.find_channel(all_channels, 'Eye Y')
        eye_tracker = eyd.eye_tracker_data()
        if x_channel != None:
            eye_tracker.x , eye_tracker.time, _ = \
                ced.load_continuous(fhand, x_channel, ced_lib)
            eye_tracker.y , _, _ = \
                ced.load_continuous(fhand, y_channel, ced_lib)
    else:
        eye_tracker = None
    
    
    # load pupil data
    if load_flag['pupil']:
        pupil_data = eyd.pupil_data()
        if pupil_flag:
            # load data from ASL
            eyd_data = eyd.eydexp(pupil_file)
            # load pupil data from Spike2
            pupil_channel = ced.find_channel(all_channels, 'EyeFilePu')
            
            if pupil_channel == None:
                pupil_channel = ced.find_channel(all_channels, 'EyeFilePulse')
                if pupil_channel == None:
                    pupil_channel = ced.find_channel(all_channels, 'INTAN')
            
            if pupil_channel != None:
                pupil_markers =  ced.load_marker(fhand, pupil_channel, ced_lib, to_chr=False)
            else:
                pupil_markers = None
        else:
            print(f"pupil file {pupil_file} not found")
            pupil_markers = None
            eyd_data = None
        
        if pupil_markers and eyd_data:
            eyd_data = eyd.synch_pupil_data(eyd_data, pupil_markers)
            pupil_data.size = eyd_data['data']['pupil']
            pupil_data.time = eyd_data['data']['time']
            pupil_data.status = eyd_data['data']['status']
            
        else:
            pupil_data = None
    else:
        pupil_data = None

    close_flg = ced.closeFile(f_handle=fhand, ced_lib=ced_lib)

    print(f"close flag = {close_flg}")

    # save data to file

    # look for folders and create if they does not exist
    
    system_uti.folder_check(data_path)

    
    if spike_data:
        with open(data_path + '/channel_' + str(spike_data.channel) + '.pkl', 'wb') as file:
            pickle.dump(spike_data, file)

    if lfp_data:
        with open(data_path + '/channel_' + str(lfp_data.channel) + '.pkl', 'wb') as file:
            pickle.dump(lfp_data, file)
    
    with open(data_path + '/eye_tracker_data.pkl', 'wb') as file:
        pickle.dump(eye_tracker, file)
    
    with open(data_path + '/pupil_data.pkl', 'wb') as file:
        pickle.dump(pupil_data, file)
    
    with open(data_path + '/parameters.pkl', 'wb') as file:
        pickle.dump(parameters, file)

    # calculate trial information and completed_trials information

    # imgs files

    # tun files

    # at later date add in code for regular tuning files, box files, etc

    
    # parameter info:
    parameter_info = {}

    # determine all the periords where the stimulus was on
    parameters['stimulus'] = trigger_check(parameters['stimulus'])
    parameter_info['stimulus'] ={}
    new_shape = (int(parameters['stimulus']['times'].shape[0]/2),2)
    parameter_info['stimulus']['times'] = np.reshape(parameters['stimulus']['times'], new_shape)
    parameter_info['stimulus']['times'] = line_noise(parameter_info['stimulus']['times'])

    # fixation periods
    parameters['fixation'] = trigger_check(parameters['fixation'])
    parameter_info['fixation'] ={}
    new_shape = (int(parameters['fixation']['times'].shape[0]/2),2)
    parameter_info['fixation']['times'] = np.reshape(parameters['fixation']['times'], new_shape)
    parameter_info['fixation']['times']  = line_noise(parameter_info['fixation']['times'])

    # Frame pulses
    parameters['frame'] = trigger_check(parameters['frame'])
    new_shape = (int(parameters['frame']['times'].shape[0]/2),2)
    parameter_info['frame'] ={}
    parameter_info['frame']['times'] = np.reshape(parameters['frame']['times'], new_shape)
    parameter_info['frame']['times']  = line_noise(parameter_info['frame']['times'])

    # INTAN pulses
    parameters['intan'] = trigger_check(parameters['intan'])
    new_shape = (int(parameters['intan']['times'].shape[0]/2),2)
    parameter_info['intan'] ={}
    parameter_info['intan']['times'] = np.reshape(parameters['intan']['times'], new_shape)
    parameter_info['intan']['times']  = line_noise(parameter_info['intan']['times'])

    

    # determine the result of each trial :  1 == succcess, 0 == aborted
    # this will be determined by the trail advance text marker that appear after each trial 
    

    # determine trial information for each file type

    # free view files (imgs files)
    if "_imgs_" in data_path:

        # using the fixation window info from par_file_info, create a vector that indicates if fixation was held
        # as a function of time (True = fixation held, False = fixation not held)
        fixation_blocks = fixation.get_fixation_blocks(par_file_info, eye_tracker)
       
        
        # for imgs files collect information on:
        # image_info: when and what images were presented. Assigned trial numbers
        # image_set: key to the image_info['id']
        # trial_info: timing and other information for each trial (attempted stimulus presentation)
        # intan_info: timing and id information for INTAN triggers

        # image_info contains keys: time, id (basically the IV number), name (filename of image)
        # image_set is a list of the unique images presented. image_info['id'] refers to this list
        # image_name: name of image presented (in order of presentation)
        # image_set: list of unique images presented
        # image_id: id for each image presented (id based on order of image_set)
        image_info, image_set = free_view_load.get_image_name(parameters['textmark'])
        
        # trial info:
        # (:,0) = trial_start_time (image series marker)
        # (:,1) = trial_end_time (trial advance marker, ++ == trial success, - == trial aborted) 
        # (:,2) = Image Series ID (part of image series marker)
        # (:,3) = StimStartMarker (stim level change low-> high)
        # (:,4) = StimEnd Marker (stim level change high-> low)
        # (:,5) = Trial result (from trial advance marker, 0 =aborted, 1 = successful), 
        # (:,6) = reward type (0 = no reward, 1 = single reward, 2 = double reward)
        # (:,7) = fixation start time (may be the same for consequtive trials because of hold-fixation option)
        trial_info, parameter_info = calculate_trial_info.free_view(parameters, parameter_info, fixation_blocks)

        # intan_info:
        # (:,0) = StartTime
        # (:,1) = EndTime
        # (:,2) = Trial Number
        # (:,4) = trigger type
        #   1 = file_start, trial_start = 2, stim_frame = 3, trial_success = 4, trial_reward = 5, file_end = 6
        # (:,5) = IV (only for trial_start (tuning) and stim_frame (free view))
        intan_info = calculate_intan_info.get_triggers(parameter_info, trial_info)

        # assign each image presentation to a trial
        image_info = calculate_trial_info.assign_trial_id(image_info, trial_info)
        
        # match the frame levels with image info
        parameter_info['frame'] = free_view_load.image_match_frame(parameter_info['frame'], image_info)

        # Not all frames will have an associated image_id. For example, in CSD files there is a "Stimulus Off" frame
        # that indicates the end of stimulus presentation for that file
        # Let's assign the "Stimulus OFF" frames a image_id ==  -1
        parameter_info['frame'] = free_view_load.find_stim_off_frames(parameter_info['frame'], parameters['textmark'])

        # for image_series files there are "image success markers" == "+". This indicates that fixation was held until the 
        # stimulus was taken down. It is associated with a frame trigger that represents the image being taken down
        # Let's add success (0 = aborted, 1 = success) and offset (time stim taken off screen) to the image_info
        #
        # when the trial was aborted before the image presentation was complete, there is no "+" text marker associated
        # with the image presentation. Thus the offset time will be assigned nan. This is okay since the image presentation
        # is invalid
        image_info = free_view_load.image_offset_match(image_info, parameters['textmark'], parameter_info['frame'])


        ##
        # isolate completed trials
        # return completed_trials with only rows for completed trials
        # rearange other information (e.g., intan_info) so that the trial_numbers correspond to the rows (e.g, intan_info[:,2])
        # of completed trials
        # completed trials have a trial result [:,5] == 1 (0 == aborted trial)
       
        completed_trials = np.where(trial_info[:,5] == 1)[0]
       

        # completed_trials is the old trial_id (e.g., it refers to the row of trial_info)
        new_trial_ids = np.arange(0,completed_trials.shape[0])
        new_trial_vector = np.full((trial_info.shape[0],), np.nan) 
        completed_trial_rows = trial_info[:,5] == 1
        new_trial_vector[completed_trial_rows] = new_trial_ids
        
        # find intan_info that belongs to completed_trials
        valid_intan =  np.isin(intan_info[:,2], completed_trials)
        valid_intan = np.where(valid_intan)[0]
        intan_info =  intan_info[valid_intan,:]
        # Because completed_trials only contains a subset of trials, we need to change intan_info[:,2]
        # so that it points to the correct row of completed_trials
        intan_info[:,2] = new_trial_vector[intan_info[:,2].astype(int)]
        
        # find the image_info that belongs to completed_trials
        valid_image =  np.isin(image_info['trial_id'], completed_trials)
        valid_image = np.where(valid_image)[0]
    
        image_key_list =  image_info.keys() 
        image_key_list = [key for key in image_key_list] 
        for key in image_key_list:
            if isinstance(image_info[key], list):
                image_info[key] =  [image_info[key][valid_index] for valid_index in valid_image]
            else:
                image_info[key] = image_info[key][valid_image]
        
        # Because completed_trials only contains a subset of trials, we need to change image_info['trial_id']
        # so that it points to the correct row of completed_trials
        image_info['trial_id'] = [new_trial_vector[trial_id] for trial_id in image_info['trial_id']]

        # find frame info that belongs to completed_trials
        frame_info = parameter_info['frame']
        valid_frame =  np.isin(frame_info['trial'], completed_trials)
        valid_frame = np.where(valid_frame)[0]
    
        frame_key_list =  frame_info.keys() 
        frame_key_list = [key for key in frame_key_list] 
        for key in frame_key_list:
            if isinstance(frame_info[key], list):
                frame_info[key] =  [frame_info[key][valid_index] for valid_index in valid_frame]
            else:
                frame_info[key] = frame_info[key][valid_frame]
        
        # Because completed_trials only contains a subset of trials, we need to change frame_info['trial']
        # so that it points to the correct row of completed_trials
        frame_info['trial'] = [new_trial_vector[trial_id] for trial_id in frame_info['trial']]


        # complete assignment to completed_trials
        completed_trials = trial_info[completed_trials,:]



        ##
        print(f"saving data to directory = {data_path}")

        with open(data_path + '/image_info.pkl', 'wb') as file:
            pickle.dump(image_info, file)
        
        with open(data_path + '/image_set.pkl', 'wb') as file:
            pickle.dump(image_set, file)

        with open(data_path + '/intan_info.pkl', 'wb') as file:
            pickle.dump(intan_info, file)

        with open(data_path + '/trial_info.pkl', 'wb') as file:
            pickle.dump(trial_info, file)

        with open(data_path + '/completed_trials.pkl', 'wb') as file:
            pickle.dump(completed_trials, file)
        
        with open(data_path + '/frame_info.pkl', 'wb') as file:
            pickle.dump(frame_info, file)
        
        with open(data_path + '/parameter_info.pkl', 'wb') as file:
            pickle.dump(parameter_info, file)
    elif "_tun_" in data_path:
        # using the fixation window info from par_file_info, create a vector that indicates if fixation was held
        # as a function of time (True = fixation held, False = fixation not held)
        fixation_blocks = fixation.get_fixation_blocks(par_file_info, eye_tracker)
    


        
        # Each trial has stimulus parameters (e.g., ori, sf) and a condition (e.g., 'Con' for contrast response function)
        trial_parameters = tuning.get_trial_parameters(parameters['textmark'])
  
        # trial_info: timing and other information for each trial (attempted stimulus presentation)
        trial_info, parameter_info = calculate_trial_info.bakers(parameters, parameter_info, fixation_blocks)

        # intan_info: since the data will already be synced, it is not necessary to load the INTAN triggers
        # intan_info = calculate_intan_info.get_triggers(parameter_info, trial_info)

        # assign each trial_parameter to a trial
        trial_index = np.digitize(trial_parameters['timestamps'], trial_info[:,0])
        trial_parameters['trial'] = trial_index - 1
        # isolate completed trials
        # return completed_trials with only rows for completed trials
        # rearange completed_trial_parameters so that the index values correspond to the rows
        # of completed trials
        # completed trials have a trial result [:,4] == 1 (0 == aborted trial)
       
        completed_trials = np.where(trial_info[:,4] == 1)[0]

        # find trial_parmeters belong to completed trials
        valid_parameters =  np.isin(trial_parameters['trial'], completed_trials)
        valid_parameters = np.where(valid_parameters)[0]
        parameter_key_list =  trial_parameters.keys() 
        parameter_key_list = [key for key in parameter_key_list] 
        for key in parameter_key_list:
            if isinstance(trial_parameters[key], list):
                trial_parameters[key] =  [trial_parameters[key][valid_index] for valid_index in valid_parameters]
            else:
                trial_parameters[key] = trial_parameters[key][valid_parameters]
        completed_trials = trial_info[completed_trials,:]
        trial_parameters['trial'] = np.arange(0,len(trial_parameters['condition']))
        
        ##
        if sum(trial_index == 0) > 0:
            raise Exception('Failed to assign each trial_parameter to a trial!')

        print(f"saving data to directory = {data_path}")
        with open(data_path + '/trial_parameters.pkl', 'wb') as file:
            pickle.dump(trial_parameters, file)
        
        with open(data_path + '/completed_trials.pkl', 'wb') as file:
            pickle.dump(completed_trials, file)

        with open(data_path + '/trial_info.pkl', 'wb') as file:
            pickle.dump(trial_info, file)
        
        with open(data_path + '/parameter_info.pkl', 'wb') as file:
            pickle.dump(parameter_info, file)
    

    else:
        raise Exception('Unknown file format. Currenly only supports "tun" and "imgs" files')
    



    
    return

def load_par_file(par_file):
    par = {}
    current_heading = 'Non Specific'
    if os.path.isfile(par_file):
        with open(par_file, 'r') as file:
            raw_parameters = file.readlines()
        for line_index in range(len(raw_parameters)):
            new_heading = re.match("-{5,}", raw_parameters[line_index])
            colon_indices = [m.start() for m in re.finditer(":", raw_parameters[line_index])]
            if new_heading is not None:
                current_heading = raw_parameters[line_index - 1]
                current_heading = current_heading.replace(" ", "_")
                current_heading = re.sub(r"\n", "", current_heading)
                par[current_heading] = {}

            if current_heading != "Non Specific":
                if current_heading == "Targets":
                    print("add code for star stim")
                elif current_heading == "Stimulus Locations":
                    print("add code for gregg task")
                elif current_heading == 'TrialByTrialOrientations':
                    print("add code for gregg task")
                else:
                    if len(colon_indices) == 1:
                        key, value = raw_parameters[line_index].split(':')
                        # remove illegal characters from key
                        key = fix_key(key)
                        legal_characters = re.findall(r"[^\s]", value)
                        value = "".join(legal_characters)
                        # print(f"key = {key}, value = {value}")
                        par[current_heading][key] = value
                    elif len(colon_indices) == 2:
                        current_line = raw_parameters[line_index]
                        current_line = current_line.replace(":", " ", 1)
                        key, value = current_line.split(':')
                        # remove illegal characters from key
                        key = fix_key(key)
                        legal_characters = re.findall(r"[^\s]", value)
                        value = "".join(legal_characters)
                        par[current_heading][key] = value

    

    return par

def fix_key(key):
    illegal_indices = [m.start() for m in re.finditer(r"[\s(),%#-/:?]", key)]
    illegal_indices = np.array(illegal_indices)
    for index in illegal_indices:
        if index < len(key)-1:
            key = key[:index+1] + key[index+1].upper() + key[index + 2:]
    legal_characters = re.findall(r"[^\s(),%#-/:?]", key)
    key = "".join(legal_characters)
    return key

def trigger_check(triggers):
    # check to make sure the first pulse edge is low - > high(level ==  1) and that there is an even
    # number of edges


    # make sure first level change is low -> high (1 == stimulus onset)
    if triggers['levels'][0] == 0:
        triggers['levels'] = triggers['levels'][1:]
        triggers['times'] =  triggers['times'][1:]

    #make sure the final change == 0 (high -> low)
    if triggers['levels'][-1] == 1:
        triggers['levels'] = triggers['levels'][0:-1]
        triggers['times'] =  triggers['times'][0:-1]

    if np.mod(triggers['times'].shape[0],2) != 0:
        raise Exception("there is an odd number of triggers levels!")
    
    return triggers

def line_noise(triggers):
        # make sure there is no line noise. Line noise is present when the width of a trigger pair (On->Off) is an ~impulse
    #(e.g., 2 consequtive trigger levels are nearly identical)
    d = np.diff(triggers)
    triggers[d[:,0] > 0.001,:]
    if np.min(triggers) < 0.001:
        raise Exception("Inspect this file for line noise to make sure we are dealing with it properly")
    
    return triggers


def load_level_channel(fhand, channel_number, ced_lib):
    if channel_number != None:
        level_channel  = ced.load_level_data(fhand, channel_number, ced_lib)
    else:
       level_channel = []
       print(f'level channel not found: {channel_number}')

    level_channel = trigger_check(level_channel)
    new_shape = (int(level_channel['times'].shape[0]/2),2)
    level_info ={}
    level_info['times'] = np.reshape(level_channel['times'], new_shape)
    level_info['times']  = line_noise(level_info['times'])

    return level_info



