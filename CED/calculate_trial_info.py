import numpy as np
import re
import matplotlib.pyplot as plt
import fixation

def free_view(parameters, parameter_info, fixation_blocks):
    # calculate the trial information for FreeView files (imgs files, e.g., tmj_230811_imgs_000.smrx)
    # input:
    #   parameters: parameters['textmark'] contains the text info from the TextMark channel (typically channel 30)
    #   paramter_info: trigger timing for stimulus, fixation, frame, intan (e.g., paramter_info['stimulus'])
    #   fixation_blocks: (N,2) np array with fixaiton times: [:,0] = fixation onset, [:,1] = fixation offset
    # each attempted trial is defined by Spike2 attempting to put a stimulus on the screen
    # trial info:
    # (:,0) = trial_start_time (image series marker)
    # (:,1) = trial_end_time (trial advance marker, ++ == trial success, - == trial aborted) 
    # (:,2) = Image Series ID (part of image series marker)
    # (:,3) = StimStartMarker (stim level change low-> high)
    # (:,4) = StimEnd Marker (stim level change high-> low)
    # (:,5) = Trial result (from trial advance marker, 0 =aborted, 1 = successful), 
    # (:,6) = reward type (0 = no reward, 1 = single reward, 2 = double reward)
    # (:,7) = fixation start time (may be the same for consequtive trials because of hold-fixation option)
    column_number = 8

    # the start of a trial is indicated by the image series marker (e.g., Image series 92 start)
    # find image start text (supplies image series index)
    image_series_index = []
    image_series_value = []
    for index, s in enumerate(parameters['textmark'].markers):
        match = re.search(r'Image series (\d{1,5}) start', s)
        if match:
            image_series_index.append(index)
            image_series_value.append(int(match.group(1)))

    image_series_time = [parameters['textmark'].times[index] for index in image_series_index]

    trial_count = len(image_series_time )
    trial_info = np.full((trial_count, column_number), np.nan)
    trial_info[:,0] = image_series_time
    trial_info[:,2] = image_series_value

    # the end of a trial is the trial advance marker (++ == trial success, - == trial aborted)

    # find success trial advance markers
    trial_success_index  = [index for index, s in enumerate(parameters['textmark'].markers) if s == '++']
    trial_success_time = [parameters['textmark'].times[i] for i in trial_success_index]

    # assign each trial advance marker to a trial
    success_trial_index = np.digitize(trial_success_time, trial_info[:,0])
    if sum(success_trial_index == 0) > 0:
        raise Exception('Failed to assign each trial advnace marker to a trial!')
    trial_info[success_trial_index-1, 5] = 1
    trial_info[success_trial_index-1, 1] = trial_success_time

    # find aborted trial advance markers
    trial_aborted_index  = [index for index, s in enumerate(parameters['textmark'].markers) if s == '-']
    trial_aborted_time = [parameters['textmark'].times[i] for i in trial_aborted_index]

    # assign each trial advance marker to a trial
    aborted_trial_index = np.digitize(trial_aborted_time, trial_info[:,0])
    if sum(aborted_trial_index == 0) > 0:
        raise Exception('Failed to assign each trial advance marker to a trial!')
    trial_info[aborted_trial_index-1, 5] = 0
    trial_info[aborted_trial_index-1, 1] = trial_aborted_time

    # it is possible that the trial is aborted after the image series marker, but before the stimulus is placed on 
    # the screen. In this case, there may not be a trial advance marker (++ or -).  When we encounter this, let's explore
    # to make sure everything is kosher
    if sum(np.isnan(trial_info[:,5])) > 0:
        print("some trials do not have a trial advance marker!")
        no_result_trials = np.where(np.isnan(trial_info[:,5]))[0]
        # it seems like this indeed the case. Some trials are aborted before the stimulus was displayed on the screen. Here
        # there are no trial advance markers

    

    # stimulus markers indicate the period of time for the preentation of a series of images

    # assign each stimulus set trigger to a trial
    stimulus_group_index = np.digitize(parameter_info['stimulus']['times'][:,0], trial_info[:,0])
    if sum(stimulus_group_index == 0) > 0:
        raise Exception('Failed to assign each stimulus group to a trial!')
    trial_info[stimulus_group_index-1, 3:5] = parameter_info['stimulus']['times']

    # all stimulus presentations of images are indicated by "frame" pulses (~4 ms duration)
    # assign each frame pulse to a trial
    frame_index = np.digitize(parameter_info['frame']['times'][:,0], trial_info[:,0])
    parameter_info['frame']['trial'] = frame_index - 1
    if sum(frame_index == 0) > 0:
        raise Exception('Failed to assign each frame to a trial!')
    
    # assign each reward textmark to a trial.
    # There are 3 potential reward types: No Reward, Reward, Double Reward
    reward_index = []
    reward_value = []
    for index, s in enumerate(parameters['textmark'].markers):
        if s == "No Reward":
            reward_index.append(index)
            reward_value.append(0)
        elif s == "Reward":
            reward_index.append(index)
            reward_value.append(1)
        elif s == "Double Reward":
            reward_index.append(index)
            reward_value.append(2)
    reward_time = [parameters['textmark'].times[index] for index in reward_index]
    reward_trial_index = np.digitize(reward_time, trial_info[:,0])
    if sum(reward_trial_index  == 0) > 0:
        raise Exception('Failed to assign each reward to a trial!')
    trial_info[reward_trial_index-1,6] = reward_value

    # determine when the animal began fixation prior to the trial
    # assign each trial a fixation start time(i.e., fixation_blocks[:,0])

    trial_info[:,7] = fixation.start_times(fixation_blocks, trial_info[:,3])

    # determine fixation start time for each trial
    stim_onset = trial_info[:,3]
    # throw out aborted trials
    aborted_trials = trial_info[:,5] == 0
    stim_onset[aborted_trials] = np.nan
    trial_info[:,7] = fixation.start_times(fixation_blocks, stim_onset)


    return trial_info, parameter_info

def bakers(parameters, parameter_info, fixation_blocks):
    # calculate the trial information for bakers files (tun files, e.g., tmj_230811_tun_001.smrx)
    # input:
    #   parameters: parameters['textmark'] contains the text info from the TextMark channel (typically channel 30)
    #   paramter_info: trigger timing for stimulus, fixation, frame, intan (e.g., paramter_info['stimulus'])
    # each attempted trial is defined by Spike2 attempting to put a stimulus on the screen
    # trial info:
    # (:,0) = trial_start_time (trial start trigger series)
    # (:,1) = trial_end_time (trial advance marker, ++ == trial success, - == trial aborted) 
    # (:,2) = StimStartMarker (stim level change low-> high)
    # (:,3) = StimEnd Marker (stim level change high-> low)
    # (:,4) = Trial result (from trial advance marker, 0 =aborted, 1 = successful), 
    # (:,5) = reward type (0 = no reward, 1 = single reward, 2 = double reward)
    # (:,6) = fixation start time (may be the same for consequtive trials because of hold-fixation option)
    column_number = 7

    # the start of a trial is indicated by a intan "Trial Start" trigger. Find these triggers (trigger duration ~ 3ms)
    intan_duration = np.diff(parameter_info['intan']['times'])
    intan_duration = np.reshape(intan_duration,(intan_duration.shape[0],))
    intan_file_start = (intan_duration > 0.0029) & (intan_duration< 0.0031)

    trial_count = np.sum(intan_file_start)
    trial_info = np.full((trial_count, column_number), np.nan)
    trial_info[:,0] = parameter_info['intan']['times'][intan_file_start,0]

    

    # the end of a trial is the trial advance marker (+ == trial success, - == trial aborted)

    # find success trial advance markers
    trial_success_index  = [index for index, s in enumerate(parameters['textmark'].markers) if s == '+']
    trial_success_time = [parameters['textmark'].times[i] for i in trial_success_index]

    # assign each trial advance marker to a trial
    success_trial_index = np.digitize(trial_success_time, trial_info[:,0])
    if sum(success_trial_index == 0) > 0:
        raise Exception('Failed to assign each trial advnace marker to a trial!')
    trial_info[success_trial_index-1, 4] = 1
    trial_info[success_trial_index-1, 1] = trial_success_time

    # find aborted trial advance markers
    trial_aborted_index  = [index for index, s in enumerate(parameters['textmark'].markers) if s == '-']
    trial_aborted_time = [parameters['textmark'].times[i] for i in trial_aborted_index]

    # assign each trial advance marker to a trial
    aborted_trial_index = np.digitize(trial_aborted_time, trial_info[:,0])
    if sum(aborted_trial_index == 0) > 0:
        raise Exception('Failed to assign each trial advance marker to a trial!')
    trial_info[aborted_trial_index-1, 4] = 0
    trial_info[aborted_trial_index-1, 1] = trial_aborted_time

    # it is possible that the trial is aborted after the start_trial trigger, but before the stimulus is placed on 
    # the screen. In this case, there may not be a trial advance marker (+ or -).  
    if sum(np.isnan(trial_info[:,4])) > 0:
        print("some trials do not have a trial advance marker!")


    # stimulus markers indicate the period of time for the preentation of a series of images

    # assign each stimulus set trigger to a trial
    stimulus_group_index = np.digitize(parameter_info['stimulus']['times'][:,0], trial_info[:,0])
    if sum(stimulus_group_index == 0) > 0:
        raise Exception('Failed to assign each stimulus group to a trial!')
    trial_info[stimulus_group_index-1, 2:4] = parameter_info['stimulus']['times']



    
    # assign each reward textmark to a trial.
    # There are 3 potential reward types: No Reward, Reward, Double Reward
    reward_index = []
    reward_value = []
    for index, s in enumerate(parameters['textmark'].markers):
        if s == "No Reward":
            reward_index.append(index)
            reward_value.append(0)
        elif s == "Reward":
            reward_index.append(index)
            reward_value.append(1)
        elif s == "Double Reward":
            reward_index.append(index)
            reward_value.append(2)
    reward_time = [parameters['textmark'].times[index] for index in reward_index]
    reward_trial_index = np.digitize(reward_time, trial_info[:,0])
    if sum(reward_trial_index  == 0) > 0:
        raise Exception('Failed to assign each reward to a trial!')
    trial_info[reward_trial_index-1,5] = reward_value

    # determine fixation start time for each trial
    stim_onset = trial_info[:,2]
    # throw out aborted trials
    aborted_trials = trial_info[:,4] == 0
    stim_onset[aborted_trials] = np.nan
    trial_info[:,6] = fixation.start_times(fixation_blocks, stim_onset)

    return trial_info, parameter_info






def assign_trial_id(image_info, trial_info):


    image_count = len(image_info['time'])
    image_info['trial_id'] = np.full((image_count,), np.nan)

    trial_edges = trial_info[:,0]

    image_index = np.digitize(image_info['time'], trial_edges)

    image_info['trial_id'] = image_index - 1

    return image_info


