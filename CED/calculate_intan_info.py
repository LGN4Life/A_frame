import numpy as np
import OpenEphys.spike2_sync as sync

def get_triggers(parameter_info, trial_info):
    # calculate information about INTAN triggers
    # intan_info:
        # (:,0) = StartTime
        # (:,1) = EndTime
        # (:,2) = Trial Number
        # (:,4) = trigger type
        #   1 = file_start, trial_start = 2, stim_frame = 3, trial_success = 4, trial_reward = 5, file_end = 6
        # (:,5) = IV (only for trial_start (tuning) and stim_frame (free view))

    column_number = 6

    # save start and end time of each intan trigger
    trigger_count = parameter_info['intan']['times'].shape[0]
    intan_info = np.full((trigger_count, column_number),np.nan)

    intan_info[:,0:2] = parameter_info['intan']['times']

    # assign each intan trigger to a trial
    trigger_index = np.digitize(intan_info[:,0], trial_info[:,0])
    # not all intan triggers will be assigned to a trial. The first trigger occurs 
    # at teh start of the SMR file to indicate the start of the SMR file to the
    # OpenEphys file    
    intan_info[:, 2] = trigger_index

    intan_info[:, 4] = assign_intan_trigger_id(intan_info[:,0:2])


    return intan_info

def assign_intan_trigger_id(triggers):
    # duration code :
        # 3 ms Trial Start
        # 4 ms stimulus on, currently only for free view experiments (i.e., imgs files)
        # 6 ms Trial Success
        # 8 ms Maintain Fixation Success Prior To Secondary Reward in Tuning 
        # 10 ms End of Session
        # 15 ms Box O' Donuts Tuning
        # 20 ms Start of Sampling or Session
        # 25 ms Baker's Dozen Tuning
        # 30 ms Orientation Tuning
        # 35 ms Contrast Tuning
        # 40 ms Area Tuning
        # 45 ms Spatial Frequency Tuning
        # 50 ms Temporal Frequency Tuning
        # 55 ms XY Position Tuning

    # determine identity of each pulse, based on duration
    trigger_id = np.full((len(triggers),), np.nan)
    trigger_duration = np.diff(triggers)
    trigger_duration = np.reshape(trigger_duration,(trigger_duration.shape[0],))

    # start of a new smr file == 1
    trigger_id[trigger_duration > 0.0149] = 1
    # start of trial == 2
    # breakpoint()
    trigger_id[(trigger_duration > .0029) & (trigger_duration < .0031)] = 2
    # stimulus onset == 3
    trigger_id[(trigger_duration > .0039) & (trigger_duration < .0041)] =  3
    # trial success ==4
    trigger_id[(trigger_duration > .0059) & (trigger_duration < .0061)] =  4
    # trial reward (reward or no reward timing) == 5
    trigger_id[(trigger_duration > .0079) & (trigger_duration < .0081)] =  5
    # end of SMR file == 6
    trigger_id[(trigger_duration > .0099) & (trigger_duration < .0101)] =  6

    return trigger_id