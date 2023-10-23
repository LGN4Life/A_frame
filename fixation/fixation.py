__all__ = ['fixation_state', 'fixation_distance', 'calculate_fixation_state', 'fixation_state_change', 
           'get_fixation_blocks', 'start_times']
import numpy as np
import CED.eye_tracker as eyd
from typing import Any

def fixation_state(fixation_position: np.ndarray[np.float64, Any], fixation_radius: np.ndarray[np.float64, Any], \
                            eye_gain: np.ndarray[np.float64, Any], eye_tracker: eyd.eye_tracker_data):
    """
    determine distance from fixation point and the fixation state
    input:
      fixation_position: [x, y] (°)
      fixation_radius: size of fixation window (°)
      eye gain: gain of eye tracker (volts/degree)
      eye_tracker: class object with eye_tracker info (obj.x ==  horizontal eye position, obj.y ==  vertical eye position; in volts)
    output:
       fixation_info:
          state: fixation state as function of time (1 ==  fixation held, 0 == not held)
          distance: eye position distance from fixation point (°)
          block: [:,0] = fixation block start, [:,1] = fixation block end
    """
    
    fixation_info ={}
    # calculate distance from fixation point
    
    fixation_info['distance'] = fixation_distance(eye_tracker, fixation_position, eye_gain)
    
    # calculate fixation state
    fixation_info['state'] = calculate_fixation_state(fixation_info['distance'], fixation_radius)

    # calculate fixation state changes
    fixation_info['block'] = fixation_state_change(fixation_info['state'],eye_tracker.x.shape[0]-1)
   




    return fixation_info

def fixation_distance(eye_tracker, fixation_position, eye_gain):

    # translate eye_tracker singal from volts to degrees

    eye_tracker.x = eye_tracker.x * eye_gain[0]
    eye_tracker.y = eye_tracker.y * eye_gain[1]

    eye_distance = np.sqrt(np.square(eye_tracker.x - fixation_position[0]) + np.square(eye_tracker.y - fixation_position[1]))


    return eye_distance

def calculate_fixation_state(eye_distance, fixation_radius):

    # determine fixation_state ()
    fixation_state = np.full((eye_distance.shape[0],), 0)
    fixation_state[eye_distance < 2*fixation_radius] = 1


    return fixation_state
def fixation_state_change(fixation_state, time_limit):
    # returns index values where the fixation state changes.
    #input:
    #   fixation: vector (1 = fixation, 0 = no fixation) over time
    #   time_limit: number of time bins in eye_tracker traces
    #output:
    #   fixation_block = [:,0] = begin fixation block
    #                    [:,1] =  end fixation block

    # Compute the differences between adjacent elements
    diff_array = np.diff(fixation_state)

    # Find the indices where value changes from no fix -> fix (0 -> 1)
    fixation_on = np.where(diff_array == 1)[0]

    # Find the indices where value changes from fix -> no fix (1 -> 0)
    fixation_off = np.where(diff_array == -1)[0]

    # combine fixation on and off into fixation blocks
    # start with first fixation on
    fixation_off = fixation_off[fixation_off > fixation_on[0]]
    # if there is no off_trigger after the last on_trigger, add an off trigger at inf
    if fixation_off[-1] < fixation_on[-1]:
        fixation_off = np.append(fixation_off, time_limit)
    
    fixation_block = np.column_stack((fixation_on, fixation_off)).astype(int)
    return fixation_block

def get_fixation_blocks(par_file_info, eye_tracker):
    """
    using the fixation window info from par_file_info, create a vector that indicates if fixation was held
    as a function of time (True = fixation held, False = fixation not held)
    input:
        par_file_info: dictionary containing information from SMR *.par
        eye_tracker: eye_tracker info from SMR file
    output:
        fixation_time: dictionary with "block" (onset and offset times for each fixation block)
    """
    if 'Fixation_point' in par_file_info.keys():
        # 'code for bakers files here'
        fixation_location =  np.array([par_file_info['Fixation_point']['PositionXDegrees'], \
                                par_file_info['Fixation_point']['PositionYDegrees']]).astype(float)
        fixation_radius =  np.array(par_file_info['Fixation_point']['WindowRadius']).astype(float)
        eye_gain = np.array([par_file_info['General_information']['EyeCoilSoftwareGainX'], \
                            par_file_info['General_information']['EyeCoilSoftwareGainY']]).astype(float)
        
    elif 'FixationX' in par_file_info['Experimental_parameters'].keys():
        # 'code for imgs files here'
        fixation_location =  np.array([par_file_info['Experimental_parameters']['FixationX'], \
                                par_file_info['Experimental_parameters']['FixationY']]).astype(float)
        fixation_radius =  np.array(par_file_info['Experimental_parameters']['FixationWindowRadius']).astype(float)
        eye_gain = np.array([par_file_info['General_information']['EyeCoilSoftwareGainX'], \
                            par_file_info['General_information']['EyeCoilSoftwareGainY']]).astype(float)
    else:
        raise Exception('Fixation information was not located in par_file_info')
    
    
    
    fixation_info = fixation_state(fixation_location, fixation_radius, eye_gain, eye_tracker)
    
    fixation_blocks = eye_tracker.time[fixation_info['block']]

    return fixation_blocks


def start_times(fixation_times: np.ndarray[np.float64, np.dtype], stim_triggers: np.ndarray[np.float64, np.dtype]) \
    -> np.ndarray[np.float64, np.dtype]:
    """
    assign each visual stimulation a fixation start time (i.e., when the animal started fixating prior to the stimulus onset )
    input:
        fixation_times: (N,2) matrix of fixation start and end times
        stim_triggers: (N,) vector of stimulus onset times
    output:
        fixation_start_times: (N,) vector of fixation start times
    """
    
    fixation_start_time = np.full((stim_triggers.shape[0],), np.nan)

    # turn (N,2) fixation_times into a (N*2,) vector 
    fix_edges = np.reshape(fixation_times,(fixation_times.size,))
    # find bins where trial stimulation occured
    fix_index = np.digitize(stim_triggers, fix_edges)
    # some trials have no stimulus (trial aborted very early), mark these as -1 so it is known that they are not valid
    # these trials will not be included because of valid_stim_trials (see below)
    fix_index[np.isnan(stim_triggers)] = -1
    # in vector space (N*2,) all stim times should occur during odd bins (odd = during fixation, even = between fixation)
    if np.sum(np.mod(fix_index,2)==0)>0:
        breakpoint()
        raise Exception('Problem assigning stim blocks to fixation period! Detected stim block during no fixation!')
    fix_index =  ((fix_index+1)/2).astype(int) # convert from vector index to row index
    valid_stim_trials = ~np.isnan(stim_triggers) 

    fixation_start_time[valid_stim_trials] = fixation_times[fix_index[valid_stim_trials]-1,0]

    return fixation_start_time