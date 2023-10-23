from logging import raiseExceptions
import re
import numpy as np
import pickle

from traitlets import Float
import spikes.spikes as spikes
import matplotlib.pyplot as plt
import pdb


class TuningData:
    def __init__(self, smr_directory, smr_file, spike_info):
        self.smr_directory = smr_directory
        self.smr_file = smr_file
        self.spike_info = spike_info
        self.spike_info.sync_spike_times(smr_directory + smr_file) # sync spike times (self.spike_info.times)
        self.trial_parameters = self.TrialParameters(smr_directory + smr_file)
    
    
        

        
    class TrialParameters:
        def __init__(self, parameter_directory, iv_list = None):
            self.fixation_start = np.array([])
            self.stimulus_triggers = np.array([])
            if iv_list:
                self.iv_list = iv_list
            else:
                self.iv_list =["CON", "ORI", "TF", "SF", "WH", "XYPOS"]
            self.get_parameters(parameter_directory)
            
        def get_parameters(self, parameter_directory):
            """extract parametes from completed_trials and trial_parameters (calculated by CED.load.load_file)

            Args:
                parameter_directory (str): directory for completed_trials and trial_parameters
            """
            with open(parameter_directory + '\\parameters\\completed_trials.pkl', 'rb') as file:
                completed_trials  = pickle.load(file)
            with open(parameter_directory +  '\\parameters\\trial_parameters.pkl', 'rb') as file:
                trial_parameters  = pickle.load(file)
                
            self.stimulus_triggers = completed_trials[:,2:4]
            self.fixation_start = completed_trials[:,6]
            self.condition = np.array(trial_parameters['condition'])

            for iv in self.iv_list:
                current_values = trial_parameters[iv]
                current_iv_values = None
                for index, str_values in enumerate(current_values):
                    str_values = str_values.split(',')
                    iv_as_float = np.array(str_values,dtype=float)
                    if index == 0:
                        current_iv_values = np.full((completed_trials.shape[0], len(str_values)), np.nan)
                  
                    current_iv_values[index,:] = iv_as_float
                setattr(self, iv, current_iv_values)
    class TuningCurve:
        def __init__(self, condition, iv, cluster_id, trial_parameters, spike_info):
            self.cluster_id = cluster_id
            self.condition = condition
            self.iv = iv
            self.trial_parameters = trial_parameters
            self.trial_iv = getattr(self.trial_parameters,iv)
            self.tuning_trials = self.trial_parameters.condition == self.condition
            self.trial_iv = self.trial_iv[self.tuning_trials]
            self.trial_tf = self.trial_parameters.TF[self.tuning_trials]
            self.IV = np.unique(self.trial_iv)
            self.get_cluster_spikes(spike_info)
            
            
        def get_cluster_spikes(self, spike_info):
            cluster_spikes_logical =  np.isin(spike_info.cluster_ids, self.cluster_id)
            self.spike_times = np.array(spike_info.times[cluster_spikes_logical])
            
                
        def calculate_trial_psth(self, bin_size = 0.001, duration = 0.8, pre_window = -.5, first_cycle = True):
                
            self.bin_size = bin_size
            max_cycle_duration = 1/np.min(self.trial_tf)
            self.trial_cycle_duration = 1/self.trial_tf
            
            n = np.sum(self.tuning_trials)
            self.trial_cycle_x = np.arange(0,max_cycle_duration+self.bin_size, self.bin_size)
            self.trial_cycle_hist = np.full((n,self.trial_cycle_x.shape[0]-1), np.nan)
            
            stim_triggers =  self.trial_parameters.stimulus_triggers[self.tuning_trials,:]
            trial_number = stim_triggers.shape[0]
            self.trial_psth_x = np.arange(pre_window, duration+bin_size, bin_size)
            self.trial_psth = np.full((trial_number, self.trial_psth_x.shape[0]-1), np.nan)
            self.iv_psth = np.full((self.IV.shape[0], self.trial_psth_x.shape[0]-1), np.nan)
            self.iv_cycle_hist = np.full((self.IV.shape[0], self.trial_cycle_x.shape[0]-1), np.nan)
            for index, current_triggers in enumerate(stim_triggers):
                # get trial spikes
                current_spike_logical = (self.spike_times >= current_triggers[0] + pre_window) \
                    & (self.spike_times <= current_triggers[0]+duration)
                current_spike_times = self.spike_times[current_spike_logical] - current_triggers[0]
                self.trial_psth[index,:], bin_edges = np.histogram(current_spike_times, self.trial_psth_x)
                self.trial_psth[index,:] = self.trial_psth[index,:] / bin_size
                if self.trial_tf[index] > 0:
                    current_cycle_duration = 1/self.trial_tf[index]
                    current_cycle_number = duration/current_cycle_duration
                    if first_cycle:
                        current_mod_spikes = np.mod(current_spike_times[current_spike_times>0], current_cycle_duration)
                    else:
                        current_mod_spikes = np.mod(current_spike_times[current_spike_times>current_cycle_duration], current_cycle_duration)
                    current_cycle_x = np.arange(0,current_cycle_duration+self.bin_size, self.bin_size)
                    current_cycle_hist, edges = np.histogram(current_mod_spikes, current_cycle_x)
                    self.trial_cycle_hist[index,0:current_cycle_hist.shape[0]] =  current_cycle_hist / current_cycle_number / bin_size
            for iv_index, IV in enumerate(self.IV):
                current_trials = self.trial_iv == IV
                self.iv_psth[iv_index,:] = np.mean(self.trial_psth[current_trials.flatten(),:], axis = 0)
                self.iv_cycle_hist[iv_index,:] = np.mean(self.trial_cycle_hist[current_trials.flatten(),:], axis = 0)

        def calculate_firing_rate(self):
            self.f0 = np.full(self.trial_cycle_hist.shape[0], np.nan)
            self.f1 = np.full(self.trial_cycle_hist.shape[0], np.nan)
            self.F0 = np.full(self.IV.shape[0], np.nan)
            self.F1 = np.full(self.IV.shape[0], np.nan)
            
            for index, cycle_hist in enumerate(self.trial_cycle_hist):
                
                if self.trial_tf[index] > 0:
                    cycle_duration = self.trial_cycle_duration[index]
                    valid = self.trial_cycle_x < cycle_duration
                    ps = np.abs(np.fft.fft(cycle_hist[valid[0:-1].flatten()])) / cycle_hist.shape[0]
                    self.f0[index] = ps[0]
                    self.f1[index] = 2*ps[1]
                else:
                    self.f0[index] = np.mean(self.trial_psth[index,:])
            for iv_index, IV in enumerate(self.IV):
                current_trials = self.trial_iv == IV
                self.F0[iv_index] = np.mean(self.f0[current_trials.flatten()])
                self.F1[iv_index] = np.mean(self.f1[current_trials.flatten()])
                
            
        def plot_tuning(self):
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.IV, self.F0)
            axs[0].plot(self.IV, self.F1)
            axs[1].plot(self.trial_psth_x[0:-1], self.iv_psth.T)
        
        def plot_trial_psth(self):
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.trial_psth_x[0:-1],np.mean(self.trial_psth, axis = 0))
            if self.condition != 'TF':
                axs[1].plot(self.trial_cycle_x[0:-1],np.mean(self.trial_cycle_hist, axis = 0))
            
                
    
    
        
    


def get_smr_info(smr_file_list):
    """get smr information that was extracted from *.smr(x) files by ced.load.load_file()
    Args:

    Returns:
    """
    
    
    sync_info_directory = [None] * len(smr_file_list)
    smr_folder = [None] * len(smr_file_list)
    completed_trials = [None] * len(smr_file_list)
    spike_info = [None] * len(smr_file_list)
    frame_info = [None] * len(smr_file_list)
    trial_parameters = [None] * len(smr_file_list)
    for index, current_smr in enumerate(smr_file_list):
        
        sync_info_directory[index] = base_directory + experiment_name + current_smr +  "\\sync_info\\sync_info.pkl"
        smr_folder[index] = base_directory + experiment_name + current_smr
        with open(smr_folder[index]  + '\\parameters\\completed_trials.pkl', 'rb') as file:
            completed_trials[index]  = pickle.load(file)
        if 'tun' in current_smr:
            with open(smr_folder[index]  + '\\parameters\\trial_parameters.pkl', 'rb') as file:
                trial_parameters[index]  = pickle.load(file)
        spike_info[index] = spikes.get_spike_info(ks_directory, sync_info_directory[index])
        spike_info[index] = ks_curation.get_cluster_templates(spike_info[index])
        if 'imgs' in current_smr:
            with open(smr_folder[index] + '\\parameters\\frame_info.pkl', 'rb') as file:
                frame_info[index]  = pickle.load(file)
    return 

def get_trial_parameters(textmark_data):
    # Input: textmark information from SMR file
    #   textmark_data.markers : text informaiton from each text box
    #   textmark_data.times : timestamp of text box
    # Output: 
    #   trial_parameters : stimulus parameter and trial condition informaiton for each trial
    #                   ['timestamps']
    #                   ['condition'] (e.g., Ori for orientation tuning, Con for contrast response function)
    #                   various parameters. For example ['ORI'] = orientation in degrees, ['CON'] = % contrast

    # called by CED.load.load_file

    trial_condition = []
    condition_time = []
    condition_marker = []
    trial_pattern = r'([A-Z]+),((?:[-*\d+.\d+]+)(?:,[-*\d+.\d+]+)*)?'
    trial_parameters = {}
    trial_parameters['timestamps'] = []
    trial_parameters['condition'] =[]
    
    for text_index, current_text in enumerate(textmark_data.markers):
        match = re.search(r'Condition: (?P<condition>[^,]+)', current_text)
        if match:
            trial_condition.append(match.group('condition'))
            condition_time.append(textmark_data.times[text_index])
            condition_marker.append(current_text)
            # get the parameters for the current trial
            matches = re.findall(trial_pattern, current_text)
            for parameter_info in matches:
                if parameter_info[0] in trial_parameters:
                    trial_parameters[parameter_info[0]].append(parameter_info[1])
                else:
                    trial_parameters[parameter_info[0]] = [parameter_info[1]]
            trial_parameters['timestamps'].append(textmark_data.times[text_index])
            trial_parameters['condition'].append(match.group('condition'))
    trial_parameters['timestamps'] = np.array(trial_parameters['timestamps'])


    return trial_parameters

def get_tuning_trials(trial_parameters, condition, iv_type, completed_trials):
    """get the trials for a specific type of tuning curve (for baker's data only)

    Args:
        trial_parameters (dictionary): stimulus parameters for each trial
        condition (str): condition from baker's file (e.g., Ori)
        iv_type (str): parameter type (e.g., ORI)
        completed_trials (numpy matrix): trial information for completed trials

    Returns:
        tuning_trials (numpy) : index for all trials for current tuning curve
        triggers (numpy) : stim on/off trigger times
        iv_list (numpy) : iv value for each trial
    """
    
    # find the trials that belong to the tuning function
    
    tuning_trials = [index for index, current_condition in enumerate(trial_parameters['condition']) if current_condition == condition]
    tuning_trials = np.array(tuning_trials)
    print(f"# of {condition} trials = {len(tuning_trials)}")
    # get the stimulus triggers 
    triggers = completed_trials[tuning_trials,2:4]
    
    # list of all the IVs presented (length = number of trials)
    
    # iv_list = np.array([float(trial_parameters[iv_type][index]) for index in tuning_trials])
    iv_list = [trial_parameters[iv_type][index] for index in tuning_trials]
    tf_list = np.array([trial_parameters['TF'][index] for index in tuning_trials])
    
    result_list = []
    for iv in iv_list:
        numbers_as_strings = iv.split(',')
        result_array = np.array(numbers_as_strings, dtype=float)
        result_list.append(result_array)
    final_array = np.array(result_list)

    unique_iv_values, indices = np.unique(final_array, axis=0, return_inverse=True)

    iv_list = np.arange(unique_iv_values.shape[0])[indices]
    
    return tuning_trials, triggers, iv_list, unique_iv_values, tf_list

def quick_tuning(spike_times, triggers, iv, params, tf_list):
    cycle_bin_count = 16
    unique_iv = np.unique(iv)
    triggers[:,1] = triggers[:,0]+params['duration']
    hist_x = np.arange(-1*params['pre_window'], params['duration'], params['bin_size'])
    z = hist_x>=0
    iv_count = unique_iv.shape[0]
    trial_count = triggers.shape[0]
    cycle_x = np.linspace(0,1, cycle_bin_count)
    hist_y = np.full((iv_count,hist_x.shape[0]-1), np.nan)
    cycle_y = np.full((iv_count,cycle_bin_count-1), np.nan)
    mean_firing_rate = np.full((iv_count,),np.nan)
    f1 = np.full((iv_count,),np.nan)
    
    for iv_index in range(iv_count):
        current_trials = iv == unique_iv[iv_index]
      
        current_tf = tf_list[current_trials]
        current_trials = [index for index, iv in enumerate(current_trials) if iv]
        current_hist = np.full((len(current_trials), hist_x.shape[0]-1), np.nan)
        
        cycle_hist = np.full((len(current_trials), cycle_bin_count-1), np.nan)
        for trial_index, trial in enumerate(current_trials):
            tf = current_tf[trial_index].astype(int)
       
            cycle_duration = 1/tf
            
            current_spikes = (spike_times >= triggers[trial, 0] - params['pre_window']) & (spike_times<= triggers[trial,1] )
            current_spikes = spike_times[current_spikes] - triggers[trial,0]
            if tf > 0:
                mod_spikes = np.mod(current_spikes, cycle_duration)
                cycle_hist[trial_index], bins = np.histogram(mod_spikes, bins=cycle_x*cycle_duration)
            current_hist[trial_index,:], bins = np.histogram(current_spikes, bins=hist_x)
        hist_y[iv_index,:] = np.mean(current_hist, axis = 0)
        cycle_y[iv_index,:] = np.mean(cycle_hist, axis = 0)
        ps = np.abs(np.fft.fft(cycle_y[iv_index,:])) 
        f1[iv_index] = 2*ps[1]
        
        mean_firing_rate[iv_index] = np.mean(cycle_hist)
    
    return f1, mean_firing_rate, hist_x, hist_y, unique_iv
def calculate(file_info,condition, IV, tuning_params):
    # calculate a specified tuning function
    #input:
    #   file_info: dictionary with info to load experiment data
    #   condition : refers to trial_parameters['condition']. 
    #   IV : refers to keys of trial_parameters:
    #   (example: condition == 'Con' and IV == "CON" to calculate contrast response function)

    # load trial info triggers from python. 

    completed_trials, trial_parameters, sync_info = load_trial_information(file_info)

    # load spike data
    
    spike_times, spike_clusters = spikes.get_synced_spike_times(file_info['kilosort_directory'], sync_info)

    # get only the spikes that belong to current cluster
    
    current_spikes = spike_clusters == file_info['cluster_number']
    percent_cluster = np.sum(current_spikes) / current_spikes.shape[0]
    spike_times =  spike_times[current_spikes]
    spike_clusters = spike_clusters[current_spikes]
   
    print(f"spike cluster {file_info['cluster_number']} is {percent_cluster} of the total spike count ({np.sum(current_spikes)} out of {current_spikes.shape[0]})")
    # edges for PSTH
    trial_x = np.arange(-tuning_params['pre_window'], tuning_params['max_trial_duration'], tuning_params['bin_size'] )

    # find the trials that belong to the tuning function
    tuning_trials = [index for index, current_condition in enumerate(trial_parameters['condition']) if current_condition == condition]
    print(f"# of {condition} trials = {len(tuning_trials)}")
    # get the stimulus triggers 
    triggers = completed_trials[tuning_trials,2:4]

    # list of all the IVs presented (length = number of trials)
    iv_list = np.array([float(trial_parameters[IV][index]) for index in tuning_trials])

    # list of the unique IVs that were presented
    unique_iv = np.unique(iv_list)
    
    # number of IVs presented
    iv_count = unique_iv.shape[0]

    # number of total trials in the contrast response functions
    trial_count = triggers.shape[0]

    print(f"# of tuning trials = {trial_count}")
    print(f"# of {IV} =  {iv_count}") 

    trial_spikes = get_trial_spikes(spike_times, triggers, tuning_params)


    TuningFunction ={}
    # hist_y = psth by trial
    # Hist_Y = psth by IV
    # hist_x = psth bins
    # e = standard error by IV
    TuningFunction['hist_y'], TuningFunction['Hist_Y'], TuningFunction['hist_x'], TuningFunction['e'] = \
        calculate_trial_psth(triggers, trial_spikes, iv_list, tuning_params)
    
    

    # calculate F0 and F1

    TF = determine_TF(trial_parameters, tuning_trials)
    
    if np.unique(TF).shape[0] == 1:
        tuning_params['TF'] = TF[0]
    else:
        raise Exception('add code for TF tuning')
    
    TuningFunction['cycle_hist'], TuningFunction['Cycle_Hist'], TuningFunction['bins'], TuningFunction['e'] = \
        calculate_cycle_hist(triggers, spike_times, iv_list, tuning_params)
    
    TuningFunction['f0'], TuningFunction['f1'], TuningFunction['F0'], TuningFunction['F1'], \
        TuningFunction['F0_e'], TuningFunction['F1_e'] = firing_rate(TuningFunction['cycle_hist'], iv_list)
    
    

    TuningFunction['IV'] = unique_iv
    TuningFunction['iv'] = iv_list
    return TuningFunction

def firing_rate(cycle_hist, IV):
    ps = np.abs(np.fft.fft(cycle_hist, axis=1)) / cycle_hist.shape[1]
    f0 = ps[:, 0]
    f0 = f0[:, np.newaxis]
    F0, F0_e = sort_by_iv(f0, IV)
    f1 = 2*ps[:, 1]
    f1 = f1[:, np.newaxis]
    F1, F1_e = sort_by_iv(f1, IV)
    return f0, f1, F0, F1, F0_e, F1_e


def determine_TF(trial_parameters, trial_list):
   
    TF = np.array([float(trial_parameters['TF'][index]) for index in trial_list])
    
    return TF

def calculate_cycle_hist(stim_times, spike_times, IV, tuning_param):
    # create copy of stim time for modification to make cycle hists
    cycle_duration = 1 / tuning_param['TF']
    cycle_stim_times = stim_times.copy()
    # plt.plot(spike_times[:, 0])
    # plt.show()
    #pdb.set_trace()
    # print('should not get here')

    # divide spikes into trials and reset spike times relative to stim onset
    # change trigger times so that the trial ends at a multiple of a full cycle
    cycle_stim_times = cycle_match(cycle_stim_times, cycle_duration)
    tuning_param['pre_window'] = 0
    cycle_spikes = get_trial_spikes(spike_times, cycle_stim_times, tuning_param)

    start = 0
    stop = cycle_duration
    step_number = tuning_param['cycle_bin_number']
    bins = np.linspace(start, stop, step_number)
    bin_size = np.diff(bins)[0]
    cycle_hist = np.full((cycle_stim_times.shape[0], bins.shape[0] - 1), np.nan, dtype=float)
    trial_duration = np.diff(cycle_stim_times)

    trial_cycle_number = trial_duration / cycle_duration

    for trial_index in range(cycle_stim_times.shape[0]):
        current_spikes = np.where((cycle_spikes[:, 1] == trial_index) & (cycle_spikes[:, 1] > 0))[0]
        current_spikes = cycle_spikes[(current_spikes), 2]
        current_modulus = np.mod(current_spikes, cycle_duration)
        cycle_hist[trial_index, :], x = psth(current_modulus, bins)
        cycle_hist[trial_index, :] = cycle_hist[trial_index, :] / bin_size / trial_cycle_number[trial_index]
    Cycle_Hist, e = sort_by_iv(cycle_hist, IV)

    return cycle_hist, Cycle_Hist, bins, e

def cycle_match(stim_times, cycle_duration):
    # Calculate the adjusted end times to make the trial duration a multiple of cycle_duration
    trial_durations = stim_times[:, 1] - stim_times[:, 0]
    # round to the nearest hundreths place that small rounding errors don't cost a full cycle
    trial_durations = np.ceil(trial_durations * 100) / 100
    adjusted_end_times = np.floor(trial_durations / cycle_duration) * cycle_duration + stim_times[:, 0]

    # Update the end times in the stim_times matrix
    stim_times[:, 1] = adjusted_end_times
    return stim_times

def calculate_trial_psth(stim_times, trial_spikes, IV, tuning_param):
    #input:
    # stim_times: Nx2 (N = trial number) [:,0] = stim onset, [:,1] = stim offset
    # trial spikes list of spikes with additional information [:,0] = (yes trial_spike == 1, no == 0)
    #       [:,1] = trial number, [:,2] =  trial time (onset subtracted)
    # IV : list of the values of IV for each trial, 
    # tuning_param: parameters to calcualte PSTH
    start = -tuning_param['pre_window']
    stop = tuning_param['max_trial_duration']
    step = tuning_param['bin_size']
    bins = np.arange(start, stop, step)
    hist_y = np.full((stim_times.shape[0], bins.shape[0] - 1), np.nan, dtype=float)
    for trial_index in range(stim_times.shape[0]):
        current_spikes = trial_spikes[:, 1] == trial_index
        current_spikes = trial_spikes[(current_spikes), 2]
        hist_y[trial_index, :], hist_x = psth(current_spikes, bins)
        hist_y[trial_index, :] = hist_y[trial_index,:] / step

    Hist_Y, e = sort_by_iv(hist_y, IV)
    return hist_y, Hist_Y, hist_x, e

def load_trial_information(file_info):


    smr_folder = file_info['base_python_directory'] + file_info['experiment_date'] + "\\" + \
        file_info['experiment_name'] +  "\\" + file_info['smr_filename'] 
    
    # load completed_trials
    with open(smr_folder + '\\parameters\\completed_trials.pkl', 'rb') as file:
        completed_trials  = pickle.load(file)

    # load SMR trial parameters from python. 
    with open(smr_folder + '\\parameters\\trial_parameters.pkl', 'rb') as file:
        trial_parameters  = pickle.load(file)

    # load sync info
    with open(smr_folder + '\\sync_info\\sync_info.pkl', 'rb') as file:
        sync_info  = pickle.load(file)
    
    return completed_trials, trial_parameters, sync_info


def get_trial_spikes(spike_times, stim_times, params):
    # include pre_window
    pre_window = stim_times[:, 0] - params['pre_window']
    trial_spikes = np.full((spike_times.shape[0], 3), -1, dtype=float)
    
    trial_duration = np.diff(stim_times)
    # decrease trial duration for all trials over max allowed
    long_trials = trial_duration > params['max_trial_duration']
    stim_times[(long_trials[:, 0]), 1] = stim_times[(long_trials[:, 0]), 0] + params['max_trial_duration']
    # [:,0] = logical (0 = not a trial spike, 1 = trial_spike)
    # [:,1] =  trial number
    # [:,2] = trial time (relative to stim onset)
    trial_duration = np.diff(stim_times)
    
    for trial_index in range(stim_times.shape[0]):
        current_spikes = np.where((spike_times > pre_window[trial_index]) & (spike_times < stim_times[trial_index, 1]))[
            0]
        if len(current_spikes) > 0:
            trial_spikes[(current_spikes), 0] = 1
            trial_spikes[(current_spikes), 1] = trial_index
            trial_spikes[(current_spikes), 2] = spike_times[(current_spikes)] - stim_times[trial_index, 0]

    ts = trial_spikes[:, 0] == 1
    trial_spikes = trial_spikes[(ts), :]
    return trial_spikes

def psth(spike_times, x):
    hist, bins = np.histogram(spike_times, bins=x)
    return hist, bins




def sort_by_iv(y, IV):
    unique_IV = np.unique(IV)

    # Calculate the mean y for each unique IV value
    Y = np.full((unique_IV.shape[0], y.shape[1]), np.nan, dtype=float)
    e = np.full((unique_IV.shape[0], y.shape[1]), np.nan, dtype=float)

    for iv_index in range(len(unique_IV)):
        # Select the rows from hist_y where IV matches the current iv_value
        current_trials = IV == unique_IV[iv_index]
        #pdb.set_trace()
        # Calculate the mean along the columns (axis=0) for the selected rows
        Y[iv_index, :] = np.mean(y[(current_trials),:], axis=0)
        e[iv_index, :] = np.std(y[(current_trials), :], axis=0) / np.sqrt(np.sum(current_trials))

        #Y[iv_index, :] = gaussian_filter1d(Y[iv_index, :], sigma=2)
    return Y, e

def for_deletion_get_completed_trials(parameters):

    ##

    # determine if this function is actively used (09-26-23)

    ##
    completed_trials = {}
    all_trials = {}
    # find all stimulus presentation periods

    # stim onset -> parameters['stimulus']['levels'] == 1
    # stim offset -> parameters['stimulus']['levels'] == 0
    # initial_stim_on = parameters['stimulus']['levels'] == 1
    # sometimes the stim level is set low at the start of the trial (happens before the ready trigger)
    # Thus we need to find the first high -> low
    initial_stim_on = np.asarray(parameters['stimulus']['levels'] == 1).nonzero()[0]
    if len(initial_stim_on)==0:
        print('there are no stimulus presentations in the current file')
    else:
        initial_stim_on = initial_stim_on[0]
        parameters['stimulus']['levels']  = parameters['stimulus']['levels'][initial_stim_on:]
        parameters['stimulus']['times']  = parameters['stimulus']['times'][initial_stim_on:]
    
    # count the number of stimulus presentations 
    
    if np.mod(parameters['stimulus']['times'].shape[0],2) != 0:
        # sometimes the file is killed while the stimulus is on the screen 
        parameters['stimulus']['levels']  = parameters['stimulus']['levels'][0:-1]
        parameters['stimulus']['times']  = parameters['stimulus']['times'][0:-1]

    # save the stimulus triggers as "on-off" pairs
    stim_count = int(len(parameters['stimulus']['times'])/2)
    all_trials['stimulus'] = np.full((stim_count,4), np.nan)
    all_trials['stimulus'][:,0:2] = np.reshape(parameters['stimulus']['times'], (stim_count,2))
    all_trials['stimulus'][:,2:] = np.reshape(parameters['stimulus']['levels'], (stim_count,2))

    # find trial result markers: text: + == completed, - == aborted

    trial_result = [(index,marker) for index, marker in enumerate(parameters['textmark'].markers) if marker in ['+', '-']]
    trial_result_index, trial_result_marker = zip(*trial_result)
    trial_result_index = np.array(trial_result_index)
    trial_result_times = np.array(parameters['textmark'].times)[trial_result_index]
    trial_result_marker = np.array([marker == '+' for marker in trial_result_marker])
    
    # assign trial results to each stimulus presentation
    all_trials['trial_advance'] = np.full(stim_count, np.nan)
    bins = np.digitize(trial_result_times, all_trials['stimulus'][:,0])
    all_trials['trial_advance'][bins-1] = trial_result_marker
    completed_trials['stimulus'] = all_trials['stimulus'][all_trials['trial_advance'].astype(bool),:]
    return completed_trials

def for_deletion_get_stimulus_parameters(completed_trials, textmark_data):

    # fairly certain this function is not used (09-26-23)

    # create a class object to hold the stimulus parameters for each trial
    stim_param = tuning_parameters()
    
    # get trial result markers
    
    trial_result = [(index,marker) for index, marker in enumerate(textmark_data.markers) if marker in ['+', '-']]
    trial_result_index, trial_result_marker = zip(*trial_result)
    trial_result_index = np.array(trial_result_index)
    trial_result_times = np.array(textmark_data.times)[trial_result_index]
    trial_result_marker = np.array([marker == '+' for marker in trial_result_marker])

    ## continue here. I am trying to assign the parameter information to a given trial. I think I need
    # to combine this into tuning.get_completed_trials()
    errors_ahead = True
    if errors_ahead:
        print('add the code!')
        return
              


    # find the text markers that contain stimulus information
    trial_pattern = r'T,(\d+)'
    matched_params = []
    for current_marker in textmark_data.markers:
        

        match = re.search(trial_pattern, current_marker)
        if match:
            pattern = r'(\w+),([^A-Z]+)' 
            matches = re.findall(pattern, current_marker)
            for match in matches:
                param_name = match[0]
                matched_params.append(param_name)
                
                values = match[1][:-1].split(',')
                if len(values) == 1:
                    values = float(values[0])
                else:
                    values = np.array([float(v) for v in values])

                # Assign the values to the corresponding fields in the tuning_parameters object
                if hasattr(stim_param, param_name):
                    param_values = getattr(stim_param, param_name)
                    param_values.append(values)
                    setattr(stim_param, param_name, param_values)

    matched_params = set(matched_params)
    # convert parameter values into numpy arrays
    for param_name in matched_params:
        if hasattr(stim_param, param_name):
            param_values = getattr(stim_param, param_name)
            param_values = np.stack(param_values) 
            setattr(stim_param, param_name, param_values)


    return stim_param