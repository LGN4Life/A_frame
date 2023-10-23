import numpy as np
import json
import pdb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def build_tuning_param(pre_window=-0.5, cycle_bin_number=16, bin_size=0.005, max_trial_duration=0.8, tf=None):
    tuning_param = {
        'pre_window': pre_window,
        'cycle_bin_number': cycle_bin_number,
        'bin_size': bin_size,
        'max_trial_duration': max_trial_duration,
        'TF': tf
    }
    return tuning_param














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








def baker_sort(file_name):

    # Open the JSON file and load its contents
    with open(file_name, 'r') as file:
        baker_info = json.load(file)

    tuning_curves = [key for key, value in baker_info.items() if np.unique(value).shape[0] > 1]
    if 'ori' in tuning_curves:
        tuning_curves.remove('ori')
        tuning_curves.insert(0, 'ori')

    return baker_info, tuning_curves


def calculate(stim_times, spike_times, trial_spikes, IV, tuning_param):

    tuning_data = {}
    # PSTH
    tuning_data['hist_y'], tuning_data['Hist_Y'], tuning_data['hist_x'], \
    tuning_data['Hist_Y_e'] = calculate_trial_psth(stim_times, trial_spikes, IV, tuning_param)

    # cycle histogram
    tuning_data['cycle_hist'], tuning_data['CycleHist'], tuning_data['cycle_hist_x'], tuning_data['CycleHist_e'] = \
        calculate_cycle_hist(stim_times, spike_times, IV, tuning_param)

    # 7. Response by trial (mean and F1)
    tuning_data['f0'], tuning_data['f1'], tuning_data['F0'], tuning_data['F1'], tuning_data['F0_e'], tuning_data['F1_e'] = \
        firing_rate(tuning_data['cycle_hist'], IV)

    return tuning_data

def baker_trial_exclude(tc, baker_info,tuning_param):
    valid_trials = np.array(range(len(baker_info[tc])))
    if tc == 'ori':
        #pdb.set_trace()
        valid_trials = np.where((baker_info['con'] == np.max(baker_info['con']))
                                & (baker_info['tf'] > np.array(3))
                                & (baker_info['tf'] < np.array(6))
                                & (baker_info['sf'] > np.array(.8))
                                & (baker_info['sf'] < np.array(1.6)))[0]
    elif tc == 'con':
        valid_trials = np.where((baker_info['ori'] == tuning_param['pref_ori'])
                                & (baker_info['tf'] > np.array(3))
                                & (baker_info['tf'] < np.array(6))
                                & (baker_info['sf'] > np.array(.8))
                                & (baker_info['sf'] < np.array(1.6)))[0]
    elif tc == 'sf':
        valid_trials = np.where((baker_info['ori'] == tuning_param['pref_ori'])
                                & (baker_info['tf'] > np.array(3))
                                & (baker_info['tf'] < np.array(6))
                                & (baker_info['con'] == np.max(baker_info['con'])))[0]
    elif tc == 'tf':
        valid_trials = np.where((baker_info['ori'] == tuning_param['pref_ori'])
                                & (baker_info['sf'] > np.array(.8))
                                & (baker_info['sf'] < np.array(1.6))
                                & (baker_info['con'] == np.max(baker_info['con'])))[0]
    else:
        print(f"no criteria set for {tc}. Using all trials")
    print(f" {len(valid_trials)} valid trials out of {len(baker_info[tc])}")
    return valid_trials

