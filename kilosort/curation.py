import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import tuning.tuning as tuning

__all__ = ['get_cluster_templates','cluster_by_channel']

def get_cluster_templates(spike_info):
    """generate list of clusters and templates used for each cluster

    Args:
        spike_info (spike_data): spike information  

    Returns:
        spike_info 
            spike.class_list (numpy array): list of clusters present in spike data
            spike.cluster_templates (numpy array): template id used for each cluster
    """
    # create histogram of cell count and spike count by depth

    # list of all clusters in dataset
    spike_info.cluster_list = np.unique(spike_info.clusters)

    # list of templates used for each cluster
    spike_info.cluster_templates = np.full(spike_info.cluster_list.shape[0],np.nan) # list of the template used for each cluster
    for index, current_cluster in enumerate( spike_info.cluster_list):
         spike_info.cluster_templates[index] = np.unique(spike_info.spike_templates[spike_info.clusters == current_cluster])[0]

    return  spike_info



def load_file_triggers(ks_directory):
    
    with open(ks_directory + '\\ttl_triggers.pkl', 'rb') as file:
        ttl_triggers  = pickle.load(file)
    
    start_of_file = np.full(len(ttl_triggers),np.nan)
    end_of_file = np.full(len(ttl_triggers),np.nan)
    for index, current_triggers in enumerate(ttl_triggers):
        trigger_duration = np.diff(current_triggers)
        current_start = np.where(trigger_duration > 0.0149)[0]
        current_start = current_start[0]
        start_of_file[index] = current_triggers[current_start,0]
        end_of_file[index] = current_triggers[np.where((trigger_duration > 0.0099) & (trigger_duration < 0.011))[0][0],0]

    return start_of_file, end_of_file

def firing_rate_hist(spike_times):
    
    # set up hist_x
    m = np.array([np.min(spike_times), np.max(spike_times)])
    hist_x = np.arange(m[0],m[1], 10)
    print(f"hist_x goes from {hist_x[0]} to {hist_x[-1]}, length = {hist_x.shape[0]}, steps = 10s")
    hist_y, bins = np.histogram(spike_times, hist_x)
    
    return hist_x, hist_y

def get_cluster_spike_times(spike_info, cluster_id):
    cluster_spike_times = spike_info.times[np.isin(spike_info.clusters, cluster_id)]
    
    return cluster_spike_times

def psth(spike_times, triggers, params):
    hist_x = np.arange(-params['pre_window'], params['duration'], params['bin_size'])
    hist_y = np.full((triggers.shape[0], hist_x.shape[0]-1), np.nan)
    triggers[:,1] = triggers[:,0] + params['duration']
    for trial_index in range(triggers.shape[0]):
        current_spikes = (spike_times >= triggers[trial_index,0] - params['pre_window']) \
            & (spike_times <= triggers[trial_index, 1] )
        current_spikes = spike_times[current_spikes] - triggers[trial_index, 0]
        hist_y[trial_index,:], bins = np.histogram(current_spikes, bins=hist_x)
    
    
    

    
    return hist_x, hist_y

def plot_cluster(cluster_id, completed_trials, spike_info, frame_info, trial_parameters,
                 full_smr_list, start_file, end_file):
    tun_spike_times = get_cluster_spike_times(spike_info[0], cluster_id)
    
    current_cluster = np.isin(spike_info[0].cluster_list, cluster_id)
    current_template = spike_info[0].templates[spike_info[0].cluster_templates[current_cluster].astype(int),:,:]
    plt.plot(np.max(np.abs(current_template),axis =1 ).T)
    best_layer = np.argmax(np.max(np.abs(current_template),axis =1 ))
    print(f"best layer for current cluster is {best_layer}, depth = {best_layer*.025}mm")
    
    # histogram of firing rate over time
    firing_rate_x, firing_rate_y = firing_rate_hist(tun_spike_times)


    fig, axs = plt.subplots(1,1, figsize=(11, 3))
    axs.plot(firing_rate_x[0:-1], firing_rate_y)
    m = np.array([0, 0])
    m[0] = np.min(firing_rate_y)
    m[1] = np.max(firing_rate_y)
    ypos = np.linspace(0,m[1],len(full_smr_list))
    for index, pos in enumerate(start_file):
        axs.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
        axs.text(pos, ypos[index], full_smr_list[index], color='black')
        print(f"add text at ypos = {ypos[index]}, index = {index}")
    for pos in end_file:
        axs.axvline(x=pos, color='blue', linestyle='--', alpha=0.7)
    
    
    
    fig, axs = plt.subplots(2,1)
    # plot all stim PSTH
    params ={}
    params['bin_size'] = 0.005; params['pre_window'] = 0.5; params['duration'] = 1.3
    [hist_x, hist_y] = psth(tun_spike_times, completed_trials[0][:,2:4], params)

    axs[0].bar(hist_x[0:-1], np.mean(hist_y, axis=0), width = params['bin_size'])
    axs[0].set_xlim([-.5, 1])

    # plot all image PSTH
    current_image_spike_times = get_cluster_spike_times(spike_info[1], cluster_id)
    params['bin_size'] = 0.005; params['pre_window'] = 0.1; params['duration'] = 0.15
    [hist_x, hist_y] = psth(current_image_spike_times, frame_info[1]['times'], params)

    axs[1].bar(hist_x[0:-1], np.mean(hist_y, axis=0), width = params['bin_size'])
    axs[1].set_xlim([-.5, 1])


    params['bin_size'] = 0.02; params['pre_window'] = 0.5; params['duration'] = 0.8
    # ori
    ori_trials, ori_triggers, ori_list,unique_ori_values, tf_list =tuning.get_tuning_trials(trial_parameters[0], 'Ori', 'ORI', completed_trials[0])
    ori_f1, ori_firing_rate, ori_hist_x, ori_hist_y, unique_ori = tuning.quick_tuning(tun_spike_times, ori_triggers, ori_list, params, tf_list)

    # con
    con_trials, con_triggers, con_list,unique_con_values, tf_list =tuning.get_tuning_trials(trial_parameters[0], 'Con', 'CON', completed_trials[0])
    con_f1,con_firing_rate, con_hist_x, con_hist_y, unique_con = tuning.quick_tuning(tun_spike_times, con_triggers, con_list, params, tf_list)

    # sf
    sf_trials, sf_triggers, sf_list, unique_sf_values, tf_list =tuning.get_tuning_trials(trial_parameters[0], 'SF', 'SF', completed_trials[0])
    sf_f1, sf_firing_rate, sf_hist_x, sf_hist_y, unique_sf = tuning.quick_tuning(tun_spike_times, sf_triggers, sf_list, params, tf_list)

    # area
    area_trials, area_triggers, area_list,unique_area_values, tf_list =tuning.get_tuning_trials(trial_parameters[0], 'Area', 'WH', completed_trials[0])
    unique_area_values = unique_area_values[:,0]

    area_f1, area_firing_rate, area_hist_x, area_hist_y, unique_area = tuning.quick_tuning(tun_spike_times, area_triggers, area_list, params, tf_list)



    fig, axs = plt.subplots(4,2)  
    plt.subplots_adjust(left=0, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=1)
    pref_ori = np.argmax(ori_firing_rate)
    axs[0,0].plot(unique_ori_values, ori_firing_rate/np.max(ori_firing_rate))
    axs[0,0].plot(unique_ori_values, ori_f1/np.max(ori_f1))
    axs[0,0].set_title('Ori')
    axs[0,1].plot(ori_hist_x[0:-1], ori_hist_y[pref_ori,:])

    axs[1,0].plot(unique_con_values, con_firing_rate)
    axs[1,0].set_title('Con')
    axs[1,1].plot(sf_hist_x[0:-1], sf_hist_y[-1,:])

    pref_sf = np.argmax(sf_firing_rate)
    axs[2,0].plot(unique_sf_values, sf_firing_rate)
    axs[2,0].set_title('SF')
    axs[2,1].plot(sf_hist_x[0:-1], sf_hist_y[pref_sf])


    pref_area = np.argmax(area_firing_rate)
    axs[3,0].plot(unique_area_values, area_firing_rate,'-+')
    axs[3,0].set_title('Area')
    axs[3,1].plot(area_hist_x[0:-1], area_hist_y[pref_area])
    return None
    


