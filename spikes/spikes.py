import numpy as np
import sys
import pickle
import copy

from typing import Self
from scipy import signal
import OpenEphys.spike2_sync as sync

import matplotlib.pyplot as plt
import pdb

    # spike_info.spike_templates = np.load(ks_directory + "spike_templates.npy")
    # spike_info.templates = np.load(ks_directory + "templates.npy")

class SpikeInfo:
    def __init__(self, ks_directory=None):
        if ks_directory:
            self.ks_directory = ks_directory # path to kilosort data
            self.times = np.load(ks_directory + "spike_times.npy") # vector of spike times, 
                                                                # initially loaded as bins
            
            # load kilosort params
            params = self.get_params()
            self.params = params
            self.bins_to_secs()    # convert spike_times from bins to seconds
                                            
            self.file_id = None # file of origin for spike information, 
                                # leave empty for now until integrated into mysql
            self.cluster_ids = np.load(ks_directory + "spike_clusters.npy") # vector of cluster ids, one for each spike time 
            self.cluster_id_list = np.unique(self.cluster_ids) # numpy array of all clusters in dataset 
            
            self.spike_template_ids = np.load(ks_directory + "spike_templates.npy")
            
            # self.templates: (n,m,t) matrix of templates waveforms 
            # (n = cluter number, t = temporal bins, m = channel number)                                                                          # one for each spike time
            self.templates = np.load(ks_directory + "templates.npy") 
            self.cluster_template_ids = np.array([]) # template index used to create each cluster
            self.get_cluster_template_ids()
            self.cluster_best_channel = np.array([]) 
            self.cluster_spike_count = np.array([]) 
            self.get_cluster_stats()
            self.channel_index = np.arange(128)
            self.channel_depth = self.channel_index * 0.025
            self.get_file_triggers()
        else:
            self.times = None
            self.params = None
    def make_tuning_copy(self):
        new_copy = SpikeInfo()
        new_copy.times = copy.deepcopy(self.times)
        new_copy.cluster_ids = copy.deepcopy(self.cluster_ids)
        new_copy.params = self.params.sample_rate
        return new_copy
        
    def get_file_triggers(self):
        with open(self. ks_directory + '\\ttl_triggers.pkl', 'rb') as file:
            ttl_triggers  = pickle.load(file)
        self.start_of_file = np.full(len(ttl_triggers),np.nan)
        self.end_of_file = np.full(len(ttl_triggers),np.nan)

        for index, current_triggers in enumerate(ttl_triggers):
            trigger_duration = np.diff(current_triggers)
            current_start = np.where(trigger_duration > 0.0149)[0]
            if len(self.start_of_file.shape)>1:
                current_start = current_start[0]
            self.start_of_file[index] = current_triggers[current_start[0],0]
            self.end_of_file[index] = current_triggers[np.where((trigger_duration > 0.0099) & (trigger_duration < 0.011))[0][0],0]
    
    def bins_to_secs(self):
        self.times = self.times / self.params.sample_rate
 
    def sync_spike_times(self, sync_info_directory):
            sync_info_directory = sync_info_directory +  "\\sync_info\\sync_info.pkl"
            if sync_info_directory:
                with open(sync_info_directory, 'rb') as file:
                    self.sync_info  = pickle.load(file)
                    
            spike_times = self.times
            spike_times = (spike_times - self.sync_info['coeff'][1] ) / self.sync_info['coeff'][0]
            self.times = spike_times   
    def get_params(self):
        """load kilosort parameters from file

        Returns:
            class object of kilosort parameters
        """
        sys.path.append(self.ks_directory)
        import params
        
        return params
    
    def get_cluster_template_ids(self):
        """generate numpy array of templates ids used for each cluster

        Returns:
                spike.cluster_template_ids (numpy array): template id used for each cluster
        """
        # list of templates used for each cluster
        self.cluster_template_ids = np.full(self.cluster_id_list.shape[0],np.nan) 

        for index, current_cluster in enumerate( self.cluster_id_list):
            current_spikes = self.cluster_ids == current_cluster
            current_templates = self.spike_template_ids[current_spikes]
            self.cluster_template_ids[index] = np.unique(current_templates)[0]
    
    def get_cluster_stats(self):
        # add code to calculate cluster depth. Need to load channel map
        self.cluster_depth = np.full(self.cluster_id_list.shape[0],np.nan)  # np array of depth for each cluster
        self.cluster_best_channel = np.full(self.cluster_id_list.shape[0],np.nan) # np array of best layers for each cluster
        self.cluster_spike_count = np.full(self.cluster_id_list.shape[0],np.nan) # np array of spike_count for each cluster
        for index, current_cluster in enumerate(self.cluster_id_list):
            current_template_id = self.cluster_template_ids[index]
            self.cluster_best_channel[index] = np.argmax(np.max(np.abs(self.templates[current_template_id.astype(int),:,:]),axis =0 ))
            self.cluster_spike_count[index] = np.sum(self.cluster_ids == current_cluster)
        
    def cluster_by_channel(self, axs=None):
        """ create histogram of clusters and spike count by channel 

        Args:
            self (spike_data): 
        
        """
        
        hist_channel, bins = np.histogram(self.cluster_best_channel, bins=self.channel_index)
        bin_size = 0.025
        if not axs:
            print('here')
            fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.5)
        
        axs[0].bar(self.channel_depth[0:-1], hist_channel, width = bin_size)
        
        axs[0].set_xlabel('Channel Depth')
        axs[0].set_ylabel('Cluster Count')
        axs[1].plot(self.channel_depth[0:-1], np.cumsum(hist_channel))
        axs[1].set_xlabel('Channel Depth')
        axs[1].set_ylabel('Cluster Cum Sum')
        
    
    def spike_count_by_channel(self, axs=None):
        channel_spike_count = np.full(self.channel_index.shape[0],0)
        for index, current_layer in enumerate(self.channel_index):
            current_clusters = self.cluster_best_channel == current_layer
            channel_spike_count[index] = np.sum(self.cluster_spike_count[current_clusters])
        bin_size = 0.025
        if not axs:
            print('here')
            fig, axs = plt.subplots(2,1)
        plt.subplots_adjust(hspace=0.5)
        
        axs[0].bar(self.channel_depth, channel_spike_count, width = bin_size)
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Channel Depth')
        axs[0].set_ylabel('Spike Count')
        axs[1].plot(self.channel_depth, np.cumsum(channel_spike_count))
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Channel Depth')
        axs[1].set_ylabel('Spike COunt Cum Sum')
    
    def spike_count_by_time(self, cluster_index = None, bin_size = 10):
        x = np.arange(self.times[0], self.times[-1],bin_size)
        cluster_spikes_logical = np.isin(self.cluster_ids, cluster_index)
        y, edges = np.histogram(self.times[cluster_spikes_logical], x) 
        y = y/bin_size
        fig, axs = plt.subplots(1,1)
        axs.plot(x[0:-1], y)
        axs.set_xlabel('time (s)')
        axs.set_ylabel('firing rate (sp/s)')
        m = np.array([0, 0])
        m[0] = np.min(y)
        m[1] = np.max(y)
        
        ypos = np.linspace(0,m[1],len(self.smr_file_list))
        
        for index, pos in enumerate(self.start_of_file):
            axs.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
            axs.text(pos, ypos[index], self.smr_file_list[index], color='black')
            print(f"add text at ypos = {ypos[index]}, index = {index}")
        for pos in self.end_of_file:
            axs.axvline(x=pos, color='blue', linestyle='--', alpha=0.7)
        
            
            






def correlation(spike_info, params):
    """
    calcualte autocorr and cross corr for one or two spike clusters
    input:
        spike_info: list length 1 or 2 (cluster_1, cluster_2)
    output:
        correlation mat: [0] spike_train_1 autocorr, [1] cross_corr, 
        [2] = spike_train_2 autocorr
    """
    corr_mat = [None]*3
    hist_y = [None]*3
    hist_x = np.arange(0, params['max_time'], params['bin_size'])
    corr_x = signal.correlation_lags(hist_x.shape[0]-1, hist_x.shape[0]-1, mode='same')
    corr_x = corr_x * params['bin_size']
    z = np.where(corr_x == 0)[0]
    full_window = (corr_x >= -1*params['window']) & (corr_x <= params['window'])
    # hist_x = np.arange(-params['window_size'], params['window_size'], params['bin_size'])
    for index, current_cluster in enumerate(spike_info):
        hist_y[index], edges = np.histogram(current_cluster.times, hist_x)
        corr_mat[index] = signal.correlate(hist_y[index], hist_y[index], mode='same', method='fft')
        corr_mat[index][z] = np.mean(corr_mat[index][[z-1,z+1]]) 
        corr_mat[index] = corr_mat[index][full_window]


    
    corr_x = corr_x[full_window]

    return corr_mat, corr_x

def get_cluster_spikes(spike_info, cluster_id):
    """""
    for a list of cluster ids, extract the matching spike times
    input:
        spike_info: spikes.spike_data object
        cluster_id: list of cluster ids. Extact a vector of spike times for each id
    output:
        cluster_spike_times: list of spike_times vectors
    """""

    cluster_info = [spike_data()]* len(cluster_id)
    for index, current_cluster in enumerate(cluster_id):
        current_spikes = spike_info.clusters == current_cluster
        cluster_info[index].times = spike_info.times[current_spikes]
        cluster_info[index].clusters = spike_info.clusters[current_spikes]
    
    return cluster_info

