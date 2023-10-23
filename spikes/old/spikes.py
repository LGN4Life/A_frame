from crypt import methods
from typing import Self
from scipy import signal
import numpy as np
import OpenEphys.spike2_sync as sync
import sys
import matplotlib.pyplot as plt
import pdb

class SpikeInfo:
    def __init__(self, ks_directory):
        self.ks_directory = ks_directory # path to kilosort data
        self.times = np.load(ks_directory + "spike_times.npy") # vector of spike times, 
                                                               # initially loaded as bins
        self.file_id = None # file of origin for spike information, 
                            # leave empty for now until integrated into mysql
        self.clusters = None # vector of cluster ids, one for each spike time 
        self.cluster_list = None # list of all clusters in dataset (i.e., np.unique(self.cluters))
        self.cluster_templates = None # template index used to create each cluster
        self.spike_templates = None # vector of template ids, one for each spike time
        self.templates = None # (n,m,t) matrix of templates waveforms 
                              # (n = cluter number, m = channel number, t = temporal bins)
        self.params = self.get_params() # params from kilosort, includes sample rate to convert bins to time
        
    def bins_to_secs(self):
        self.times = self.times / self.params['sample_rate']
    
    def get_params(self):
        with open(ks_directory + 'params.py', 'r') as params_file:
            exec(params_file.read(), self.params)
        
            
            
        

def get_spike_info(ks_directory, sync_info_directory):
    """
    input: kilosort directory
    output: spike_data object
    """
    spike_info = spike_data()
    spike_info.times = np.load(ks_directory + "spike_times.npy")
    spike_info.clusters = np.load(ks_directory + "spike_clusters.npy")
    spike_info.spike_templates = np.load(ks_directory + "spike_templates.npy")
    spike_info.templates = np.load(ks_directory + "templates.npy")
    sys.path.append(ks_directory)
    import params
    spike_info.times = spike_info.times / params.sample_rate
    spike_info.params = params

    spike_info.times = sync.sync_spike_times(spike_info.times, sync_info = None, sync_info_directory=sync_info_directory)
    return spike_info


def get_synced_spike_times(kilosort_directory, sync_info, sync_info_directory):
    """ get spike OpenEphys spike times from Kilosort and sync to SMR times
        This function will be depracated
    Args:
        kilosort_directory (str): path to kilosort data
        sync_info (dictionary): information to convert OE spike times to SMR spike times
        sync_info_directory (str): path to sync_info, only used if sync_info = None

    Returns:
        _type_: _description_
    """
    raise Exception("This function is repalced by spikes.get_spike_times.")

    spike_times = np.load(kilosort_directory + "spike_times.npy")
    # spikes are loaded as bins, convert to seconds
    spike_times = spike_times / sync_info['Fs']
    spike_clusters = np.load(kilosort_directory + "spike_clusters.npy")


    synced_spike_times = sync.sync_spike_times(spike_times, sync_info, sync_info_directory=sync_info_directory)


    return synced_spike_times, spike_clusters

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

