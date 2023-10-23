import numpy as np
import matplotlib.pyplot as plt

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

def cluster_by_channel(spike_info):
    """ create histogram of clusters and spike count by channel 

    Args:
        spike_info (spike_data): 
        
    Returns:
        hist_channel (numpy array): hist of cluters per channel
        best_channel (numpy array): array of best layer for each cluster
        spike_count (numpy array): array of total spike count for each cluster
        channel_depth (numpy array): depth for each channel  
    """
    best_channel = np.full(spike_info.cluster_list.shape[0],np.nan) # list of best layers for each cluster
    spike_count = np.full(spike_info.cluster_list.shape[0],np.nan) # list of spike_count for each cluster
    for index, current_cluster in enumerate(spike_info.cluster_templates):
        best_channel[index] = np.argmax(np.max(np.abs(spike_info.templates[current_cluster.astype(int),:,:]),axis =0 ))
        spike_count[index] = np.sum(spike_info.clusters == index)
        

    layer_x = np.arange(128)
    channel_depth = layer_x * 0.025
    hist_channel, bins = np.histogram(best_channel, bins=layer_x)
    
    
    channel_spike_count = np.full(layer_x.shape[0],np.nan)
    for index, current_layer in enumerate(layer_x):
        current_clusters = best_channel == current_layer
        channel_spike_count[index] = np.sum(spike_count[current_clusters])

    
    
    return hist_channel, channel_spike_count, best_channel, spike_count, channel_depth

def test_fun():
    print("function loaded!")
    
    return


