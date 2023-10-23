
import numpy as np
def load_spike_times(phy_dir, fs):
    spike_times_file = 'spike_times.npy'
    spike_clusters_file = 'spike_clusters.npy'
    spike_times = np.load(phy_dir + spike_times_file)
    spike_times = spike_times / fs
    spike_clusters = np.load(phy_dir + spike_clusters_file)
    return spike_times, spike_clusters
