

import numpy as np
from quantities import Hz, s, uV, mm
from elephant.current_source_density import estimate_csd
from neo.core import AnalogSignal
import matplotlib.pyplot as plt



def calculate_csd(erp_mat, electrode_distance, save_path):
    # transpose the erp_mat for compatibility
    erp_mat = erp_mat.T
    N = erp_mat.shape[0] # number of time samples
    M = erp_mat.shape[1]  # Number of electrodes

    # Create a neo.AnalogSignal object from the numpy array
    lfp_signal = AnalogSignal(erp_mat * uV, t_start=0 * s, sampling_rate=500 * Hz)
    coordinates = electrode_distance*np.linspace(0,M,M).reshape(M, 1) * mm  # Replace with your actual coordinates

    # caluclate csd
    # Calculate the CSD using Elephant's estimate_csd function
    csd_mat = estimate_csd(lfp_signal, coordinates=coordinates, method='KCSD1D')

    # plot and save figure
    fig, axs = plt.subplots(1,1)
    axs.imshow(csd_mat.T,cmap='jet')
    new_M = csd_mat.annotations['x_coords']
    ytick = np.linspace(0, new_M.shape[0]-1, 12, dtype = int)
    ytick_labels = csd_mat.annotations['x_coords'][ytick]
    ytick_labels = np.around(ytick_labels, 2)
    axs.set_yticks(ytick)
    axs.set_yticklabels(ytick_labels)

    time_bins = csd_mat.magnitude.shape[0]
    xtick = np.linspace(0, time_bins-1, 4, dtype = int)
    axs.set_xticks(xtick)
    bin_size = 0.002
    time = np.arange(0, bin_size*time_bins, bin_size)
    xtick_labels = time[xtick]
    xtick_labels = np.around(xtick_labels, 2)
    axs.set_xticklabels(xtick_labels)
    axs.set_xlabel('time (s)')
    axs.set_ylabel('depth (mm)')
    plt.savefig(save_path)
    plt.show()
    plt.close()

    return csd_mat