import time
import numpy as np


def load_recording(recording, params):
    bins_loaded = 0
    start_index = 1
    end_index = params['sample_per']
    







                while bins_loaded < sample_count:

                    if params['scale_lfp']:
                        time_1 = time.time()
                        data = recording.continuous[0].get_samples(start_sample_index=start_index,
                                                                   end_sample_index=sample_count,
                                                                   selected_channels=np.array(chan_index))
                        time_2 = time.time()
                        print(time_2 - time_1)
                        bins_loaded += sample_per
                    data = recording.continuous[0].samples
                    breakpoint()

    return
