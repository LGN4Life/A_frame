import time
import numpy as np


def load_recording(session, params):
    bins_loaded = 0
    start_index = 1
    end_index = params['sample_per']
    for node_index in range(len(session.recordnodes)):
        # iterate through all recordings in the current recordnode
        for rec_index in range(len(session.recordnodes[node_index].recordings)):

            recording = session.recordnodes[node_index].recordings[rec_index]
            sample_count = len(recording.continuous[0].sample_numbers)

            chan_num = np.array(recording.continuous[0].samples.shape[1])
            for chan_index in range(chan_num):

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
