import numpy as np

def trial_shuffle(iv_list, stim_repeats):

    shuffled_trials = np.array([])

    # Loop 5 times to create 5 shuffled copies
    for _ in range(5):
        # Make a copy of the original array
        temp_array = np.copy(iv_list)
        
        # Shuffle the copy
        np.random.shuffle(temp_array)
        
        # Concatenate the shuffled copy to the result array
        shuffled_trials = np.concatenate((shuffled_trials, temp_array))
    
    return shuffled_trials