import re
import scipy
import os
import numpy as np
import pickle
import OpenEphys.spike2_sync as sync
import CED.ced as ced
import CED.load as load

def get_matching_smr_files(directory, file_root):
    # find all the *.smr(x) files within diectory that contain the file_root:
    # example: file_root = 'tmj_230811' (all data collected from Tomi John on 08-11-2023), 
    # directory  = 'D:\AwakeData\Spike2Files\Tomi'
    files = []
    pattern = re.compile(fr'.*{re.escape(file_root)}.*\.(smr|smrx)$', re.IGNORECASE)

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if pattern.match(filename):
                files.append(os.path.join(root, filename))

    return files

def smr_trigger_info(file_name):

    trigger_count = []

    # load ced library
    ced_lib = ced.createWrapper()
    # make sure all files that have been previously opened, have been properly closed
    ced.closeFile(ced_lib=ced_lib)

    # open SMR file
    fhand = ced.open_file(file_name, ced_lib)
    
    # get channel names to find Frame and INTAN channels
    all_channels = ced.get_channel_info(fhand, ced_lib)

    # get trigger info for Frame Channel
    frame_channel = ced.find_channel(all_channels, 'Frame')
    print(f'frame channel  = {frame_channel}')
    
    frame_info = load.load_level_channel(fhand, frame_channel, ced_lib)

    # get trigger info for Frame Channel
    intan_channel = ced.find_channel(all_channels, 'INTAN-T')
    print(f'intan channel  = {intan_channel}')
    
    intan_info = load.load_level_channel(fhand, intan_channel, ced_lib)


    # close smr file
    close_flg = ced.closeFile(f_handle=fhand, ced_lib=ced_lib)
    
    return frame_info, intan_info

def open_ephys_trigger_info(file_name):

    ttl_info = ttl(file_name+'/')

    
    # count the number of pulses that belong to each smr file (according to OE)
    # ttl_info.file_id == -1 indicates that the pulse was not assigned to a SMR file.
    # this is likely a "heartbeat" when data is not being collected
    # pulse_counts  == number of pulses assigne to each SMR file
    unique_smr_files, pulse_counts = np.unique(ttl_info.file_id, return_counts=True)

    
    
    # remove reference to the unassigned pulses
    pulse_counts = pulse_counts[unique_smr_files!=-1]
    unique_smr_files = unique_smr_files[unique_smr_files!=-1]
    return unique_smr_files, pulse_counts, ttl_info


def get_matching_python_folders(directory, file_root):
    folders = []
    pattern = re.compile(fr'.*{re.escape(file_root)}.*', re.IGNORECASE)

    for root, dirnames, _ in os.walk(directory):
        for dirname in dirnames:
            if pattern.match(dirname):
                folders.append(dirname)
                #folders.append(os.path.join(root, dirname))

    return folders


class ttl:
    def __init__(self, ttl_directory):
        # create a ttl object and extract the INTAN pulse information for
        # ttl_directory (ttl_directory + event_dir)
        event_dir = 'events\\Acquisition_Board-100.Rhythm Data\\TTL\\'
        self.ttl_directory = ttl_directory
        print(f"ttl dir =  {ttl_directory}")
        self.acqu_start_time = np.nan
        self.Fs = np.nan
        self.words  = np.load(ttl_directory + event_dir + 'full_words.npy')
        self.states = np.load(ttl_directory + event_dir + 'states.npy')
        self.timestamps = np.load(ttl_directory + event_dir + 'timestamps.npy')
        self.load_synch()
        self.timestamps = self.timestamps - self.acqu_start_time
        self.timestamps = self.timestamps.reshape(np.int64(self.timestamps.size/2),2)
        self.onset = self.timestamps[:,0]
        self.duration = np.diff(self.timestamps)
        self.trial_start =  np.where((self.duration > .0029) & (self.duration < .0031))[0]
        self.trial_end =  np.where((self.duration > .0059) & (self.duration < .0061))[0]
        
        # Stim frames are ~0.004s and are stimulus onset (currently as of 230915)
        # code changed 230915: from (self.duration > .0031) & (self.duration < .0039) to 
        # (self.duration > .0031) & (self.duration < .0041)
        # self.frame =  np.where((self.duration > .0031) & (self.duration < .0039))[0]
        self.frame =  np.where((self.duration > .0039) & (self.duration < .0041))[0]
        file_start = np.where(self.duration > .0149)[0]
        file_end = np.where((self.duration > .0099) & (self.duration< 0.011))[0]
        self.FileMarkers = np.full(shape=(len(file_start), 2), fill_value=-999, dtype=int)
        self.file_id = np.full(shape=(len(self.duration), 1), fill_value=-1, dtype=int)
        # assign each pulse to a SMR file number. The start of a SMR file is indicated by 
        # a pulse > 0.0149s. The end of a SMR file is indicated by a pulse duration ~ 0.01
        
        if len(file_start) != len(file_end):
            # it is possible that the final file_end pulse is missing. Perhaps the
            # OE GUI was killed before the SMR file? Let's assume this is not a problem (for now) 
            # If this is the case, (for now) let's do nothing but report the condition
            # breakpoint()
            # self.file_id = np.full(shape=(len(self.duration)+1, 1), fill_value=-1, dtype=int)
            # file_end = np.append(file_end, [len(self.duration)+1])
            print(f'Uneven file markers. There are {len(file_start)} file start markers and {len(file_end)} file end \
                  markers')
            

        for file_index in range(len(file_start)):
            if file_index < len(file_end):
                self.FileMarkers[file_index,:] = [file_start[file_index], file_end[file_index]] 
                # assign each ttl pulse to a file
                self.file_id[file_start[file_index]:file_end[file_index]+1,0] = file_index
            else:
                print(f"file_index  {file_index} is greater than the length of file_end")
                temp_end = self.timestamps.shape[0]
                self.FileMarkers[file_index,:] = [file_start[file_index], temp_end] 
                # assign each ttl pulse to a file
                self.file_id[file_start[file_index]:temp_end+1,0] = -999
               

        # for each detected smr file, save a vector of trigger times
        trigger_info =[]
        file_list = np.unique(self.file_id)

        for file_index, current_id in enumerate(file_list):
            if current_id != -1:
                current_triggers = self.file_id == current_id
                trigger_info.append(self.timestamps[current_triggers[:,0],:])

        self.trigger_info = trigger_info
        

    def load_synch(self):
        # load sync message
        sync_file = self.ttl_directory + 'sync_messages.txt'
        with open(sync_file , 'r') as file:
            synch_message = file.read()
            #print(synch_message)
            pattern = r"Start Time for Acquisition Board \(100\) - Rhythm[_\s]Data @ (\d+) Hz: (\d+)"
            match = re.search(pattern, synch_message)
            if match:
                self.Fs = np.int64(match.group(1))
                self.acqu_start_time = np.int64(match.group(2))
                self.acqu_start_time = self.acqu_start_time/ self.Fs
                #print("Frequency:", self.Fs)  
                #print("Start Time:", self.acqu_start_time) 
            else:
                print('no match found in synch file')
        return self
    
def ttl_converstion(smr_trigger_count, ooe_trigger_count):
    
    print(f"OE pulse count = {len(ooe_trigger_count)}")
    print(f"smr pulse count = {len(smr_trigger_count)}")
    coefficients = np.polyfit(smr_trigger_count, ooe_trigger_count, deg=1)
    m = coefficients[0]  # Slope
    b = coefficients[1]  # Intercept
    return m,b

#load matlab data
def load_mat_data(mat_file):
    current_mat_data = scipy.io.loadmat(mat_file)
    trial_info = current_mat_data['TrialInfo']
    intan_info = current_mat_data['IntanInfo']
    return trial_info, intan_info

def intan_pulse_duration(intan_info):

    # intan_info: dictionary with keys: times and levels. 
    # Goal is to join the start (levels = 1) and end of the pulses (levels = 0) for
    # each pulse and determine the : pulse: start_time, end_time, duration, id
    # pulse id: 1 = file start, 2 = trial start, 3 = trial success, 4 = trial reward,
    # 5 = file_end
    # duration code :
        # 3 ms Trial Start
        # 4 ms stimulus on, currently only for free view experiments (i.e., imgs files)
        # 6 ms Trial Success
        # 8 ms Maintain Fixation Success Prior To Secondary Reward in Tuning 
        # 10 ms End of Session
        # 15 ms Box O' Donuts Tuning
        # 20 ms Start of Sampling or Session
        # 25 ms Baker's Dozen Tuning
        # 30 ms Orientation Tuning
        # 35 ms Contrast Tuning
        # 40 ms Area Tuning
        # 45 ms Spatial Frequency Tuning
        # 50 ms Temporal Frequency Tuning
        # 55 ms XY Position Tuning
    #returns:
        # pulse_duration: duration of each intan pulse (s)
        # pulse_id: id of each intan pulse (code defined above)
        # pulse_time: start time of pulse (s)
    

    # make sure the first level change == 1 (low-> high)
    if intan_info['levels'][0] == 0:
        intan_info['levels'] = intan_info['levels'][1:]
        intan_info['times'] =  intan_info['times'][1:]
    #make sure the final change == 0 (high -> low)
    if intan_info['levels'][-1] == 1:
        intan_info['levels'] = intan_info['levels'][0:-1]
        intan_info['times'] =  intan_info['times'][0:-1]
    # now the number of level changes should be even
    if np.mod(len(intan_info['levels']),2) != 0:
        raise Exception("Number of level changes is strange and mysterious!")
    
    # group levels into pulses
    pulse_count = int(len(intan_info['levels'])/2)
    pulses = np.reshape(intan_info['levels'], (pulse_count,2))
    pulse_times = np.reshape(intan_info['times'], (pulse_count,2))
    pulse_duration =  np.diff(pulse_times)
    
    # determine identity of each pulse, based on duration
    pulse_id = np.full((len(pulse_duration),1), np.nan)

    # start of a new smr file == 1
    pulse_id[pulse_duration > 0.0149] = 1
    # start of trial == 2
    # breakpoint()
    pulse_id[(pulse_duration > .0029) & (pulse_duration < .0031)] = 2
    # stimulus onset == 3
    pulse_id[(pulse_duration > .0039) & (pulse_duration < .0041)] =  3
    # trial success ==4
    pulse_id[(pulse_duration > .0059) & (pulse_duration < .0061)] =  4
    # trial reward (reward or no reward timing) == 5
    pulse_id[(pulse_duration > .0079) & (pulse_duration < .0081)] =  5
    # end of SMR file == 6
    pulse_id[(pulse_duration > .0099) & (pulse_duration < .0101)] =  6
    pulse_times = pulse_times[:,0]
    return pulse_duration, pulse_id, pulse_times
    

def pulse_count_match(recording_folders, smr_ttl, python_directory, python_folders):

    # We collect neural data using the OpenEphys GUI. The experiment is controlled by SPike2. Thus
    # all the parameters for data collection are stored in SMR files. Spike2 also sends "INTAN" pulses to OpenEphys
    # that can be used to sync the information in the SMR files (stimulus parameters, eye signal, etc) with the OpenEhpys data
    #In intan_pulse_duration, we calcualte the duration of each INTAN pulse present in the SMR files and assign them ids. Here we will use
    # this informaiton to sync the SMR and OpenEphys data
    # input:  
    #       recording_folders: list of OpenEphys folders that belong to the current experiment. 
    #           Calculated by get_folders find_recording_folders
    #       smr_ttl: number of ttl pulses in each smr file
    #       python_directory: root directory for python folders 
    #       python_folders: list of python folders (one smr file per folder)

    total_file_count = 0
    index =0
    for folder in recording_folders:
        ttl_info = ttl(folder['file_path']+'/')
        
        # count the number of pulses that belong to each file (according to OE)
        # ttl_info.file_id == -1 indicates that the pulse was not assigned to a SMR file.
        # this is likely a "heartbeat" when data is not being collected
        # pulse_counts  == number of pulses assigne to each SMR file
        unique_oe_files, pulse_counts = np.unique(ttl_info.file_id, return_counts=True)

        
        
        # remove reference to the unassigned pulses
        pulse_counts = pulse_counts[unique_oe_files!=-1]
        unique_oe_files = unique_oe_files[unique_oe_files!=-1]
        
        # For each OE file there is 0 or more smr files. 
        # pulse_counts is vector of length N (N == number of SMR files recorded during the current OE file)
        smr_match  = np.where(np.isin(smr_ttl[:,0], pulse_counts))[0]
        
        # smr_match is a vector containing the IDs for the SMR files that may be contained within the current OE file, based
        # on pulse count
        
        for match in smr_match:
            print(f"current match index = {match}")
            match_index = np.where(np.isin(pulse_counts, smr_ttl[match,0]))[0]
            if len(match_index) == 1:
                smr_ttl[match,2] = unique_oe_files[match_index]
                smr_ttl[match,1] = index            

                # load pulse info for SMR file that matches the current segment of the current OE file
                python_parameter_path = python_directory + python_folders[match] + '/parameters.pkl'
                with open(python_parameter_path, 'rb') as file:
                    # Unpickle the data from the file
                    python_parameters = pickle.load(file)
                
                #calculate pulse information for current smr file
                pulse_duration, pulse_id, pulse_time = sync.intan_pulse_duration(python_parameters['intan'])
                frame_duration, frame_id, frame_time = sync.intan_pulse_duration(python_parameters['frame'])
                
                # convert smr times (saved in python) to OE time and save to file
                # current_ttl is a vector of timestamps for the pulses that belong to a single SMR file
                current_ttl = ttl_info.file_id == unique_oe_files[match_index]
                current_ttl = ttl_info.timestamps[(current_ttl[:,0]),0]
                breakpoint()
                current_smr_pulses = smr_pulse_info[int(match)]
                
                m,b = ttl_converstion(current_smr_pulses['times'], current_ttl)
                if python_folders[match].find('imgs')!= -1:
                    
                    current_frames = current_smr_pulses['ids'] == 3
                    breakpoint()
                    frame_info = intan_info[(current_frames),:]
                    frame_trials = frame_info[:,2]
                    frame_trials = frame_trials.astype(int)
                    frame_trial_outcome = trial_info[frame_trials-1,5]
                    frame_info = frame_info[(frame_trial_outcome==1),:]
                    stim_triggers = frame_info[:,0:2]
                    stim_triggers = m*stim_triggers + b
                    image_id = frame_info[:,4]
                    stim_info =np.column_stack((stim_triggers, image_id))
                    
                else:
                    valid_trials = trial_info[:,5] == 1
                    stim_triggers = trial_info[valid_trials,3:5]
                    stim_triggers = m*stim_triggers + b
                    iv = trial_info[valid_trials,2]
                    stim_info =np.column_stack((stim_triggers, iv))
                    
                # save stim info to file
                python_file_dir = python_directory + 'smr_files\\' + mat_folders[match] + '\\stim_info.npy'
                print(f"saving stim info to {python_file_dir}")
                print(f"trial count = {stim_info.shape[0]}")
                np.save(python_file_dir, stim_info)
                # pdb.set_trace() 
            
            elif len(match_index) > 1:
                smr_ttl[match,2] = 999
                smr_ttl[match,1] = 999
                stim_info = []
            
            
            
        if index == 0:
            all_pulse_counts = pulse_counts
        else:
            all_pulse_counts = np.append(all_pulse_counts, pulse_counts)
        #print(f"all pulse counts = {all_pulse_counts}")
        #print(f"pulse counts for {folder}: {pulse_counts} ")
        #print(f"files for {folder}: {unique_oe_files} ")
        total_file_count = total_file_count + len(unique_oe_files)
        index +=1
        
    print(f"total file count = {total_file_count}")
    print(f"possible matlab files: {len(mat_folders)}")

    is_identical = len(np.unique(all_pulse_counts)) != len(all_pulse_counts)

    if is_identical:
        print("There are identical pulse durations for OE files.")
    else:
        print("There are NO identical pulse durations for OE files. Thus, there is a unique matlab file for each OE file")
        # walk through all the matlab files and determine if it can be matched to the current
        # OE recording
    #     for mat_recording in mat_folders:
    #         print(f" current mat file: {mat_recording}")
    #         mat_file = mat_directory + mat_recording + '\\parameters_python.mat'
    #         trial_info, intan_info = load_mat_data(mat_file)
    #         current_mat_data = matlab_ttl[mat_recording]
    #         current_mat_data = match_pulse_count(current_mat_data, pulse_counts, intan_info)
    return 


def ITI_match(oe_triggers, smr_triggers, smr_files):
    # For a list of oe_triggers (from a single OE file, e.g., Experiment 1, Recording 3), find the mathcing smr files from a list of triggers
    sync_info_list = []
    
    smr_match_index = np.full(len(oe_triggers,), np.nan)
    
    for oe_index, current_oe_triggers in enumerate(oe_triggers):
        oe_recording_duration = current_oe_triggers[-1,0] - current_oe_triggers[0,0]
        smr_recording_duration = np.full(len(smr_triggers,), np.nan)
        for smr_index, current_smr_triggers in enumerate(smr_triggers):
            smr_recording_duration[smr_index] = current_smr_triggers[-1,0] - current_smr_triggers[0,0]
            # print(f'smr recording duration  =  {smr_recording_duration}')
        
        # find the smr file has the most similar recording duration
        smr_match_index[oe_index] =  np.argmin(np.abs(smr_recording_duration - oe_recording_duration)).astype(int)
        
        d = np.abs(smr_recording_duration[int(smr_match_index[oe_index])] - oe_recording_duration)
        current_match_index = smr_match_index[oe_index].astype(int)
        smr_match_trigger_count =  smr_triggers[current_match_index].shape[0]
        oe_match_trigger_count =  current_oe_triggers.shape[0]
        # 
        # sync_info: the best matching smr file, trigger counts for OE and smr (should be identical), 
        # time difference(should be very small), linear fit to translate the OE times into smr times 
        if smr_match_trigger_count == oe_match_trigger_count:
            coeff = ttl_converstion(smr_triggers[current_match_index][:,0], current_oe_triggers[:,0])
            sync_info = {}
            sync_info['smr_match'] = [current_match_index, smr_files[current_match_index]]
            sync_info['coeff'] = coeff
            sync_info['trigger_count'] = [smr_match_trigger_count, oe_match_trigger_count]
            sync_info['recording_duration'] = [smr_recording_duration[current_match_index], oe_recording_duration]
            sync_info_list.append(sync_info)
            print(f" match with {sync_info['smr_match']}")
        else:
            coeff = []
            sync_info_list.append(None)
            print("no match")


        

            
    
    
    return sync_info_list

def extract_experiment_and_recording_numbers(file_directory):
    # The regular expression to match 'experiment' followed by digits and 'recording' followed by digits
    pattern = r'experiment(\d+).*?recording(\d+)'
    
    # Search the pattern in the file_directory string
    match = re.search(pattern, file_directory)
    
    if match:
        # group(1) corresponds to the first group of digits, which is the experiment number
        # group(2) corresponds to the second group of digits, which is the recording number
        experiment_number = int(match.group(1))
        recording_number = int(match.group(2))
        return experiment_number, recording_number
    else:
        return None, None
def sync_spike_times(spike_times, sync_info, sync_info_directory): 
    # convert OpenEphys spike_times to SMR spike times
    #input:
        # spike_times : Spike times from kilosort / Phy that are in OE time
        # sync_info: run sync_template (e.g., D:\AwakeData\Python\230811\python_scripts\data_sync_tmj_230811.py) to get
        # this dictionary that contains coeff = [slope intercept] for the linear conversion of OE times to SMR times
        # sync_info_directory: directory sync_info file (optional) If defined, then sync_info can be None
    # output:
        # SMR spike times

    if sync_info_directory:
        with open(sync_info_directory, 'rb') as file:
            sync_info  = pickle.load(file)

    synced_spike_times = (spike_times - sync_info['coeff'][1] ) / sync_info['coeff'][0]

    return synced_spike_times 