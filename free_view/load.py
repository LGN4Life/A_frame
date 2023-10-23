import re
import numpy as np
def get_image_name(textmark):
    # for free view files, extract the image name for each stimulus presentation.
    # create a list of image files that were presented on each trial
    # create a set of unique images that were presented
    # assign each image presentation an index based on the position in the image_set
    # input:
    #   textmarks from SMR file
    # output:
    #   image_name: name of image presented (in order of presentation)
    #   image_set: list of unique images presented
    #   image_id: id for each image presented (id based on order of image_set)

    # Initialize empty lists to store matching indices and file_names
    image_info = {}
    image_info['name'] = []

    image_index = []

    # Define the regex pattern
    # pattern = r"([^\s]+),(-?\d+\.\d+),(-?\d+\.\d+)"
    pattern = r"([^\s]+),(-?\d+\.\d+),(-?\d+\.\d+)(?!.*--serial COM8)"

    # Iterate through the list to find matches and their index positions
    for index, s in enumerate(textmark.markers):
        match = re.search(pattern, s)
        if match:
            image_index.append(index)
            image_info['name'].append(match.group(1))
    
    # set of images that were presented
    image_set = set(image_info['name'])

    image_info['id'] = np.full(len(image_index,), np.nan)
    image_info['time'] = [textmark.times[index] for index in image_index]

    for image_index, image in enumerate(image_set):
        current_match = np.array([index for index, s in enumerate(image_info['name']) if s == image])
        image_info['id'] [current_match] = image_index
    
    image_set =  [image for image in image_set]

    return image_info, image_set


def image_match_frame(frame_info, image_info):
    # we have levels that indicate when images were displayed
    # we also have textmarks that indicate which images were displayed.
    # in principle, the time of the textmark should be nearly identical 
    # to the time of the level change (frame channel in the SMR file)
    # Let's be safe and match the textmarks with the level changes
    frame_count = len(frame_info['times'])
    frame_info['image_id'] = np.full((frame_count,), np.nan)

    for index, current_time in enumerate(image_info['time']):
        d = np.abs(frame_info['times'][:,0] - current_time)
        min_index = np.argmin(d)
        frame_info['image_id'][min_index] =  image_info['id'][index]


    return frame_info

def find_stim_off_frames(frame_info, textmark):
    # for CSD files (from free view) there are frames associated with a "Stimulus off" textmark
    # Let's assign these frames a image_id == -1 
    stim_off_frames =  [index for index, s in enumerate(textmark.markers) if s == 'Stimulus off']
    stim_off_times = [textmark.times[index] for index in stim_off_frames]
    
    for index, current_time in enumerate(stim_off_times):
        d = np.abs(frame_info['times'][:,0] - current_time)
        min_index = np.argmin(d)
        frame_info['image_id'][min_index] =  - 1


    return frame_info


def image_offset_match(image_info, textmark, frame_info):
    # for image series files (from free view) there are "+" textmarks that indicate that the image presentation
    # was successfull and the image is being taken down. Match the "+" with Frame triggers to get the stim offset times

    stim_off_frames =  [index for index, s in enumerate(textmark.markers) if s == '+']
    stim_off_times = [textmark.times[index] for index in stim_off_frames]

    stim_off_level_times = np.full(len(stim_off_frames,),np.nan)

    for index, current_time in enumerate(stim_off_times):
        d = np.abs(frame_info['times'][:,0] - current_time)
        min_index = np.argmin(d)
        stim_off_level_times[index]  = frame_info['times'][min_index,0]

    # assign each stim_off_level to a image presentation
    image_on_times = image_info['time']
    trial_off_index = np.digitize(stim_off_level_times, image_on_times)
    image_count = len(image_info['time'])
    image_times =np.full((image_count,2),np.nan)
    image_times[:,0] =  image_info['time']
    image_times[trial_off_index-1,1] = stim_off_level_times

    image_info['time'] = image_times


    return image_info