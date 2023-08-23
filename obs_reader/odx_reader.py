import xml.etree.ElementTree as ET
import csv
import os

class event_info():
    def __init__(self, **kwargs):
        self.obs_number = 0
        self.start_time = -9999
        self.stop_time = -9999
        self.behavior = 'unk'
        self.modifiers = []
    def get(self, event_element, n,  namespace="{nxmlvosobservations}"):

        self.obs_number = n  # Assign current number as observation number for this event

        # Get the timestamp from OBS_EVENT_TIMESTAMP element
        self.start_time = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text
        self.start_time = self.format_time_stamps(self.start_time)
        # Get the OBS_EVENT_BEHAVIOR NAME
        self.behavior = event_element.find("{}OBS_EVENT_BEHAVIOR".format(namespace)).attrib["NAME"]
        # Get all OBS_EVENT_BEHAVIOR_MODIFIER elements within the event
        behavior_modifiers = event_element.findall(".//{}OBS_EVENT_BEHAVIOR_MODIFIER".format(namespace))

        # Create a list to store all behavior modifiers
        self.modifiers = []
        for modifier in behavior_modifiers:
            modifier_class = modifier.attrib["CLASS"]
            modifier_value = modifier.text
            self.modifiers.append((modifier_class, modifier_value))
        return self

    def format_time_stamps(self, time_stamp):
        time_parts = time_stamp.split()[1].split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = float(time_parts[2])
        time_stamp = (f"{hour}:{minute}:{second:.1f}")
        # breakpoint()
        return time_stamp



def get_obs_data(file_list, file_directory, namespace="{nxmlvosobservations}"):

        # file_info will either be a file name or file directory. Determine which one it is


        for file_name in file_list:
            current_path = os.path.join(file_directory, file_name)

            date, id = get_header_info(file_name)


            # Parse the current odx file (XML file) into an ElementTree
            tree = ET.parse(current_path)

            # get header information from name of file and from within the file
            header_info = tree.findall(".//{}OBS_OBSERVATIONS".format(namespace))
            extracted_file_name = header_info[0].find("{}OBS_OBSERVATION".format(namespace)).attrib["NAME"]
            e_date, e_id = get_header_info(extracted_file_name)
            header_info = [date, e_date, id, e_id]
            # Find all "event" elements indicated by <OBS_EVENT>
            event_elements = tree.findall(".//{}OBS_EVENT".format(namespace))

            event_list = parse_event_info(event_elements, namespace)

            output_directory = os.path.join(file_directory, 'output', file_name[:-4] +'.csv')
            to_csv(event_list, output_directory, header_info)



def file_check(file_info):
    # Check if the path is a file
    if os.path.isfile(file_info):
        file_list = file_info
        file_directory = os.path.dirname(file_info)

    # Check if the path is a directory
    elif os.path.isdir(file_info):
        file_list = []
        file_directory = file_info
        for filename in os.listdir(file_info):
            if filename.endswith(".odx"):
                file_list.append(filename)
    else:
        return f"Error: Path {file_info} does not exist."

    # check for output folder . If it does not exist, create it.
    outputpath = os.path.join(file_directory, 'output')
    if not os.path.isdir(outputpath):
        os.makedirs(outputpath)

    return file_list, file_directory

def get_header_info(file_name):

    # header info from file-name
    if 'M-I' in file_name:
        # Extracting the date (month.day.year)
        date_part = file_name.split()[1]  # Split the string by whitespace and take the second part
        date_parts = date_part.split('.')  # Split the date part by periods
        month, day, year = date_parts[:3]  # Extract month, day, and year

        # Extracting the ID
        id_part = file_name.split()[1]  # Split the string by whitespace and take the third part
        id_value = id_part.split('.')[4]  # Split the ID part by periods and take the second part

        # Displaying the extracted values
        date =  f"{month}.{day}.{year}"
        id =  f"{id_value}"
    else:
        date = ''
        id = ''

    return date, id

def parse_event_info(event_elements, namespace):
    # Process each event element
    current_observation_number = 0
    event_list = []
    for event_element in event_elements:
        # Get the OBS_EVENT_STATE
        event_state = event_element.find("{}OBS_EVENT_STATE".format(namespace)).text

        # Start of an observation (OBS_EVENT_STATE = 1)
        if event_state == "1":
            current_duration_event = event_info()
            current_observation_number += 1  # Increment current observation number
            current_duration_event.get(event_element, current_observation_number, namespace)

        # End of an observation (OBS_EVENT_STATE = 2)
        elif event_state == "2":

            current_duration_event.stop_time = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text
            current_duration_event.stop_time = current_duration_event.format_time_stamps\
                (current_duration_event.stop_time)
            event_list.append(current_duration_event)
        # OBS_EVENT_STATE = 3 events will be labeled with the current observation number
        elif  event_state == "3":
            current_single_event = event_info()
            current_observation_number += 1  # Increment current observation number
            current_single_event.get(event_element, current_observation_number, namespace)
            current_single_event.stop_time = current_single_event.start_time
            event_list.append(current_single_event)
        else:
            raise ValueError(f"invalid event marker detected {event_state} .")

    return event_list

def to_csv(event_list, outfile, header_info):
    # Create a CSV file to write the information
    with open(outfile, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['date: ' + header_info[0], 'e_date: ' + header_info[1],
                        'id: '+ header_info[2], 'e_id: ' + header_info[3]])
       # writer.writerow(f"date: {header_info[0]}")
        # breakpoint()
        # Write the column names in the first row of the CSV
        writer.writerow(["Observation Number", "StartTimeStamp", "StopTimeStamp", "Behavior", "Modifiers"])

        for event in event_list:
            writer.writerow([event.obs_number, event.start_time, event.stop_time,
                             event.behavior, event.modifiers])
    return






