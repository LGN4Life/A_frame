import xml.etree.ElementTree as ET
import csv
# Namespace prefix
namespace = "{nxmlvosobservations}"

# Parse the XML file into an ElementTree
tree = ET.parse("M-I 01.09.18.PM2.45610.SJM (bjmccowan@ucdavis.edu).odx")

# Find all "event" elements indicated by <OBS_EVENT>
event_elements = tree.findall(".//{}OBS_EVENT".format(namespace))

# Create a CSV file to write the information
with open("event_information.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write the column names in the first row of the CSV
    writer.writerow(["Observation Number", "StartTimeStamp", "StopTimeStamp", "Behavior", "Modifiers"])

    current_observation_number = 0

    # Process each event element
    for event_element in event_elements:
        # Get the OBS_EVENT_STATE
        event_state = event_element.find("{}OBS_EVENT_STATE".format(namespace)).text

        # Start of an observation (OBS_EVENT_STATE = 1)
        if event_state == "1":
            current_observation_number += 1  # Increment current observation number
            d_current_start = current_observation_number
            d_observation_number = current_observation_number  # Assign current number as observation number for this event

            # Get the timestamp from OBS_EVENT_TIMESTAMP element
            d_start_timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text

            # Get the OBS_EVENT_BEHAVIOR NAME
            d_behavior_name = event_element.find("{}OBS_EVENT_BEHAVIOR".format(namespace)).attrib["NAME"]
            # Get all OBS_EVENT_BEHAVIOR_MODIFIER elements within the event
            d_behavior_modifiers = event_element.findall(".//{}OBS_EVENT_BEHAVIOR_MODIFIER".format(namespace))

            # Create a list to store all behavior modifiers
            d_modifiers_list = []
            for modifier in d_behavior_modifiers:
                modifier_class = modifier.attrib["CLASS"]
                modifier_value = modifier.text
                d_modifiers_list.append((modifier_class, modifier_value))
        # End of an observation (OBS_EVENT_STATE = 2)
        elif event_state == "2":
            d_observation_number = d_current_start   # Assign the previous number as observation number for this event

            d_stop_timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text

        # OBS_EVENT_STATE = 3 events will be labeled with the current observation number
        else:
            current_observation_number += 1  # Increment current observation number
            e_observation_number = current_observation_number

            # Get the timestamp from OBS_EVENT_TIMESTAMP element
            e_start_timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text
            e_stop_timestamp = e_start_timestamp
            # Get the OBS_EVENT_BEHAVIOR NAME
            e_behavior_name = event_element.find("{}OBS_EVENT_BEHAVIOR".format(namespace)).attrib["NAME"]
            # Get all OBS_EVENT_BEHAVIOR_MODIFIER elements within the event
            e_behavior_modifiers = event_element.findall(".//{}OBS_EVENT_BEHAVIOR_MODIFIER".format(namespace))

            # Create a list to store all behavior modifiers
            e_modifiers_list = []
            for modifier in e_behavior_modifiers:
                modifier_class = modifier.attrib["CLASS"]
                modifier_value = modifier.text
                e_modifiers_list.append((modifier_class, modifier_value))

        if event_state == "2":
            # Create a list to store all behavior modifiers as strings
            modifiers_list = [f"{modifier.attrib['CLASS']}:{modifier.text}" for modifier in d_behavior_modifiers]
            writer.writerow([d_observation_number,
                             d_start_timestamp, d_stop_timestamp, d_behavior_name, ", ".join(modifiers_list)])
        elif event_state == "3":
            # Create a list to store all behavior modifiers as strings
            modifiers_list = [f"{modifier.attrib['CLASS']}:{modifier.text}" for modifier in e_behavior_modifiers]
            writer.writerow([e_observation_number,
                             e_start_timestamp, e_stop_timestamp, e_behavior_name, ", ".join(modifiers_list)])

