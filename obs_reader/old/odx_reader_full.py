import xml.etree.ElementTree as ET

# Namespace prefix
namespace = "{nxmlvosobservations}"

# Parse the XML file into an ElementTree
tree = ET.parse("M-I 01.09.18.PM2.45610.SJM (bjmccowan@ucdavis.edu).odx")

# Find all "event" elements indicated by <OBS_EVENT>
event_elements = tree.findall(".//{}OBS_EVENT".format(namespace))

# Create a text file to write the information
with open("event_information.txt", "w") as file:
    current_observation_number = 0

    # Process each event element
    for event_element in event_elements:
        # Get the OBS_EVENT_STATE
        event_state = event_element.find("{}OBS_EVENT_STATE".format(namespace)).text

        # Start of an observation (OBS_EVENT_STATE = 1)
        if event_state == "1":
            current_observation_number += 1  # Increment current observation number
            current_start = current_observation_number
            observation_number = current_observation_number  # Assign current number as observation number for this event

            # Get the timestamp from OBS_EVENT_TIMESTAMP element
            start_timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text

            # Get the OBS_EVENT_BEHAVIOR NAME
            behavior_name = event_element.find("{}OBS_EVENT_BEHAVIOR".format(namespace)).attrib["NAME"]
            # Get all OBS_EVENT_BEHAVIOR_MODIFIER elements within the event
            behavior_modifiers = event_element.findall(".//{}OBS_EVENT_BEHAVIOR_MODIFIER".format(namespace))

            # Create a list to store all behavior modifiers
            modifiers_list = []
            for modifier in behavior_modifiers:
                modifier_class = modifier.attrib["CLASS"]
                modifier_value = modifier.text
                modifiers_list.append((modifier_class, modifier_value))

            # Get the OBS_EVENT_MARKER
            event_marker = event_element.find("{}OBS_EVENT_MARKER".format(namespace)).text
        # End of an observation (OBS_EVENT_STATE = 2)
        elif event_state == "2":
            observation_number = current_start   # Assign the previous number as observation number for this event

            stop_timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text

        # OBS_EVENT_STATE = 3 events will be labeled with the current observation number
        else:
            current_observation_number += 1  # Increment current observation number
            observation_number = current_observation_number

            # Get the timestamp from OBS_EVENT_TIMESTAMP element
            start_timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text
            stop_timestamp = start_timestamp
            # Get the OBS_EVENT_BEHAVIOR NAME
            behavior_name = event_element.find("{}OBS_EVENT_BEHAVIOR".format(namespace)).attrib["NAME"]
            # Get all OBS_EVENT_BEHAVIOR_MODIFIER elements within the event
            behavior_modifiers = event_element.findall(".//{}OBS_EVENT_BEHAVIOR_MODIFIER".format(namespace))

            # Create a list to store all behavior modifiers
            modifiers_list = []
            for modifier in behavior_modifiers:
                modifier_class = modifier.attrib["CLASS"]
                modifier_value = modifier.text
                modifiers_list.append((modifier_class, modifier_value))

            # Get the OBS_EVENT_MARKER
            event_marker = event_element.find("{}OBS_EVENT_MARKER".format(namespace)).text

        if event_state == "2" or event_state == "3":
            # Write the information to the text file
            file.write("Observation Number: {}\n".format(observation_number))
            file.write("Timestamp: {},{}\n".format(start_timestamp, stop_timestamp))
            file.write("Behavior: {}\n".format(behavior_name))
            file.write("Modifiers:\n")
            for modifier_class, modifier_value in modifiers_list:
                file.write("  {} : {}\n".format(modifier_class, modifier_value))
            file.write("Marker: {}\n".format(event_marker))
            file.write("\n")
