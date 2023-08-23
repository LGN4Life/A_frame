import xml.etree.ElementTree as ET

# Namespace prefix
namespace = "{nxmlvosobservations}"

# Parse the XML file into an ElementTree
tree = ET.parse("M-I 01.09.18.PM2.45610.SJM (bjmccowan@ucdavis.edu).odx")

# Find all "event" elements indicated by <OBS_EVENT>
event_elements = tree.findall(".//{}OBS_EVENT".format(namespace))

# Create a text file to write the information
with open("event_information.txt", "w") as file:
    # Process each event element
    for event_element in event_elements:
        # Get the timestamp from OBS_EVENT_TIMESTAMP element
        timestamp = event_element.find("{}OBS_EVENT_TIMESTAMP".format(namespace)).text

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

        # Get the OBS_EVENT_STATE
        event_state = event_element.find("{}OBS_EVENT_STATE".format(namespace)).text

        # Get the OBS_EVENT_MARKER
        event_marker = event_element.find("{}OBS_EVENT_MARKER".format(namespace)).text

        # Write the information to the text file
        file.write("Timestamp: {}\n".format(timestamp))
        file.write("Behavior: {}\n".format(behavior_name))
        file.write("Modifiers:\n")
        for modifier_class, modifier_value in modifiers_list:
            file.write("  {} : {}\n".format(modifier_class, modifier_value))
        file.write("State: {}\n".format(event_state))
        file.write("Marker: {}\n".format(event_marker))
        file.write("\n")
