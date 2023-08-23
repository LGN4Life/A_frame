import ced
def load_file(file_name):
    # file_name = 'C:\\Henry\\PythonProjects\\CED_interface\\data\\tmj_230731_tun_000'
    #file_name = 'C:\\Henry\\PythonProjects\\CED_interface\\data\\bet_150527_atn_000'
    chan_num = 1

    # params for loading spike data
    spike_channel = 3
    wave_mark = 1
    mask_handle = 1

    # load ced library
    ced_lib = ced.createWrapper()
    print(f"Successfully loaded ced library: type {type(ced_lib)}")

    # check for file and open file (if it exists)
    flag, file_name = ced.check_for_file(file_name)

    if flag:
        fhand = ced.open_file(file_name, ced_lib)
        if fhand > 0:
            print(f"file opened successfully: {file_name}")
        else:
            print(f"file could not be opened: {file_name}")
    else:
        print(f"file not found: {file_name}")


    chan_num = 1
    all_channels = ced.get_channel_info(fhand, ced_lib)

    #chan_title = ced.getChanTitle(1, fhand, ced_lib)

    print(f"{len(all_channels)} channels processed")
    return all_channels


