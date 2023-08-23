import ctypes as ct
import numpy as np

def createWrapper():
    # load CED lib
    ced_lib = ct.LibraryLoader(ct.WinDLL).LoadLibrary(
        "C:\\Henry\\PythonProjects\\CED_interface\\CEDS64ML\\x64\\ceds64int.dll")
    return ced_lib

def openFile(file_name, ced_lib):
    # openfile: opens .smr(x) files
    # inputs:
    # file_name: name of file (with directory)
    # ced_lib: created with createWrapper()
    ced_lib.S64Open.argtypes = [ct.POINTER(ct.c_char)]
    fhand = ced_lib.S64Open(file_name)
    return fhand

def ceds64_max_chan(fhand, ced_lib):
    # Load the C library

    # Define the argument and return types for the C function
    ced_lib.S64MaxChans.restype = ct.c_int
    ced_lib.S64MaxChans.argtypes = [ct.c_int]

    # Call the C function
    iMaxChans = ced_lib.S64MaxChans(fhand)

    return iMaxChans

# def get_channel_label(fhand):

