import numpy as np
import matplotlib.pyplot as plt

def eydexp(filename):
    s = {}
    s['filename'] = filename
    
    with open(filename, 'rb') as fid:
        s = getHeader(fid, s)
        
        s = readSegmentDirectory(fid, s)
  
        s = readSegmentData(fid, s, 0)
        
    printSummary(s)
    return s

def readNumberFromEydLine(s):
    i = np.nan
    C = s.split()
    if len(C) > 1:
        i = float(C[1])
    return i

def readStringFromEydLine(line):
    s = line.split(':')[1].strip()
    return s

def readSegmentDirectory(fid, s):
    if 'nsegments' in s and 'segdiraddr' in s:
        
        seg_count = int(s['nsegments'])
        seg_dir = int(s['segdiraddr'])
        fid.seek(seg_dir)
        s['segments'] = np.zeros((7, seg_count), dtype=np.uint32)
        for i in range(seg_count):
            b = np.fromfile(fid, dtype=np.uint8, count=1)[0]
            if b != 0xFB:
                raise ValueError('ERROR - start of segment record not found.')
            
            seg = np.fromfile(fid, dtype=np.uint32, count=7)
        
            s['segments'][:, i] = seg
    else:
        print('No segment directory in this file!')

    return s

def getHeader(fid, s):
    for tline in fid:
        tline = tline.decode('utf-8')
        if 'Update_Rate(Hz)' in tline:
            s['rate'] = readNumberFromEydLine(tline)
        elif 'Creation_Date' in tline:
            s['date'] = readStringFromEydLine(tline)
        elif 'User_Recorded_Segments' in tline:
            s['nsegments'] = readNumberFromEydLine(tline)
        elif 'Segment_Directory_Start_Address' in tline:
            s['segdiraddr'] = readNumberFromEydLine(tline)
        elif '[Segment_Data]' in tline:
            s['segdata'] = fid.tell()  # filepos of segment data
            break
    return s

def readSegmentData(fid, s, n):
    print(f'readSegmentData: Expecting {s["segments"][6, n]} records in segment {n}')

    data = {'status': np.zeros(s['segments'][6, n], dtype=np.uint8),
            'overtime': np.zeros(s['segments'][6, n], dtype=np.uint16),
            'xdat': np.zeros(s['segments'][6, n], dtype=np.uint8),
            'pupil': np.zeros(s['segments'][6, n], dtype=np.uint16),
            'x': np.zeros(s['segments'][6, n], dtype=np.double),
            'y': np.zeros(s['segments'][6, n], dtype=np.double),
            'videofield': np.zeros(s['segments'][6, n], dtype=np.uint16)}
    fid.seek(s['segments'][1, n])
    for i in range(s['segments'][6, n]):
        startrecord = np.fromfile(fid, dtype=np.uint8, count=1)[0]
        if startrecord != 0xFA:
            raise ValueError(f'Error at record {i}, start of record not found')
        data['status'][i] = np.fromfile(fid, dtype=np.uint8, count=1)
        data['overtime'][i] = np.fromfile(fid, dtype=np.uint16, count=1)
        fid.seek(1, 1)  # marker - skip
        data['xdat'][i] = np.fromfile(fid, dtype=np.uint16, count=1)
        data['pupil'][i] = np.fromfile(fid, dtype=np.uint16, count=1)
        x = np.fromfile(fid, dtype=np.int16, count=1)
        data['x'][i] = x * 0.1
        y = np.fromfile(fid, dtype=np.int16, count=1)
        data['y'][i] = x * 0.1
        fid.seek(12, 1)
        data['videofield'][i] = np.fromfile(fid, dtype=np.uint16, count=1)

    bb = np.fromfile(fid, dtype=np.uint8, count=27)
    if len(bb) != 27:
        print('cannot read end of segdata record?')
    s['data'] = data
    return s

def printSummary(s):
    # breakpoint()
    print(f'\nSummary for {s["filename"]}')
    print(f'created : {s["date"]}')
    print(f'rate(Hz): {s["rate"]}')
    print(f'# segments: {s["nsegments"]}')
    for i in range(int(s["nsegments"])):
        print(f'Segment {i + 1}:')
        print(f'Start frame: {s["segments"][2, i]}')
        print(f'End frame  : {s["segments"][3, i]}')
        print(f'num records: {s["segments"][6, i]} (expect {s["segments"][3, i] - s["segments"][2, i]})')
        print(f'overtime (s/b=0): {np.sum(s["data"]["overtime"][i])}')





def synch_pupil_data(eyd, smr):

    code_values, time_values = zip(*[(x['code'], x['time']) for x in smr])

    # Converting the extracted lists to numpy arrays
    code_array = np.array(code_values)
    time_array = np.array(time_values)
    
    pupil_markers = [101, 102, 103, 104, 105, 106, 107]
    valid_code = np.isin(code_array, pupil_markers)
    
    code_array = code_array[valid_code]
    time_array = time_array[valid_code]
    
    temp_remove = np.where(code_array != 101)[0][0]
    code_array = code_array[np.r_[:1, temp_remove:code_array.shape[0]]]     
    time_array = time_array[np.r_[:1, temp_remove:time_array.shape[0]]]    

    eyd['data']['time'] = np.full(len(eyd['data']['xdat']), np.nan)

    trans_vector_full = np.diff(eyd['data']['xdat'])
    
    trans_vector_full = np.concatenate((trans_vector_full, [0]))
    trans_vector = np.nonzero(trans_vector_full != 0)[0]
    
    eyd['rate'] = np.median(np.diff(trans_vector) / np.diff(time_array))
    
    trans_tracker = np.zeros((len(trans_vector), 3))
    sync_check = np.zeros((len(trans_vector), 3))
    for index in range(len(trans_vector)):
        trans_tracker[index, :] = [code_array[index], eyd['data']['xdat'][trans_vector[index]], trans_vector[index]]
        if index == 0:
            eyd['data']['time'][:trans_vector[index]] = np.arange(-trans_vector[index], 0, 1) * \
                (1 / eyd['rate']) + time_array[index]
            
            eyd['data']['time'][trans_vector[index]] = time_array[index]
            s = len(eyd['data']['time'][trans_vector[index] + 1:])
            eyd['data']['time'][trans_vector[index] + 1:] = np.arange(1, s + 1) * (1 / eyd['rate']) + time_array[index]
        else:
            pass  # You can add the necessary code here

        
        sync_check[index, :] = [time_array[index], eyd['data']['time'][trans_vector[index] + 1], \
                                time_array[index] - eyd['data']['time'][trans_vector[index] + 1]]

    return eyd
