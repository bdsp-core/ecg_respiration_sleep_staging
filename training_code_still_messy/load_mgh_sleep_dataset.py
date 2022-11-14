import datetime
import re
import os.path
import subprocess
import numpy as np
import scipy.io as sio
import h5py


def check_load_Twin_dataset(data_path, label_path, channels=None, report_and_actual_time_tol=300, reverse_sign=False):

    """
    try:
        ff = h5py.File(data_path)
        EEG = ff['s'][()]
        channel_names = ff['hdr']['Label'][()].flatten()
        channel_names = [''.join(chr(ff[channel_names[i]][()].flatten()[j]) for j in range(ff[channel_names[i]][()].flatten().shape[0])).upper() for i in range(len(channel_names))]
        physicalMin = ff['hdr']['PhysMin'][()].flatten()
        physicalMax = ff['hdr']['PhysMax'][()].flatten()
        digitalMin = ff['hdr']['DigMin'][()].flatten()
        digitalMax = ff['hdr']['DigMax'][()].flatten()
        gender = ff['hdr']['Patient']['Sex'][()][0,0]
        handedness = ff['hdr']['Patient']['Handedness'][()][0,0]
        # check sample frequency
        if 'SampleRate' in ff['hdr']:
            Fs = ff['hdr']['SampleRate'][()][0,0]
        else:
            Fs = None
    except:
    """
    ff = sio.loadmat(data_path)
    data_path = os.path.basename(data_path)
    if 's' not in ff:
        raise Exception('No signal found in %s.'%data_path)
    signal = ff['s']
    if reverse_sign:
        signal = -signal
    channel_names = [ff['hdr'][0,i]['signal_labels'][0].upper().replace('EKG','ECG') for i in range(ff['hdr'].shape[1])]
    physicalMin = np.array([ff['hdr'][0,i]['physical_min'][0,0] for i in range(ff['hdr'].shape[1])])
    physicalMax = np.array([ff['hdr'][0,i]['physical_max'][0,0] for i in range(ff['hdr'].shape[1])])
    digitalMin = np.array([ff['hdr'][0,i]['digital_min'][0,0] for i in range(ff['hdr'].shape[1])])
    digitalMax = np.array([ff['hdr'][0,i]['digital_max'][0,0] for i in range(ff['hdr'].shape[1])])
    gender = None
    handedness = None
    """
    for i in range(len(ff['hdr']['Patient'][0,0].dtype.names)):
        if ff['hdr']['Patient'][0,0].dtype.names[i].lower()=='sex':
            gender = ff['hdr']['Patient'][0,0][0,0][i][0,0]
        elif ff['hdr']['Patient'][0,0].dtype.names[i].lower()=='handedness':
            handedness = ff['hdr']['Patient'][0,0][0,0][i][0,0]
    try:
        Fs = ff['hdr']['SampleRate'][0,0][0,0]
    except:
        Fs = None
    if Fs is None:
        #print('\nNo sampleFrequency in %s. Use default 200Hz.'%data_path)
    """
    Fs = 200.

    # load labels
    with h5py.File(label_path) as ffl:
        sleep_stage = ffl['stage'][()].flatten()
        time_str_elements = ffl['features']['StartTime'][()].flatten()
        start_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
        time_str_elements = ffl['features']['EndTime'][()].flatten()
        end_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
    """
    except:
    ffl = sio.loadmat(label_path)
    sleep_stage = ffl['stage'].flatten()
    start_time = ffl['features']['StartTime'][0,0][0,0]
    end_time = ffl['features']['EndTime'][0,0][0,0]
    """

    start_time = start_time.split(':')
    second_elements = start_time[-1].split('.')
    start_time = datetime.datetime(1990,1,1,hour=int(float(start_time[0])), minute=int(float(start_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))
    end_time = end_time.split(':')
    second_elements = end_time[-1].split('.')
    end_time = datetime.datetime(1990,1,1,hour=int(float(end_time[0])), minute=int(float(end_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))

    # check signal length = sleep stage length
    if sleep_stage.shape[0]!=signal.shape[1]:
        raise Exception('Inconsistent sleep stage length (%d) and signal length (%d) in %s'%(sleep_stage.shape[0],signal.shape[1],data_path))

    # check end_time - start_time = signal signal_duration
    #reported_time_diff = (end_time-start_time).seconds
    #signal_duration = signal.shape[1]*1.0/Fs
    #time_diff = abs(reported_time_diff-signal_duration)
    #if time_diff>report_and_actual_time_tol:
    #    raise Exception('end_time-start_time= %ds, signal= %ds, difference= %ds in %s'%(reported_time_diff,signal_duration,time_diff,data_path))

    # check channel number
    if not signal.shape[0]==len(channel_names)==\
            physicalMin.shape[0]==physicalMax.shape[0]==\
            digitalMin.shape[0]==digitalMax.shape[0]:
        raise Exception('Inconsistent channel number in %s'%data_path)

    # only take signal channels to study
    if channels is None:
        signal_channel_ids = list(range(len(channel_names)))
    else:
        signal_channel_ids = []
        for i in range(len(channels)):
            #channel_name_pattern = re.compile(channels[i][:2].upper()+'-*'+channels[i][-2:].upper())
            found = False
            for j in range(len(channel_names)):
                if channel_names[j]==channels[i].upper():
                    signal_channel_ids.append(j)
                    found = True
                    break
            if not found:
                raise Exception('Channel %s is not found.'%channels[i])
        signal = signal[signal_channel_ids,:]#.T
        physicalMin = physicalMin[signal_channel_ids]
        physicalMax = physicalMax[signal_channel_ids]
        digitalMin = digitalMin[signal_channel_ids]
        digitalMax = digitalMax[signal_channel_ids]

    # check whether the signal contains NaN
    if np.any(np.isnan(signal)):
        raise Exception('Found Nan in signal in %s'%data_path)

    # check signal min/max
    s_min = np.min(signal,axis=1)
    s_max = np.max(signal,axis=1)
    if np.any(s_min<physicalMin) or np.any(s_max>physicalMax):
        raise Exception('Signal exceeds physical min/max in %s'%data_path)
    if np.any(s_min<digitalMin) or np.any(s_max>digitalMax):
        raise Exception('Signal exceeds digital min/max in %s'%data_path)

    # check whether sleep_stage contains NaN
    #if np.any(np.isnan(sleep_stage)):
    #    raise Exception('Found Nan in sleep stages in %s'%data_path)

    # check whether sleep_stage contains all 5 stages
    stages = np.unique(sleep_stage[np.logical_not(np.isnan(sleep_stage))]).astype(int).tolist()
    if len(stages)<=2:
        raise Exception('#sleep stage <= 2: %s in %s'%(stages,data_path))

    params = {'Fs':Fs, 'signal_channel_ids':signal_channel_ids}
    if gender is not None:
        params['gender'] = gender
    if handedness is not None:
        params['handedness'] = handedness
    return signal, sleep_stage, params


"""
def convert_to_RR_interval(signal, Fs):
    np.savetxt('ecg.txt', signal.T, fmt='%f')
    subprocess.check_call(['wrsamp', '-F', '%d'%Fs, '-i', 'ecg.txt', '-o', 'ecg', '0'])#, '-G', '102.4'
    #subprocess.check_call(['gqrs', '-r', 'ecg', '-c', 'gqrs.conf'])
    #subprocess.check_call(['gqpost', '-r', 'ecg', '-c', 'gqrs.conf', '-a', 'qrs'])
    subprocess.check_call(['wqrs', '-r', 'ecg'])
    res = subprocess.check_output(['ann2rr', '-r', 'ecg', '-a', 'wqrs', '-v', '-V', '-i', 's', '-c', '-r', 'ecg'])
    
    res2 = []
    res_rows = res.split('\n')
    for row in res_rows:
        elements = row.split('\t')
        if len(elements)==3:
            res2.append([int(elements[0]),int(elements[2]),float(elements[1])])
        
    return np.array(res2)
"""
