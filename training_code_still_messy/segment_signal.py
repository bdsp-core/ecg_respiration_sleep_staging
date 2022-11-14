#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
import mne
#mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter
#from extract_features_parallel import *
from biosppy.signals import ecg


seg_mask_explanation = [
    'normal',
    'around sleep stage change point',
    'NaN in sleep stage',
    'NaN in signal',
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'muscle artifact',
    'spurious spectrum',
    'missing or additional R peak']


def segment_ecg_signal(signal, labels, window_time, step_time, Fs, start_end_remove_window_num=1, n_jobs=-1, to_remove_mean=False, amplitude_thres=500):#, BW=4,
    """Segment ECG signals.

    Arguments:
    signal -- np.ndarray, size=(channel_num, sample_num)
    labels -- np.ndarray, size=(sample_num,)
    window_time -- in seconds
    Fz -- in Hz

    Keyword arguments:
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the signal
    amplitude_thres -- default 1000, mark all segments with np.any(signal_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of signal from each channel

    Outputs:    
    signal segments -- a list of np.ndarray, each has size=(window_size, channel_num)
    labels --  a list of labels of each window
    segment start ids -- a list of starting ids of each window in (sample_num,)
    segment masks --
    """
    std_thres = 2
    std_thres2 = 5
    flat_seconds = 30
    
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    
    start_ids = np.arange(0, signal.shape[1]-window_size+1, step_size)
    if len(start_ids)!=len(labels):
        labels = labels[start_ids+window_size//2]
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
        labels = labels[start_end_remove_window_num:-start_end_remove_window_num]
    assert len(start_ids)==len(labels)

    seg_masks = [seg_mask_explanation[0]]*len(start_ids)

    if np.any(np.isnan(labels)):
        ids = np.where(np.isnan(labels))[0]
        for i in ids:
            seg_masks[i] = seg_mask_explanation[2]

    assert signal.shape[0]==1
    ecg_analysis_res = ecg.ecg(signal=signal.flatten(),sampling_rate=Fs,show=False)
    rpeaks = ecg_analysis_res['rpeaks']

    signal_segs = signal[:,map(lambda x:np.arange(x,x+window_size), start_ids)].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    
    ## find nan in signal
    
    nan2d = np.any(np.isnan(signal_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = seg_mask_explanation[3]
            
    amplitude_large2d = np.any(np.abs(signal_segs)>amplitude_thres, axis=2)
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
        seg_masks[i] = seg_mask_explanation[4]
            
    flat_length = int(round(flat_seconds*Fs))
    # if there is any flat signal with flat_length
    short_segs = signal_segs.reshape(signal_segs.shape[0], signal_segs.shape[1], signal_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres, axis=2)
    flat2d = np.logical_or(flat2d, np.std(signal_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        seg_masks[i] = seg_mask_explanation[5]

    # artifact removal for the rr peaks using ADARRI
    adarri = np.log1p(np.abs(np.diff(np.diff(rpeaks))))
    artifact_pos = np.where(adarri>=5)[0]+2
    rpeak_q10, rpeak_q90 = np.percentile(signal[0,rpeaks],(10,90))
    artifact_pos = artifact_pos.tolist() + np.where((signal[0,rpeaks]<rpeak_q10-300) | (signal[0,rpeaks]>rpeak_q90+300))[0].tolist()
    artifact_pos = np.sort(np.unique(artifact_pos))
    
    rr_peak_artifact1d = []
    for ap in artifact_pos:
        rr_peak_artifact1d.extend(np.where((start_ids<=rpeaks[ap]) & (rpeaks[ap]<start_ids+window_size))[0])
    rr_peak_artifact1d = np.sort(np.unique(rr_peak_artifact1d))
    for i in rr_peak_artifact1d:
        seg_masks[i] = seg_mask_explanation[10]

    # create binary signal
    signal = np.zeros_like(signal, dtype='double')
    signal[0,rpeaks] = 1.
    signal_segs = signal[:,map(lambda x:np.arange(x,x+window_size), start_ids)].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    
    # mark binary signals with all 0's
    fewbeats2d = signal_segs.sum(axis=2)<4.5*60/3 # can't have less than 20 beats/min
    fewbeats1d = np.where(np.any(fewbeats2d, axis=1))[0]
    for i in fewbeats1d:
        seg_masks[i] = seg_mask_explanation[10]
        
    return signal_segs, labels, start_ids, seg_masks#, specs, freq


def segment_chest_signal(signal, labels, window_time, step_time, Fs, newFs=None, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=1, amplitude_thres=None, n_jobs=-1, to_remove_mean=False):#, BW=4
    """Segment CHEST signals.

    Arguments:
    signal -- np.ndarray, size=(channel_num, sample_num)
    labels -- np.ndarray, size=(sample_num,)
    window_time -- in seconds
    Fz -- in Hz

    Keyword arguments:
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the signal
    amplitude_thres -- default 1000, mark all segments with np.any(signal_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of signal from each channel

    Outputs:    
    signal segments -- a list of np.ndarray, each has size=(window_size, channel_num)
    labels --  a list of labels of each window
    segment start ids -- a list of starting ids of each window in (sample_num,)
    segment masks --
    """
    std_thres1 = 5
    std_thres2 = 10
    
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    
    start_ids = np.arange(0, signal.shape[1]-window_size+1, step_size)
    if len(start_ids)!=len(labels):
        labels = labels[start_ids+window_size//2]
        
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
        labels = labels[start_end_remove_window_num:-start_end_remove_window_num]
    assert len(start_ids)==len(labels)

    seg_masks = [seg_mask_explanation[0]]*len(start_ids)

    if np.any(np.isnan(labels)):
        ids = np.where(np.isnan(labels))[0]
        for i in ids:
            seg_masks[i] = seg_mask_explanation[2]

    assert signal.shape[0]==1
    
    ## filter signal
    if notch_freq is not None and np.max(bandpass_freq)>=notch_freq:
        signal = notch_filter(signal, Fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size+2padding)
    signal = filter_data(detrend(signal, axis=1), Fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR')  # take the value starting from *padding*, (#window, #ch, window_size+2padding)
    signal_segs = signal[:,map(lambda x:np.arange(x,x+window_size), start_ids)].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    if newFs is not None:
        assert int(Fs)//int(newFs)==Fs/newFs
        mne_epochs = mne.EpochsArray(signal_segs, mne.create_info(ch_names=['aa'], sfreq=Fs, ch_types='eeg'), verbose=False)
        mne_epochs.decimate(int(Fs)//int(newFs))
        signal_segs = mne_epochs.get_data()
        Fs = newFs
    
    ## find nan in signal
    
    nan2d = np.any(np.isnan(signal_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[3], np.where(nan2d[i])[0])
        
    if amplitude_thres is not None:
        ## find large amplitude in signal
                
        amplitude_large2d = np.any(np.abs(signal_segs)>amplitude_thres, axis=2)
        amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
        for i in amplitude_large1d:
            seg_masks[i] = '%s_%s'%(seg_mask_explanation[4], np.where(amplitude_large2d[i])[0])
            
    ## find flat signal
    
    flat_seconds = 6
    flat_length = int(round(flat_seconds*Fs))
    assert signal_segs.shape[2]//flat_length*flat_length==signal_segs.shape[2]
    short_segs = signal_segs.reshape(signal_segs.shape[0], signal_segs.shape[1], signal_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres1, axis=2)
    flat2d = np.logical_or(flat2d, np.std(signal_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[5], np.where(flat2d[i])[0])
        
    return signal_segs, labels, start_ids, seg_masks#, specs, freq
