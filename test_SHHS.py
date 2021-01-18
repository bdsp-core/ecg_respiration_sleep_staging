from copy import deepcopy
import datetime
import glob
import os
import pickle
import sys
import numpy as np
import pandas as pd
import h5py
#from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
import mne
import torch as th
from segment_signal import *
from mymodel import *


"""
from
https://github.com/robintibor/braindecode/blob/master/braindecode/torch_ext/util.py
"""
def np_to_var(
    X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    """
    Convenience function to transform numpy array to `torch.Tensor`.
    Converts `X` to ndarray using asarray if necessary.
    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor
    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


def var_to_np(var):
    """Convenience function to transform `torch.Tensor` to numpy
    array.
    Should work both for CPU and GPU."""
    return var.cpu().data.numpy()
    
    
if __name__ == '__main__':
    data_source = sys.argv[1] # ECG or ABD or CHEST or ECG+ABD or ECG+CHEST
    data_source_split = data_source.split('+')
    window_time = 4.5*60 # [s]
    step_time = 30 # [s]
    notch_freq = [50, 60]  # [Hz]
    bandpass_freq = [None, 4.]  # [Hz]
    n_gpu = 1
    trained_model_paths = {
            'ECG':'models/CNN_ECG_fold1.pth',
            'ABD':'models/CNN_ABD_fold1.pth',
            'CHEST':'models/CNN_CHEST_fold1.pth' }

    # assume you have these
    shhs_dir = '/media/mad3/Projects/SLEEP/SLEEP_STAGING/shhs_0.8.0/shhs'
    all_shhs_paths = glob.glob(os.path.join(shhs_dir, 'edfs/shhs1/*.edf')) + glob.glob(os.path.join(shhs_dir, 'edfs/shhs2/*.edf'))
    all_shhs_paths = sorted(all_shhs_paths)
        
    # decide average amplitude of SHHS respiration
    amplitude_path = 'amplitudes.pickle'
    if os.path.exists(amplitude_path):
        with open(amplitude_path, 'rb') as ff:
            SHHS_chest_amp, SHHS_abd_amp, MGH_chest_amp, MGH_abd_amp = pickle.load(ff, encoding='latin1')
    else:
        """
        with h5py.File('/data/cross_channel_sleep_staging/preprocessed_data/ABD_feedforward.h5', 'r') as ff:
            N = len(ff['X'])
            ids = np.random.choice(N, N//1000, replace=False)
            MGH_abd_amp = np.mean(np.abs(ff['X'][sorted(ids)]))
        with h5py.File('/data/cross_channel_sleep_staging/preprocessed_data/CHEST_feedforward.h5', 'r') as ff:
            N = len(ff['X'])
            ids = np.random.choice(N, N//1000, replace=False)
            MGH_chest_amp = np.mean(np.abs(ff['X'][sorted(ids)]))
        """
        MGH_abd_amp = 327.0
        MGH_chest_amp = 200.9
        
        SHHS_chest_amp = []
        SHHS_abd_amp = []
        tostudy_paths = np.random.choice(all_shhs_paths, 100, replace=False)
        for pi, path in enumerate(tqdm(tostudy_paths)):
            ff = mne.io.read_raw_edf(path, preload=True, verbose=False, stim_channel=None)
            channel_names = ff.info['ch_names']
            signals = ff.get_data()
            SHHS_abd_amp.append(np.mean(np.abs(signals[channel_names.index('ABDO RES')])))
            SHHS_chest_amp.append(np.mean(np.abs(signals[channel_names.index('THOR RES')])))
        SHHS_chest_amp = np.mean(SHHS_chest_amp)
        SHHS_abd_amp = np.mean(SHHS_abd_amp)
        #SHHS_chest_amp = 0.18637139142980938
        #SHHS_abd_amp = 0.2417486199656767
        
        with open(amplitude_path, 'wb') as ff:
            pickle.dump([SHHS_chest_amp, SHHS_abd_amp, MGH_chest_amp, MGH_abd_amp], ff)
    
    output_path = 'SHHS_predictions_%s.pickle'%data_source.lower()
    paths = []
    ys = []
    yps = []
    
    for pi, path in enumerate(all_shhs_paths):
        print('[%d/%d %s] %s'%(pi+1, len(all_shhs_paths), datetime.datetime.now(), '/'.join(path.split('/')[-2:])))
        try:
            # read signal
            ff = mne.io.read_raw_edf(path, preload=True, verbose=False, stim_channel=None)
            channel_names = ff.info['ch_names']
            signals = ff.get_data()
            Fs = ff.info['sfreq']
            
            """      
            # read stages
            # 0 = Wake  ==> 5
            # 1 = Stage 1 Sleep ==> 3
            # 2 = Stage 2 Sleep ==> 2
            # 3 = Stage 3 Sleep ==> 1
            # 4 = Stage 4 Sleep ==> 1
            # 5 = REM Sleep ==> 4
            # 6 = Wake/Movement ==> nan
            # 9 = Unscored ==> nan
            mapping = {0:5, 1:3, 2:2, 3:1, 4:1, 5:4}
            stage_path = os.path.join(shhs_dir, 'annotations-staging', path.split(os.sep)[-2],os.path.basename(path).replace('.edf','-staging.csv'))
            df_stages = pd.read_csv(stage_path)
            stages  = np.zeros(signals.shape[-1])+np.nan
            for si in range(len(df_stages)):
                start = int((df_stages.Epoch.iloc[si]-1)*30*Fs)
                end = int(df_stages.Epoch.iloc[si]*30*Fs)
                if 0<=start<len(stages):
                    stages[start:end] = mapping.get(df_stages.Stage.iloc[si], np.nan)
            
            epoch_len = min(int(np.floor(signals.shape[1]/30./Fs)), len(df_stages))
            signal_len = int(epoch_len*30*Fs)
            stages = stages[:signal_len]
            signals = signals[:, :signal_len]
            """ 
            stages = None     

            # preprocess signals
            if 'ECG' in data_source_split:
                ecg = signals[channel_names.index('ECG')].reshape(1,-1)*1e3
                ecg_segs, _, _, _ = segment_ecg_signal(ecg, stages, window_time, step_time, Fs, newFs=200, start_end_remove_window_num=0, n_jobs=-1, amplitude_thres=6000)
            if 'ABD' in data_source_split:
                abd = signals[channel_names.index('ABDO RES')].reshape(1,-1)*MGH_abd_amp/ SHHS_abd_amp
                abd_segs, _, _, _ = segment_chest_signal(abd, stages, window_time, step_time, Fs, newFs=10, notch_freq=notch_freq, bandpass_freq=bandpass_freq, start_end_remove_window_num=0, amplitude_thres=6000, n_jobs=-1)
            if 'CHEST' in data_source_split:
                chest = signals[channel_names.index('THOR RES')].reshape(1,-1)*MGH_chest_amp/ SHHS_chest_amp
                chest_segs, _, _, _ = segment_chest_signal(chest, stages, window_time, step_time, Fs, newFs=10, notch_freq=notch_freq, bandpass_freq=bandpass_freq, start_end_remove_window_num=0, amplitude_thres=6000, n_jobs=-1)

            if data_source=='ECG':
                # load ECG model
                ecg_cnn_model = ECGSleepNet()
                ecg_cnn_model.load_state_dict(th.load(trained_model_paths[data_source]))
                if n_gpu>0:
                    ecg_cnn_model = ecg_cnn_model.cuda()
                    if n_gpu>1:
                        ecg_cnn_model = nn.DataParallel(ecg_cnn_model, device_ids=list(range(n_gpu)))
                ecg_cnn_model.eval()
                ecg_rnn_model = SleepNet_RNN(1280, 5, 20, 2, dropout=0, bidirectional=True)
                ecg_rnn_model.load_state_dict(th.load('models/LSTM_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    ecg_rnn_model = ecg_rnn_model.cuda()
                    if n_gpu>1:
                        ecg_rnn_model = nn.DataParallel(ecg_rnn_model, device_ids=list(range(n_gpu)))
                ecg_rnn_model.eval()
                
                # feed to ECG model
                X = np_to_var(ecg_segs.astype('float32'))
                if n_gpu>0:
                    X = X.cuda()
                with th.no_grad():
                    ids = np.array_split(np.arange(len(X)), 50)
                    H = []
                    for id_ in ids:
                        _, H_ = ecg_cnn_model(X[id_])
                        H.append(H_)
                    H = th.cat(H, dim=0)
                    H = H.reshape(1, H.shape[0], -1)
                    yp, _ = ecg_rnn_model(H)
                yp = var_to_np(yp)[0].astype(float)
                yp = np.argmax(yp, axis=1)+1
                
            elif data_source=='ABD':
                
                # load ABD model
                abd_cnn_model = CHESTSleepNet()
                abd_cnn_model.load_state_dict(th.load(trained_model_paths[data_source]))
                if n_gpu>0:
                    abd_cnn_model = abd_cnn_model.cuda()
                    if n_gpu>1:
                        abd_cnn_model = nn.DataParallel(abd_cnn_model, device_ids=list(range(n_gpu)))
                abd_cnn_model.eval()
                abd_rnn_model = SleepNet_RNN(768, 5, 100, 2, dropout=0, bidirectional=True)
                abd_rnn_model.load_state_dict(th.load('models/LSTM_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    abd_rnn_model = abd_rnn_model.cuda()
                    if n_gpu>1:
                        abd_rnn_model = nn.DataParallel(abd_rnn_model, device_ids=list(range(n_gpu)))
                abd_rnn_model.eval()
                
                # feed to ABD model
                X = np_to_var(abd_segs.astype('float32'))
                if n_gpu>0:
                    X = X.cuda()
                with th.no_grad():
                    _, H = abd_cnn_model(X)
                    H = H.reshape(1, H.shape[0], -1)
                    yp, _ = abd_rnn_model(H)
                yp = var_to_np(yp)[0].astype(float)
                yp = np.argmax(yp, axis=1)+1
                
            elif data_source=='CHEST':
            
                # load CHEST model
                chest_cnn_model = CHESTSleepNet()
                chest_cnn_model.load_state_dict(th.load(trained_model_paths[data_source]))
                if n_gpu>0:
                    chest_cnn_model = chest_cnn_model.cuda()
                    if n_gpu>1:
                        chest_cnn_model = nn.DataParallel(chest_cnn_model, device_ids=list(range(n_gpu)))
                chest_cnn_model.eval()
                chest_rnn_model = SleepNet_RNN(768, 5, 100, 3, dropout=0.5, bidirectional=True)
                chest_rnn_model.load_state_dict(th.load('models/LSTM_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    chest_rnn_model = chest_rnn_model.cuda()
                    if n_gpu>1:
                        chest_rnn_model = nn.DataParallel(chest_rnn_model, device_ids=list(range(n_gpu)))
                chest_rnn_model.eval()
                
                # feed to CHEST model
                X = np_to_var(chest_segs.astype('float32'))
                if n_gpu>0:
                    X = X.cuda()
                with th.no_grad():
                    _, H = chest_cnn_model(X)
                    H = H.reshape(1, H.shape[0], -1)
                    yp, _ = chest_rnn_model(H)
                yp = var_to_np(yp)[0].astype(float)
                yp = np.argmax(yp, axis=1)+1

            elif data_source=='ECG+ABD':

                # load ECG+ABD model
                ecgabd_cnn_model = CombinedSleepNet2([trained_model_paths[ds] for ds in data_source_split])
                ecgabd_cnn_model.load_state_dict(th.load('models/CNN_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    ecgabd_cnn_model = ecgabd_cnn_model.cuda()
                    if n_gpu>1:
                        ecgabd_cnn_model = nn.DataParallel(ecgabd_cnn_model, device_ids=list(range(n_gpu)))
                ecgabd_cnn_model.eval()
                ecgabd_rnn_model = SleepNet_RNN(1280, 5, 100, 2, dropout=0, bidirectional=True)
                ecgabd_rnn_model.load_state_dict(th.load('models/LSTM_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    ecgabd_rnn_model = ecgabd_rnn_model.cuda()
                    if n_gpu>1:
                        ecgabd_rnn_model = nn.DataParallel(ecgabd_rnn_model, device_ids=list(range(n_gpu)))
                ecgabd_rnn_model.eval()
                
                # feed to ECG+ABD model
                X = [np_to_var(ecg_segs.astype('float32')),
                     np_to_var(abd_segs.astype('float32'))]
                if n_gpu>0:
                    X = [X[0].cuda(), X[1].cuda()]
                N = len(X[0])
                split_ids = np.array_split(np.arange(N), 50)
                with th.no_grad():
                    H = []
                    for si in split_ids:
                        _, H_ = ecgabd_cnn_model([X[0][si], X[1][si]])
                        H.append(H_)
                    H = th.cat(H, dim=0)
                    H = H.reshape(1, H.shape[0], -1)
                    yp, _ = ecgabd_rnn_model(H)
                yp = var_to_np(yp)[0].astype(float)
                yp = np.argmax(yp, axis=1)+1

            elif data_source=='ECG+CHEST':

                # load ECG+CHEST model
                ecgchest_cnn_model = CombinedSleepNet2([trained_model_paths[ds] for ds in data_source_split])
                ecgchest_cnn_model.load_state_dict(th.load('models/CNN_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    ecgchest_cnn_model = ecgchest_cnn_model.cuda()
                    if n_gpu>1:
                        ecgchest_cnn_model = nn.DataParallel(ecghchest_cnn_model, device_ids=list(range(n_gpu)))
                ecgchest_cnn_model.eval()
                ecgchest_rnn_model = SleepNet_RNN(1280, 5, 100, 3, dropout=0.5, bidirectional=True)
                ecgchest_rnn_model.load_state_dict(th.load('models/LSTM_%s_fold1.pth'%data_source))
                if n_gpu>0:
                    ecgchest_rnn_model = ecgchest_rnn_model.cuda()
                    if n_gpu>1:
                        ecgchest_rnn_model = nn.DataParallel(ecgchest_rnn_model, device_ids=list(range(n_gpu)))
                ecgchest_rnn_model.eval()
                
                # feed to ECG+CHEST model
                X = [np_to_var(ecg_segs.astype('float32')),
                     np_to_var(chest_segs.astype('float32'))]
                if n_gpu>0:
                    X = [X[0].cuda(), X[1].cuda()]
                N = len(X[0])
                split_ids = np.array_split(np.arange(N), 50)
                with th.no_grad():
                    H = []
                    for si in split_ids:
                        _, H_ = ecgchest_cnn_model([X[0][si], X[1][si]])
                        H.append(H_)
                    H = th.cat(H, dim=0)
                    H = H.reshape(1, H.shape[0], -1)
                    yp, _ = ecgchest_rnn_model(H)
                yp = var_to_np(yp)[0].astype(float)
                yp = np.argmax(yp, axis=1)+1

            paths.append(path)
            #ys.append(sleep_stages)
            yps.append(yp)
            #print(data_source, cohen_kappa_score(sleep_stages, yp))
            
        except Exception as ee:
            print(ee.message)
            continue
        
        # save result
        with open(output_path, 'wb') as ff:
            pickle.dump({
                'paths':paths,
                #'ys':ys,
                'yps':yps}, ff, protocol=2)

