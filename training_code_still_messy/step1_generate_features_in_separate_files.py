from collections import Counter
from itertools import combinations
import datetime
import os
import os.path
#import pickle
import sys
#import subprocess
import numpy as np
from scipy import io as sio
#from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')
#from kw_dunn import *
from load_mgh_sleep_dataset import *
from segment_signal import *


window_time = 4.5*60 # [s]
window_step = 30 # [s]
start_end_remove_epoch_num = 1
amplitude_thres = {'ECG':6000,'CHEST':6000,'ABD':6000}[sys.argv[1]]
channels = [sys.argv[1]]
#line_freq = [50., 60.]  # [Hz]  # mutli-center study, not sure 50 or 60?
chest_bandpass_freq = [None, 4.]  # [Hz]
all_sleep_stages = np.array(['N3','N2','N1','R','W'])
all_sleep_stages_num = np.array([1,2,3,4,5])
random_state = 2
n_jobs = -1


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
    
    
def myprint(seg_mask):
    sm = Counter(map(lambda x:x.split('_')[0], seg_mask))
    for ex in seg_mask_explanation:
        if ex in sm:
            print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))
    
    
if __name__=='__main__':   # python __file__ ECG/CHEST/ABD
    normal_only = True
    np.random.seed(random_state)
    ##################
    # use data_list_paths to specify which dataset to use
    # data_list.txt:
    # data_path    label_path   feature_path    state
    ##################
    data_list_paths = ['data/%s/data_list.txt'%sys.argv[1]]
    subject_files = np.zeros((0,3))
    for data_list_path in data_list_paths:
        subject_files_ = np.loadtxt(data_list_path,dtype=str,delimiter='\t',skiprows=1)
        good_ids = filter(lambda i:subject_files_[i,3].startswith('good'), range(len(subject_files_)))
        subject_files = np.r_[subject_files,subject_files_[good_ids,:3]]
    subject_num = subject_files.shape[0]

    features = []
    labels = []
    unique_subjects = []
    subjects = []
    seg_start_pos = []
    #full_subjects = []
    #full_sleep_stages = []
    seg_masks = []
    subject_ages = []
    subject_err_path = 'data/%s/err_subject_reason.txt'%sys.argv[1]
    if os.path.isfile(subject_err_path):
        err_subject_reason = []
        with open(subject_err_path,'r') as f:
            for row in f:
                if row.strip()=='':
                    continue
                i = row.split(':::')
                err_subject_reason.append([i[0].strip(), i[1].strip()])
        err_subject = [i[0] for i in err_subject_reason]
    else:
        err_subject_reason = []
        err_subject = []
        
    for si in range(subject_num):
        data_path = subject_files[si,0]
        label_path = subject_files[si,1]
        feature_path = subject_files[si,2]
        #age = int(subject_files[si,3])
        subject_file_name = os.path.basename(feature_path)
        if subject_file_name in err_subject:
            continue
        unique_subjects.append(subject_file_name)
        if os.path.isfile(feature_path):
            print('\n====== [(%d)/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))

        else:
            print('\n====== [%d/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
            try:
                if sys.argv[1]=='ECG':
                    reverse_sign = True
                else:
                    reverse_sign = False
                signal_raw, sleep_stages_, params = check_load_Twin_dataset(data_path, label_path, channels=channels, reverse_sign=reverse_sign)
                if signal_raw.shape[0]!=1:
                    raise ValueError('Requires 1 signal channel.')
                Fs = params.get('Fs')
                if Fs!=200:
                    raise ValueError('Spurious Fs at %gHz'%Fs)
                
                # segment signal
                if sys.argv[1] == 'ECG':
                    segs_, sleep_stages_, seg_start_pos_, seg_mask = segment_ecg_signal(signal_raw, sleep_stages_, window_time, window_step, Fs,
                            to_remove_mean=False, amplitude_thres=amplitude_thres, n_jobs=n_jobs)
                elif sys.argv[1] in ['CHEST', 'ABD']:
                    newFs = 10
                    segs_, sleep_stages_, seg_start_pos_, seg_mask = segment_chest_signal(signal_raw, sleep_stages_, window_time, window_step, Fs,
                            newFs=newFs, bandpass_freq=chest_bandpass_freq, amplitude_thres=amplitude_thres,
                            to_remove_mean=False, n_jobs=n_jobs)
                    Fs = newFs
                else:
                    raise NotImplementedError(sys.argv[1])
                if segs_.shape[0] <= 0:
                    raise ValueError('No segments')
                if Counter(seg_mask)[seg_mask_explanation[0]] <= 0:
                    raise ValueError('No normal segments')

                if normal_only:
                    good_ids = np.where(np.in1d(seg_mask,seg_mask_explanation[:2]))[0]
                    if len(good_ids)<=0:
                        myprint(seg_mask)
                        raise ValueError('No normal segments')
                    segs_ = segs_[good_ids]
                    sleep_stages_ = sleep_stages_[good_ids]
                    seg_start_pos_ = seg_start_pos_[good_ids]
                else:
                    good_ids = np.arange(len(seg_mask))
                
                myprint(seg_mask)
                """
                stage_ids = np.where(np.in1d(all_sleep_stages_num, sleep_stages_))
                all_sleep_stages_num_ = all_sleep_stages_num[stage_ids]
                all_sleep_stages_ = all_sleep_stages[stage_ids]

                std_rri = np.array([np.std(np.diff(np.where(segs_[iii,0]==1)[0]))/Fs for iii in range(len(segs_))])
                kwdnn_res = kw_dunn([std_rri[sleep_stages_==ii] for ii in all_sleep_stages_num_])

                sm = ['(%s, %s) *'%jj for ii,jj in enumerate(combinations(all_sleep_stages_,2)) if kwdnn_res[-1][ii]]
                nmi = normalized_mutual_info_score(sleep_stages_, std_rri)
                corr = spearmanr(sleep_stages_, std_rri)

                fig = plt.figure(figsize=(7,5))
                ax = fig.add_subplot(111)
                seaborn.violinplot(x=sleep_stages_,y=std_rri,palette='Set2',ax=ax)
                ax.text(0.05,0.99,'\n'.join(sm),ha='left',va='top',transform=ax.transAxes)
                ax.text(0.25,0.95,'Spearman R = %g (p-val = %g)\nNMI = %g'%(corr[0],corr[1],nmi),ha='left',va='top',transform=ax.transAxes)
                ax.yaxis.grid(True)
                ax.set_ylabel('Stdev of RRI (s) in 4.5min windows')
                ax.set_xticklabels(all_sleep_stages_)
                ax.set_title(subject_file_name.replace('Feature_','').replace('.mat',''))
                seaborn.despine(offset=5, trim=True)
                plt.tight_layout()
                #plt.show()
                plt.savefig('RRI_stats/%s.png'%subject_file_name.replace('Feature_','').replace('.mat',''), bbox_inches='tight',pad_inches=0.05)
                """
                if sys.argv[1]=='ECG':
                    segs_ = csr_matrix(segs_.squeeze())
                
            except Exception as e:
                err_info = e.message.split('\n')[0].strip()
                print('\n%s.\nSubject %s is IGNORED.\n'%(err_info,subject_file_name))
                err_subject_reason.append([subject_file_name,err_info])
                err_subject.append(subject_file_name)

                with open(subject_err_path,'a') as f:
                    msg_ = '%s::: %s\n'%(subject_file_name,err_info)
                    f.write(msg_)
                continue
                
            sio.savemat(feature_path, {
                'segs':segs_,
                #'specs':specs_,
                #'frequency':freq,
                'sleep_stages':sleep_stages_,
                'seg_start_pos':seg_start_pos_,
                #'age':age,
                'seg_mask':seg_mask,
                'Fs':Fs,}, do_compression=True)
                #'subject':subject
                #'gender':gender,

