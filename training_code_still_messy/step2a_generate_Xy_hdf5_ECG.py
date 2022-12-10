import datetime
from collections import Counter
from itertools import groupby
import re
import os
import os.path
import h5py
import pickle
import sys
#import subprocess
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix

network_type = 'feedforward'#recurrent
input_mat_path = '/data/cross_channel_sleep_staging/preprocessed_data/ECG'
output_h5_path = '/data/cross_channel_sleep_staging/preprocessed_data/ECG_%s.h5'%network_type
dtypef = 'float16'
dtypei = 'int32'
#cvf = 0
#random_state = 2



#def get_feedforward_data(specs, freqin, y, seg_start_pos):
    
    
if __name__=='__main__':
    chunk_size = 128  # np.random.rand(512,54000).astype(dtypef).nbytes ~~ 1MB = 1x1024x1024x8 bytes
    records = filter(lambda x:x.endswith('.mat'), os.listdir(input_mat_path))
    with h5py.File(output_h5_path, 'w') as f:
        seg_num = 0
        sparse_shape = 0
        for rid, record in enumerate(records):
            record_path = os.path.join(input_mat_path, record)
            res = sio.loadmat(record_path, variable_names=['Fs','segs','seg_start_pos','sleep_stages'])#,'seg_mask'
            Fs = res['Fs'][0,0]
            seg_start_pos = res['seg_start_pos'].flatten()
            labels = res['sleep_stages'].flatten()
            #assert np.all(np.unique(res['seg_mask']).tolist()=='normal')
            segs = res['segs'].tocoo()
            
            if network_type=='feedforward':
                X = segs
                y = labels
            elif network_type=='recurrent':
                raise ValueError(network_type)
            
            if X.shape[0]<100:
                continue
            print('[%d/%d]\t%s\t%d'%(rid+1, len(records), record.replace('Feature_',''), X.shape[0]))
                    
            if 'X' not in f:  # first write
                dtypes1 = np.array([record+'                     ']).dtype  # to be enough to put long record names
                seg_num = X.shape[0]
                sparse_shape = X.data.shape[0]
                
                gX = f.create_group('X')
                #dXdata = gX.create_dataset('data', shape=X.data.shape, maxshape=(None,),
                #                        chunks=True, dtype=dtypef)
                dXrow = gX.create_dataset('row', shape=X.row.shape, maxshape=(None,), # save as sparse marix for only R peaks
                                        chunks=True, dtype=dtypei)
                dXcol = gX.create_dataset('col', shape=X.col.shape, maxshape=(None,),
                                        chunks=True, dtype=dtypei)
                #gX.attrs['shape'] = X.shape
                #dX = f.create_dataset('X', shape=X.shape, maxshape=(None,)+X.shape[1:],
                #                        chunks=(chunk_size,)+X.shape[1:], dtype=dtypef)
                dy = f.create_dataset('y', shape=y.shape, maxshape=(None,),
                                        chunks=True, dtype=dtypef)
                drecords = f.create_dataset('record', shape=(X.shape[0],), maxshape=(None,),
                                        chunks=True, dtype=dtypes1)
                dstartpos = f.create_dataset('seg_start_pos', shape=(X.shape[0],), maxshape=(None,),
                                        chunks=True, dtype=dtypei)
                                        
                #dXdata[:] = X.data
                dXrow[:] = X.row
                dXcol[:] = X.col
                #dX[:] = X
                dy[:] = y
                dstartpos[:] = seg_start_pos
                drecords[:] = [record]*X.shape[0]
            else:
                #dXdata.resize(sparse_shape + X.data.shape[0], axis=0)
                dXrow.resize(sparse_shape + X.row.shape[0], axis=0)
                dXcol.resize(sparse_shape + X.col.shape[0], axis=0)
                #dX.resize(seg_num + X.shape[0], axis=0)
                dy.resize(seg_num + X.shape[0], axis=0)
                dstartpos.resize(seg_num + X.shape[0], axis=0)
                drecords.resize(seg_num + X.shape[0], axis=0)
                
                #dXdata[sparse_shape:] = X.data
                dXrow[sparse_shape:] = X.row+seg_num
                dXcol[sparse_shape:] = X.col
                #dX[seg_num:] = X
                dy[seg_num:] = y
                dstartpos[seg_num:] = seg_start_pos
                drecords[seg_num:] = [record]*X.shape[0]

                seg_num += X.shape[0]
                sparse_shape += X.data.shape[0]

                
