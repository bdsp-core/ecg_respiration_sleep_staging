import copy
from collections import Counter
import numpy as np
from scipy import signal
import h5py
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix, csr_matrix, load_npz
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset


def fit_one_hot_encode(x):
    _, idx = np.unique(x, return_index=True)
    unique_x = x[np.sort(idx)]
    #mapping = {zz:i for i, zz in enumerate(unique_x)}
    #x_enc1d = np.array(map(lambda i:mapping[i], x))
    ## one-hot encoding
    #encoder = LabelBinarizer().fit(x_enc1d)
    #x_enc2d = encoder.transform(x_enc1d)
    return unique_x#, mapping, encoder, x_enc1d, x_enc2d
    
    
class MySingleChannelDataset(Dataset):
    def __init__(self, input_path, select_ids=None, class_weighted=False):#, inmemory=True
        super(MySingleChannelDataset, self).__init__()
        self.input_path = input_path
        #self.inmemory = inmemory
        self.select_ids = select_ids
        self.class_weighted = class_weighted
        self.channel_num = 1
        self.len = 0
    
    #def apply_weight(self, weight_mapping):
    #    if hasattr(self, 'Z') and hasattr(self, 'y1d'):
    #        self.Z[self.y1d==ll] = weight_mapping[ll]

    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        raise NotImplementedError
        
    def set_X(self, X):
        #assert len(X)==self.len
        if hasattr(self, 'close_data_source'):
            self.close_data_source()
        self.X = X
        """
        self.select_ids = np.arange(len(self.X))
        if self.X.ndim==2:
            self.len, self.Dch = self.X.shape
            self.D = self.Dch
        elif self.X.ndim==3:
            self.len, self.time_step, self.Dch = self.X.shape
            self.D = self.Dch
        elif self.X.ndim==4:
            self.len, self.time_step, self.channel_num, self.Dch = self.X.shape
            self.D = self.Dch*self.channel_num
        """
        

class FeedForwardRPeaks(MySingleChannelDataset):
    def __init__(self, input_path, select_ids=None, class_weighted=False, jitter=None, load_x=True):#, variable_names=None
        super(FeedForwardRPeaks, self).__init__(input_path, select_ids=select_ids, class_weighted=class_weighted)
        self.jitter = jitter
        self.downsample = None#1
        
        # load into memory
        with h5py.File(self.input_path, 'r') as data_source:
            self.len = data_source['y'].shape[0]
            if self.select_ids is None:
                self.select_ids = range(self.len)
            self.select_ids = np.array(list(self.select_ids))
                
            #if variable_names is None:
            #    variable_names = data_source.keys()
            #variable_names = list(set(variable_names))
        
            #if 'y' in variable_names:
            self.y1d = data_source['y'][:]
            if len(self.select_ids)<len(self.y1d):
                self.y1d = self.y1d[self.select_ids]
            self.label_binarizer = LabelBinarizer().fit(self.y1d)
            #self.y2d = self.label_binarizer.transform(self.y1d)
            self.unique_y = self.label_binarizer.classes_
            self.K = len(self.unique_y)
            self.len = len(self.y1d)
            self.Z = np.ones(self.len)
            if self.class_weighted:
                label_counter = Counter(self.y1d)
                for ll in self.unique_y:
                    self.Z[self.y1d==ll] = 1./label_counter[ll]
                self.Z = self.Z/self.Z.mean()
                self.weight_mapping = {ll:self.Z[self.y1d==ll][0] for ll in self.unique_y}
                
            #if 'record' in variable_names:
            self.patients = data_source['record'][:]
            if len(self.select_ids)<len(self.patients):
                self.patients = self.patients[self.select_ids]
            #self.len = len(self.patients)
            self.unique_patients = fit_one_hot_encode(self.patients)
                
            #if 'seg_start_pos' in variable_names:
            self.seg_start_pos = data_source['seg_start_pos'][:]
            if len(self.select_ids)<len(self.seg_start_pos):
                self.seg_start_pos = self.seg_start_pos[self.select_ids]
            #self.len = len(self.seg_start_pos)

            self.D = 54000
            if self.downsample is not None:
                self.D = self.D//self.downsample
            if load_x:
                """
                #if 'X' in variable_names:
                Xrow = data_source['X/row'][:]
                Xcol = data_source['X/col'][:]
                #self.len = len(self.X)
                self.X = coo_matrix((np.ones_like(Xrow), (Xrow, Xcol)), shape=(data_source['y'].shape[0], self.D))
                #self.X = coo_matrix((np.random.rand(self.y1d.shape[0], self.D)>0.5).astype(int))
                #self.X = self.X.tocsc()
                self.X = self.X.tocsr()
                """
                self.X = load_npz('/data/cross_channel_sleep_staging/preprocessed_data/ECG_feedforward_csr.npz')
            #self.D = 5400
        
    def __getitem__(self, id_):
        idx = self.select_ids[id_]
        X = self.X[idx]
        if type(X)==csr_matrix:
            X = X.toarray()
            if self.downsample is not None:
                ids = list(np.where(X==1))
                ids[1] = np.floor(ids[1]*1./self.downsample).astype(int)
                X = np.zeros((X.shape[0], X.shape[1]//self.downsample))
                X[ids] = 1.
                
            if type(id_)!=int:
                X = X.reshape(X.shape[0], self.channel_num, -1)
            else:
                X = X.reshape(self.channel_num, -1)
        
            if self.jitter is not None:
                Fs = 200  # [Hz] ######TODO
                x1, x2 = np.where(X==1)
                #x2 = np.round(x2/10.).astype(int) #downsample 10 times
                x2 = x2+np.round(np.random.randn(len(x2))*self.jitter*Fs).astype(int)
                X = np.zeros((X.shape[0], X.shape[1]))#//10
                x2 = np.clip(x2, 0, X.shape[1]-1)
                X[x1,x2] = 1.
            
        y = self.y1d[id_]-1
        Z = self.Z[id_]#.astype('float32')
            
        return {'X': X.astype('float32'), 'Z':Z.astype('float32'), 'y': y.astype('float32')}
        
        
class FeedForwardRespiration(MySingleChannelDataset):
    def __init__(self, input_path, inmemory=True, select_ids=None, class_weighted=False, noise_std=None, load_x=True):#, variable_names=None
        super(FeedForwardRespiration, self).__init__(input_path, select_ids=select_ids, class_weighted=class_weighted)
        self.inmemory = inmemory
        self.noise_std = noise_std
        
        # load into memory
        with h5py.File(self.input_path, 'r') as data_source:
            self.len = data_source['y'].shape[0]
            if self.select_ids is None:
                self.select_ids = range(self.len)
            self.select_ids = np.array(list(self.select_ids))
                
            #if variable_names is None:
            #    variable_names = data_source.keys()
            #variable_names = list(set(variable_names))
        
            #if 'y' in variable_names:
            self.y1d = data_source['y'][:]
            if len(self.select_ids)<len(self.y1d):
                self.y1d = self.y1d[self.select_ids]
            self.label_binarizer = LabelBinarizer().fit(self.y1d)
            #self.y2d = self.label_binarizer.transform(self.y1d)
            self.unique_y = self.label_binarizer.classes_
            self.K = len(self.unique_y)
            self.len = len(self.y1d)
            self.Z = np.ones(self.len)
            if self.class_weighted:
                label_counter = Counter(self.y1d)
                for ll in self.unique_y:
                    self.Z[self.y1d==ll] = 1./label_counter[ll]
                self.Z = self.Z/self.Z.mean()
                self.weight_mapping = {ll:self.Z[self.y1d==ll][0] for ll in self.unique_y}
                
            #if 'record' in variable_names:
            self.patients = data_source['record'][:]
            if len(self.select_ids)<len(self.patients):
                self.patients = self.patients[self.select_ids]
            #self.len = len(self.patients)
            self.unique_patients = fit_one_hot_encode(self.patients)
                
            #if 'seg_start_pos' in variable_names:
            self.seg_start_pos = data_source['seg_start_pos'][:]
            if len(self.select_ids)<len(self.seg_start_pos):
                self.seg_start_pos = self.seg_start_pos[self.select_ids]
            #self.len = len(self.seg_start_pos)

        with h5py.File(self.input_path, 'r') as data_source:
            #if 'X' in variable_names:
            self.D = data_source['X'].shape[-1]
        if load_x:
            if self.inmemory:
                with h5py.File(self.input_path, 'r') as data_source:
                    #if 'X' in variable_names:
                    self.X = data_source['X'][:]
            else:
                self.data_source = h5py.File(self.input_path, 'r')
                self.X = self.data_source['X']
    
    def close_data_source(self):
        if hasattr(self, 'data_source'):
            self.data_source.close()
        
    def __getitem__(self, id_):
        idx = self.select_ids[id_]
        X = self.X[idx]
        
        if self.noise_std is not None:
            noise = np.random.randn(*X.shape)*self.noise_std
            Fs = 10  # [Hz] ######TODO
            size = Fs*1 # window 1s
            size = size//2*2+1 # make sure odd
            polyorder = size//2
            noise = savgol_filter(noise, size, polyorder, axis=-1)
            X = X+noise
                
        y = self.y1d[id_]-1
        Z = self.Z[id_]#.astype('float32')
            
        return {'X': X.astype('float32'), 'Z':Z.astype('float32'), 'y': y.astype('float32')}


class CombinedDataset(Dataset):
    def __init__(self, dataset_names, datasets, combine_keys, class_weighted=False):
        super(CombinedDataset, self).__init__()
        self.dataset_names = dataset_names
        self.datasets = datasets
        assert type(self.datasets[0])==FeedForwardRPeaks and type(self.datasets[1])==FeedForwardRespiration, 'Currently only support ECG+ABD or ECG+CHEST'
        assert len(self.dataset_names)==len(self.datasets) and len(self.datasets)>0
        #try: iter(combine_keys) except TypeError: combine_keys = [combine_keys]
        self.combine_keys = combine_keys
        self.key_sep = '*'
        self.class_weighted = class_weighted
            
        # slice all datasets to have same ids
        alldataset_keys = []  # a list of keys from all datasets
        for i in range(len(self.datasets)):
        
            # construct key for this dataset
            for ki, k in enumerate(self.combine_keys):
                key = getattr(self.datasets[i], k)
                if ki==0:
                    this_key = key
                else:
                    this_key = map(lambda x:this_key[x]+self.key_sep+str(key[x]), range(len(key)))
            alldataset_keys.append(this_key)
            
            if i==0:
                self.common_keys = set(this_key)
            else:
                self.common_keys = self.common_keys & set(this_key)
        self.common_keys = list(self.common_keys)
        self.len = len(self.common_keys)
        
        for i in range(len(self.datasets)):
            this_slice = np.where(np.in1d(alldataset_keys[i], self.common_keys))[0]
            if len(this_slice)<len(self.datasets[i]):
                self.datasets[i] = slice_dataset(self.datasets[i], this_slice)
            #self.datasets[i].select_ids = this_slice
        
        self.seg_start_pos = np.array(self.datasets[0].seg_start_pos, copy=True)
        self.patients = np.array(self.datasets[0].patients, copy=True)
        self.unique_patients = np.array(self.datasets[0].unique_patients, copy=True)
        self.y1d = np.array(self.datasets[0].y1d, copy=True)
        #self.y2d = np.array(self.datasets[0].y2d, copy=True)
        self.label_binarizer = LabelBinarizer().fit(self.y1d)
        #self.y2d = self.label_binarizer.transform(self.y1d)
        self.unique_y = self.label_binarizer.classes_
        #self.Z = (self.datasets[0].Z+self.datasets[1].Z)/2.
        self.Z = np.ones(self.len)
        if self.class_weighted:
            label_counter = Counter(self.y1d)
            for ll in self.unique_y:
                self.Z[self.y1d==ll] = 1./label_counter[ll]
            self.Z = self.Z/self.Z.mean()
            self.weight_mapping = {ll:self.Z[self.y1d==ll][0] for ll in self.unique_y}
        self.channel_num = sum([self.datasets[i].channel_num for i in range(len(self.datasets))])
        
    def close_data_source(self):
        for i in range(len(self.datasets)):
            if hasattr(self.datasets[i], 'data_source'):
                self.datasets[i].data_source.close()
    
    def to_single_channel_dataset(self, dataset_type=FeedForwardRespiration):
        i = 0 if dataset_type==FeedForwardRPeaks else 1
        dataset = self.datasets[i]
        dataset.seg_start_pos = self.seg_start_pos
        dataset.patients = self.patients
        dataset.unique_patients = self.unique_patients
        dataset.y1d = self.y1d
        dataset.label_binarizer = self.label_binarizer
        dataset.unique_y = self.unique_y
        dataset.Z = self.Z
        dataset.weight_mapping = self.weight_mapping
        dataset.channel_num = self.channel_num
        return dataset
            
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        """
        res = {}
        for i in range(len(self.datasets)):
            this_res = self.datasets[i][idx]
            for k in this_res:
                # rename keys to dataset_name+key
                res[k+self.key_sep+self.dataset_names[i]] = this_res[k]
        """
        #TODO now assume ECG+CHEST/ABD
        res1 = self.datasets[0][idx]
        res2 = self.datasets[1][idx]
        #assert res1['y']==res2['y']
        
        y = self.y1d[idx]-1
        Z = self.Z[idx]
        
        res = {'X1': res1['X'], 'X2':res2['X'],
               'y': y.astype('float32'),
               'Z': Z.astype('float32')}
            
        return res


class RecurrentDataset(Dataset):
    """
    Take a dataset, change its __getitem__()
    """
    def __init__(self, dataset, L):
        super(RecurrentDataset, self).__init__()
        self.dataset = dataset
        self.L = L
        
        # generate self.start_ids
        self.start_ids = []
        self.y1d = []
        #self.y2d = []
        self.Z = []
        self.patients = []
        self.seg_start_pos = []
        for patient in self.dataset.unique_patients:
            this_ids = np.where(self.dataset.patients==patient)[0]
            assert np.all(np.diff(self.dataset.seg_start_pos[this_ids])>0)
            if self.L<=0:
                # self.L<=0 ==> take L==length, can have different lengths for different subjects
                start_ids = [this_ids.tolist()]
            else:
                # get all possible start positions
                start_ids = np.arange(0, len(this_ids)-self.L+1, 1)
                start_ids = this_ids[np.array(map(lambda x:np.arange(x, x+self.L), start_ids))]
                seg_pos = self.dataset.seg_start_pos[start_ids]
                start_ids = start_ids[(seg_pos[:,-1]-seg_pos[:,0])/6000.-self.L+1<self.L*0.2]  # number of skipped steps < 20% of L
                if len(start_ids)<=0:
                    continue
                
            self.start_ids.extend(start_ids)
            self.y1d.extend(self.dataset.y1d[start_ids])
            # self.y2d.extend(self.dataset.y2d[start_ids])
            self.patients.extend([patient]*len(start_ids))
            self.seg_start_pos.extend(self.dataset.seg_start_pos[start_ids])

        if self.L>0:
            self.start_ids = np.array(self.start_ids)
        self.y1d = np.array(self.y1d)
        self.patients = np.array(self.patients)
        self.seg_start_pos = np.array(self.seg_start_pos)
        
        self.len = len(self.start_ids)
        self.Z = np.ones(self.len)
        self.K = self.dataset.K
        self.unique_patients = fit_one_hot_encode(self.patients)
        self.weight_mapping = self.dataset.weight_mapping
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        start_id = self.start_ids[idx]
        res = self.dataset.__getitem__(start_id)
        
        return res
        

def summarize_dataset(dataset, suffix='', items=None):
    print(suffix)
    if items is None:
        items = ['unique_patients','samples','labels','Z']
    if 'unique_patients' in items and hasattr(dataset, 'unique_patients'):
        patient_num = len(dataset.unique_patients)
        print('patient number %d'%patient_num)
    if 'samples' in items:
        print('sample number %d'%len(dataset))
    """
    if 'labels' in items:
        print('labels %s'%Counter(dataset.y1d))
    if 'Z' in items:
        print('Zs %s'%Counter(dataset.Z.flatten()))
    """
    
    
def copy_dataset(dataset):
    if type(dataset)==FeedForwardRPeaks:
        newdataset = FeedForwardRPeaks(copy.deepcopy(dataset.input_path),
                        #inmemory=copy.deepcopy(dataset.inmemory),
                        select_ids=copy.deepcopy(dataset.select_ids),
                        class_weighted=copy.deepcopy(dataset.class_weighted),
                        jitter=copy.deepcopy(dataset.jitter),
                        load_x=False)#variable_names=None
        newdataset.D = copy.deepcopy(dataset.D)
                        
    elif type(dataset)==FeedForwardRespiration:
        newdataset = FeedForwardRespiration(copy.deepcopy(dataset.input_path),
                        inmemory=copy.deepcopy(dataset.inmemory),
                        select_ids=copy.deepcopy(dataset.select_ids),
                        class_weighted=copy.deepcopy(dataset.class_weighted),
                        load_x=False)#variable_names=None,
        newdataset.D = copy.deepcopy(dataset.D)
    
    elif type(dataset)==CombinedDataset:
        newdataset = CombinedDataset(copy.deepcopy(dataset.dataset_names),
                        [copy_dataset(dataset.datasets[i]) for i in range(len(dataset.datasets))],
                        copy.deepcopy(dataset.combine_keys),
                        class_weighted=copy.deepcopy(dataset.class_weighted))
    
    else:
        raise NotImplementedError
                        
    newdataset.len = copy.deepcopy(dataset.len)
    newdataset.channel_num = copy.deepcopy(dataset.channel_num)
    if hasattr(dataset, 'weight_mapping'):
        newdataset.weight_mapping = copy.deepcopy(dataset.weight_mapping)

    if hasattr(dataset, 'X'):
        if type(dataset)==FeedForwardRPeaks:
            newdataset.X = dataset.X#.copy()#TODO
        elif type(dataset)==FeedForwardRespiration:
            if dataset.inmemory:
                #newdataset.X = np.array(dataset.X, copy=True)
                newdataset.X = dataset.X#.copy()#TODO
            else:
                newdataset.data_source = h5py.File(dataset.input_path, 'r')
                newdataset.X = newdataset.data_source['X']
    """
    if hasattr(dataset, 'y1d'):
        newdataset.y1d = np.array(dataset.y1d, copy=True)
        newdataset.K = copy.deepcopy(dataset.K)

    if hasattr(dataset, 'y2d'):
        newdataset.y2d = np.array(dataset.y2d, copy=True)

    if hasattr(dataset, 'Z'):
        newdataset.Z = np.array(dataset.Z, copy=True)
        
    if hasattr(dataset, 'patients'):
        newdataset.patients = np.array(dataset.patients, copy=True)
        newdataset.unique_patients = np.array(dataset.unique_patients, copy=True)
        #newdataset.patients_mapping = np.array(dataset.patients_mapping, copy=True)
        #newdataset.patients_encoder = np.array(dataset.patients_encoder, copy=True)
        #newdataset.patients_enc1d = np.array(dataset.patients_enc1d, copy=True)
        #newdataset.patients_enc2d = np.array(dataset.patients_enc2d, copy=True)

    if hasattr(dataset, 'seg_start_pos'):
        newdataset.seg_start_pos = np.array(dataset.seg_start_pos, copy=True)
    """
    return newdataset
    

def slice_dataset(dataset, ids, tocopy=True):
    assert len(ids)==len(np.unique(ids))
    
    if tocopy:
        newdataset = copy_dataset(dataset)
    else:
        newdataset = dataset
        
    if hasattr(dataset, 'y1d'):
        newdataset.y1d = dataset.y1d[ids]
        newdataset.K = len(np.unique(newdataset.y1d))
        newdataset.len = newdataset.y1d.shape[0]

    #if hasattr(dataset, 'y2d'):
    #    newdataset.y2d = dataset.y2d[ids]

    if hasattr(dataset, 'Z'):
        newdataset.Z = dataset.Z[ids]
        
    if hasattr(dataset, 'patients'):
        newdataset.patients = dataset.patients[ids]
        newdataset.unique_patients = fit_one_hot_encode(newdataset.patients)

    if hasattr(dataset, 'seg_start_pos'):
        newdataset.seg_start_pos = dataset.seg_start_pos[ids]

    #if hasattr(dataset, 'X'):
    #    newdataset.X = dataset.X[ids]

    if type(dataset)!=CombinedDataset:
        newdataset.select_ids = dataset.select_ids[ids]
    else:
        for i in range(len(dataset.datasets)):
            newdataset.datasets[i].select_ids = dataset.datasets[i].select_ids[ids]

    return newdataset
    
