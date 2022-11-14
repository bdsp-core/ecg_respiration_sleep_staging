import pickle
import sys
import torch as th
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from braindecode.torch_ext.util import np_to_var, var_to_np
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
#sys.path.insert(0, r'myml_lib')
from dataset import *
from experiment import *
from mymodel import *        
    

def report_result(y, yp):
    cf1 = confusion_matrix(y,yp)
    cf1 = cf1[[4,2,1,0,3],:][:,[4,2,1,0,3]]
    kappa1 = cohen_kappa_score(y,yp)
    
    y2 = np.array(y, copy=True); y2[y2==3]=5; y2[y2==2]=1
    yp2 = np.array(yp, copy=True); yp2[yp2==3]=5; yp2[yp2==2]=1
    cf2 = confusion_matrix(y2,yp2)
    cf2 = cf2[[2,0,1],:][:,[2,0,1]]
    kappa2 = cohen_kappa_score(y2,yp2)
    
    y3 = np.array(y, copy=True); y3[y3==3]=1; y3[y3==2]=1
    yp3 = np.array(yp, copy=True); yp3[yp3==3]=1; yp3[yp3==2]=1
    cf3 = confusion_matrix(y3,yp3)
    cf3 = cf3[[2,0,1],:][:,[2,0,1]]
    kappa3 = cohen_kappa_score(y3,yp3)
    
    return kappa1, cf1, kappa2, cf2, kappa3, cf3
    
    
if __name__=='__main__':
    
    ## config
    random_state = 10
    n_gpu = 2
    
    data_source = sys.argv[1]#'ECG+ABD'
    assert '+' in data_source, 'must be x+y'
    data_sources = data_source.split('+')
    data_paths = ['/data/cross_channel_sleep_staging/preprocessed_data/%s_feedforward.h5'%ds for ds in data_sources]
    nte = 1000
    nva = 1000
    
    batch_size = 32#*n_gpu
    lr = 0.001#*n_gpu
    max_epoch = 10
    loss_function = 'ce'

    ## set random seed
    
    np.random.seed(random_state+10)
    th.manual_seed(random_state)
    if n_gpu>0 and th.cuda.is_available():
        th.cuda.manual_seed(random_state)
        th.cuda.manual_seed_all(random_state)
    else:
        n_gpu = 0

    ## read all data
    datasets = []
    for i, ds in enumerate(data_sources):
        if ds=='ECG':
            datasets.append(FeedForwardRPeaks(data_paths[i]))
        else:
            datasets.append(FeedForwardRespiration(data_paths[i], inmemory=False))#, select_ids=range(50000)
            datasets[-1].close_data_source()
        summarize_dataset(datasets[-1], suffix='%s all'%ds)
        
    dall = CombinedDataset(data_sources, datasets, ['patients', 'seg_start_pos'], class_weighted=True)
    dall.close_data_source()
    summarize_dataset(dall, suffix='all')
    
    patients_tr = pd.read_csv('training_subjects.csv').SubjectID.values
    patients_va = pd.read_csv('validation_subjects.csv').SubjectID.values
    patients_te = pd.read_csv('testing_subjects.csv').SubjectID.values
    
    dtr = slice_dataset(dall, np.where(np.in1d(dall.patients, patients_tr))[0])
    dva = slice_dataset(dall, np.where(np.in1d(dall.patients, patients_va))[0])
    
    summarize_dataset(dtr, suffix='\ntr')
    summarize_dataset(dva, suffix='\nva')
    
    pretrained_models = {'ECG':'models/current_best_model_ECG_cnn.pth',
                       'ABD':'models/current_best_model_ABD_cnn.pth',
                       'CHEST':'models/current_best_model_CHEST_cnn.pth'
                       }
    model = CombinedSleepNet2([pretrained_models[ds] for ds in data_sources])
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)#, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=2)
    model_path = 'models/current_best_model.pth'
    exp = Experiment(model=model, batch_size=batch_size, max_epoch=max_epoch, n_jobs=0,
                optimizer=optimizer, loss_function=loss_function, scheduler=scheduler,
                n_gpu=n_gpu, verbose=True)
    exp.fit(dtr, Dva=dva)
    exp.save(model_path)
    #exp.load(model_path)
    #print('model loaded from %s'%model_path)
    
    dte = slice_dataset(dall, np.where(np.in1d(dall.patients, patients_te))[0])
    summarize_dataset(dte, suffix='\nte')
    
    #ytr = dtr.y1d
    #yva = dva.y1d
    yte = dte.y1d
    #yptr2d = exp.predict(dtr)
    #ypva2d = exp.predict(dva)
    ypte2d = exp.predict(dte)

    #yptr = np.argmax(yptr2d, axis=1)+1
    #ypva = np.argmax(ypva2d, axis=1)+1
    ypte = np.argmax(ypte2d, axis=1)+1

    #kappa_tr = cohen_kappa_score(ytr, yptr); cf_tr = confusion_matrix(ytr, yptr)
    #kappa_va = cohen_kappa_score(yva, ypva); cf_va = confusion_matrix(yva, ypva)
    kappa_te = cohen_kappa_score(yte, ypte); cf_te = confusion_matrix(yte, ypte)
    
    #print('tr kappa = %g'%kappa_tr)
    #print('va kappa = %g'%kappa_va)
    print('te kappa = %g'%kappa_te)
    
    #kappa1_tr, cf1_tr, kappa2_tr, cf2_tr, kappa3_tr, cf3_tr = report_result(ytr, yptr)
    #kappa1_va, cf1_va, kappa2_va, cf2_va, kappa3_va, cf3_va = report_result(yva, ypva)
    kappa1_te, cf1_te, kappa2_te, cf2_te, kappa3_te, cf3_te = report_result(yte, ypte)
    print(kappa1_te, cf1_te)
    print(kappa2_te, cf2_te)
    print(kappa3_te, cf3_te)
    
    dtr.close_data_source()
    dva.close_data_source()
    dte.close_data_source()

    with open('results/results_%s.pickle'%data_source, 'wb') as fff:
        pickle.dump({
            #'ytr':ytr, 'yptr2d': yptr2d, 'yptr': yptr,
            #'yva':yva, 'ypva2d': ypva2d, 'ypva': ypva,
            'yte':yte, 'ypte2d': ypte2d, 'ypte': ypte,
            }, fff, protocol=2)
    #plt.close();fig=plt.figure();ax=fig.add_subplot(111);ax.plot(jj,ks1,c='r',label='all 5 sleep stages',marker='.',ms=10,alpha=0.5,lw=2);ax.plot(jj,ks2,c='g',label='W+N1 vs N2+N3 vs R',marker='*',ms=8,alpha=0.5,lw=2);ax.plot(jj,ks3,c='b',label='W vs NR vs R',marker='d',ms=6,lw=2,alpha=0.5);ax.set_xlabel('Standard deviation of R peak jitter (s)');ax.set_ylabel('Cohen\'s kappa on testing set');ax.legend(frameon=False);seaborn.despine();plt.tight_layout();plt.show()
    
    #plt.close();fig=plt.figure();ax=fig.add_subplot(311);ax.hist(kappas_tr,bins=50,color='r',alpha=0.5);ax.set_xlim([-0.25,1]);ax.set_ylabel('count');seaborn.despine();;ax=fig.add_subplot(312);ax.hist(kappas_va,bins=50,color='g',alpha=0.5);ax.set_xlim([-0.25,1]);ax.set_ylabel('count');seaborn.despine();ax=fig.add_subplot(313);ax.hist(kappas_te,bins=50,color='b',alpha=0.5);ax.set_xlim([-0.25,1]);seaborn.despine();ax.set_ylabel('count');ax.set_xlabel('Cohen\'s kappa');;plt.tight_layout();plt.show()

    hte = exp.predict(dte, output_id=1, return_only_loss=False, concatenate=True, apply_activation=False, use_gpu=None)
    hte = hte.astype(float)
    ids = np.sort(np.random.choice(np.arange(len(hte)),len(hte)//10,replace=False))
    hte2 = hte[ids]
    
    from MulticoreTSNE import MulticoreTSNE as TSNE
    tsne = TSNE(n_jobs=-1)
    hte_tsne=tsne.fit_transform(hte2)
    colors = 'rgbym'
    labels=['N3','N2','N1','R','W']
    plt.close()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for ii in range(5):
        ax.scatter(hte_tsne[dte.y1d[ids]==ii+1,0],hte_tsne[dte.y1d[ids]==ii+1,1],s=6,c=colors[ii],alpha=0.1)
        ax.scatter([10000],[10000],s=12,c=colors[ii],alpha=1,label=labels[ii])
    ax.set_xlim([hte_tsne[:,0].min(),hte_tsne[:,0].max()])
    ax.set_ylim([hte_tsne[:,1].min(),hte_tsne[:,1].max()])
    ax.axis('off')
    ax.legend(frameon=False, scatterpoints=3)
    plt.tight_layout()
    plt.show()
    
    ww=var_to_np(exp.model.conv1.weight).astype(float)
    fig=plt.figure()
    for i in range(8):
        for j in range(8):
            ax=fig.add_subplot(8,8,i*8+j+1)
            ax.plot(ww[i*8+j],c='b')
            ax.plot([0,16],[0,0],c='k',alpha=0.5)
            ax.set_xlim([0,16])
            ax.set_ylim([-max(abs(ww.max()),abs(ww.min())),max(abs(ww.max()),abs(ww.min()))])
            ax.axis('off')
    plt.tight_layout()
    plt.show()
