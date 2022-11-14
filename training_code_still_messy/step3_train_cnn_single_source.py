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
    cf1 = cf1*100./cf1.sum(axis=1, keepdims=True)
    kappa1 = cohen_kappa_score(y,yp)
    
    y2 = np.array(y, copy=True); y2[y2==3]=5; y2[y2==2]=1
    yp2 = np.array(yp, copy=True); yp2[yp2==3]=5; yp2[yp2==2]=1
    cf2 = confusion_matrix(y2,yp2)
    cf2 = cf2[[2,0,1],:][:,[2,0,1]]
    cf2 = cf2*100./cf2.sum(axis=1, keepdims=True)
    kappa2 = cohen_kappa_score(y2,yp2)
    
    y3 = np.array(y, copy=True); y3[y3==3]=1; y3[y3==2]=1
    yp3 = np.array(yp, copy=True); yp3[yp3==3]=1; yp3[yp3==2]=1
    cf3 = confusion_matrix(y3,yp3)
    cf3 = cf3[[2,0,1],:][:,[2,0,1]]
    cf3 = cf3*100./cf3.sum(axis=1, keepdims=True)
    kappa3 = cohen_kappa_score(y3,yp3)
    
    return kappa1, cf1, kappa2, cf2, kappa3, cf3
    
    
if __name__=='__main__':
    
    ## config
    random_state = 10
    n_gpu = 2
    data_source = sys.argv[1]#'ABD'
    data_path = '/data/cross_channel_sleep_staging/preprocessed_data/%s_feedforward.h5'%data_source
    #label_mapping = {5:4,4:3,3:2,2:1,1:0}
    nte = 1000
    nva = 1000
    rnn_L = 9
    
    batch_size = 12#*n_gpu
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
    if data_source=='ECG':
        dall = FeedForwardRPeaks(data_path, class_weighted=True)#, select_ids=range(3))#
    else:
        dall = FeedForwardRespiration(data_path, class_weighted=True, inmemory=True)#, select_ids=range(500000))
    summarize_dataset(dall, suffix='all')
    
    # generate tr, va, te data
    trids = []; vaids = []; teids = []
    n_patients = len(dall.unique_patients)
    ntr = n_patients-nte-nva
    assert ntr>0 and nva>0 and nte>0
    
    patients_tr = pd.read_csv('training_subjects.csv').SubjectID.values
    patients_va = pd.read_csv('validation_subjects.csv').SubjectID.values
    patients_te = pd.read_csv('testing_subjects.csv').SubjectID.values
    
    """
    unique_patients = np.array(dall.unique_patients, copy=True)
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)
    
    patients_tr = unique_patients[:ntr]
    patients_va = unique_patients[ntr:ntr+nva]
    patients_te = unique_patients[ntr+nva:]
    pd.DataFrame(data={'SubjectID':patients_tr}).to_csv('training_subjects.csv',index=False)
    pd.DataFrame(data={'SubjectID':patients_va}).to_csv('validation_subjects.csv',index=False)
    pd.DataFrame(data={'SubjectID':patients_te}).to_csv('testing_subjects.csv',index=False)
    """
    
    trids = np.where(np.in1d(dall.patients, patients_tr))[0]
    vaids = np.where(np.in1d(dall.patients, patients_va))[0]
    teids = np.where(np.in1d(dall.patients, patients_te))[0]
    
    ## step1: train the CNN part

    dtr = slice_dataset(dall, trids)
    dva = slice_dataset(dall, vaids)
    #dte = slice_dataset(dall, teids)
    
    summarize_dataset(dtr, suffix='\ntr')
    summarize_dataset(dva, suffix='\nva')
    #summarize_dataset(dte, suffix='\nte')
    """
    if data_source=='ECG':
        model_cnn = ECGSleepNet()
    else:
        model_cnn = CHESTSleepNet()
    optimizer = optim.RMSprop(filter(lambda x:x.requires_grad, model_cnn.parameters()), lr=lr)#, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=2)
    model_path = 'models/current_best_model.pth'
    exp = Experiment(batch_size=batch_size, max_epoch=max_epoch, n_jobs=0,
                optimizer=optimizer, loss_function=loss_function, scheduler=scheduler,
                n_gpu=n_gpu, verbose=True)
    exp.model = model_cnn
    exp.fit(dtr, Dva=dva, Dte=dte)#, Dtrva=dataset_trva)
    exp.save(model_path)
    """
    exp = Experiment(batch_size=batch_size, loss_function=loss_function, n_jobs=0, n_gpu=n_gpu, verbose=True)
    model_path = 'models/current_best_model_%s_cnn.pth'%sys.argv[1]
    exp.load(model_path)
    model_cnn = exp.model
    print('model loaded from %s'%model_path)
    
    exp.batch_size = 160
    yptr_ff = exp.predict(dtr)
    ypva_ff = exp.predict(dva)
    #ypte_ff = exp.predict(dte)
    
    ## step2: train the overall CNN-RNN with smaller learning rate
    
    #dtr_rnn = RecurrentDataset(dtr, rnn_L)
    #dva_rnn = RecurrentDataset(dva, rnn_L)
    
    """
    if data_source=='ECG':
        # val_loss: 0.970961, current best: [epoch 1] 0.970961
        model_cnnrnn = th.load('models/current_best_model_ECG_cnnrnn_2epoch.pth')
        model_cnnrnn = model_cnnrnn.module
    else:
        model_cnnrnn = CHESTSleepNet_RNN(feedforward_model=model_cnn)
    # fix the first layer of CNN
    for pp in model_cnnrnn.feedforward.conv1.parameters():
        pp.requires_grad = False
    optimizer = optim.RMSprop([
                                {'params':filter(lambda x:x.requires_grad, model_cnnrnn.feedforward.parameters()), 'lr': lr/10.},
                                {'params':filter(lambda x:x.requires_grad, model_cnnrnn.rnn.parameters()), 'lr': lr},
                            ])#, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=2)
    model_path = 'models/current_best_model_%s_cnnrnn.pth'%sys.argv[1]
    exp = Experiment(batch_size=batch_size, max_epoch=max_epoch, n_jobs=0,
                optimizer=optimizer, loss_function=loss_function, scheduler=scheduler,
                n_gpu=n_gpu, verbose=True)
    exp.model = model_cnnrnn
    exp.fit(dtr_rnn, Dva=dva_rnn)#, Dte=dte_rnn)#, Dtrva=dataset_trva)
    exp.save(model_path)
    """
    exp = Experiment(batch_size=batch_size, loss_function=loss_function, n_jobs=0, n_gpu=n_gpu, verbose=True)
    model_path = 'models/current_best_model_ECG_cnnrnn_2epoch.pth'#'models/current_best_model_%s_cnnrnn.pth'%sys.argv[1]
    exp.load(model_path)
    print('model loaded from %s'%model_path)
    #print(exp.model)
    
    dte_rnn = RecurrentDataset(dte, rnn_L)
    
    #exp.batch_size = 1
    #yptr, Htr = exp.predict(dtr_rnn, output_id=[0,1])
    #ypva, Hva = exp.predict(dva_rnn, output_id=[0,1])
    ypte, Hte = exp.predict(dte_rnn, output_id=[0,1])
    
    #ytr = dtr.y1d
    #yva = dva.y1d
    yte = dte.y1d

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
    """
    res = {}
    yps = {}
    for jj in [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5]:
        print(jj)
        dte_jj=copy_dataset(dte)
        dte_jj.jitter=int(np.round(jj*200.))
        ypp2d=exp.predict(dte_jj)
        ypp=np.argmax(ypp2d,axis=1)+1
        yps[jj]=ypp2d
        kappa1,cf1,kappa2,cf2,kappa3,cf3=report_result(yte,ypp)
        res[jj]=[kappa1,cf1,kappa2,cf2,kappa3,cf3]
        print(jj,kappa1,kappa2,kappa3)
    """
    
    """
    te kappa = 0.412774
    0.01
    (0.01, 0.32109057199123126, 0.48885117796669275, 0.47593769110545725)
    0.02
    (0.02, 0.26752920453808648, 0.42805108565642447, 0.40713784465753611)
    0.03
    (0.03, 0.23081440759994665, 0.38811191741565987, 0.36521671112228071)
    0.04
    (0.04, 0.1993771416015504, 0.34560550027025116, 0.32649069193503566)
    0.05
    (0.05, 0.1703236950689696, 0.29734609134963375, 0.28847585038466539)
    0.1
    (0.1, 0.079658487208647877, 0.15001857843805821, 0.14053410768980179)
    0.2
    (0.2, 0.026442873734013883, 0.066379953708723627, 0.033426445722545495)
    0.3
    (0.3, 0.0156575458265511, 0.041777701542955903, 0.017868843782507304)
    0.4
    (0.4, 0.010276555264748022, 0.033717298500410386, 0.015293902795819769)
    0.5
    (0.5, 0.0093927551467152259, 0.031521196163831089, 0.015827790645976614)
    """

    #with open('results/results_%s.pickle'%data_source, 'wb') as fff:
    #    pickle.dump({
    #        #'ytr':ytr, 'yptr2d': yptr2d, 'yptr': yptr,
    #        #'yva':yva, 'ypva2d': ypva2d, 'ypva': ypva,
    #        'yte':yte, 'ypte2d': ypte2d, 'ypte': ypte,
    #        }, fff, protocol=2)
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
