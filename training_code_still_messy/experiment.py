import os
import os.path
import datetime
import time
import timeit
import multiprocessing
from collections import OrderedDict
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.exceptions import NotFittedError
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from braindecode.torch_ext.util import np_to_var, var_to_np
from torch.utils.data import DataLoader
from utils import softmax


class Experiment(object):
    def __init__(self, model=None, batch_size=32, max_epoch=10, n_jobs=1, label_smoothing_amount=None,
            optimizer=None, loss_function=None, scheduler=None, remember_best_metric='loss',
            verbose=False, n_gpu=0, save_base_path='models'):#, model_constraint=None
        self.model = model
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.n_jobs = n_jobs
        self.label_smoothing_amount = label_smoothing_amount
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.remember_best_metric = remember_best_metric
        self.verbose = verbose
        self.n_gpu = n_gpu
        #self.model_constraint = model_constraint
        self.save_base_path = save_base_path
    
    def run_one_epoch(self, epoch, gen, mode, use_gpu=False):#, evaluate_loss=False
        if use_gpu:
            th.cuda.empty_cache()
            time.sleep(0.1)
        if mode=='train':
            running_loss = 0.
            verbosity = 100
            self.model.train()
        else:
            total_loss = 0.
            total_outputs = []
            total_Hs = []
            self.model.eval()
        
        weights = np_to_var(np.array([gen.dataset.weight_mapping[kkk] for kkk in range(1,6)], dtype='float32'))#TODO range(1,6)
        if use_gpu:
            weights = weights.cuda()
            
        #self.batch_loss_history = []
        N = 0.
        for bi, batch in enumerate(gen):
            if 'X' in batch:
                X = Variable(batch['X'])
            elif 'X1' in batch and 'X2' in batch:  #TODO assume X1 and X2
                X = [Variable(batch['X1']), Variable(batch['X2'])]
            else:
                raise NotImplementedError
            Z = Variable(batch['Z'])#.numpy()
            #Z = Variable(batch['Z']).view(-1,1)
            y = Variable(batch['y']).long()#.numpy()
            #if self.label_smoothing_amount is not None:
            #    ynp = np.clip(ynp, self.label_smoothing_amount/(self.model.K-1), 1.-self.label_smoothing_amount)
            #y = Variable(batch['y']).view(-1,1)
            batch_size = len(y)#np)
            N += batch_size
            
            if use_gpu:
                if type(X)==list:
                    X = [X[xi].cuda() for xi in range(len(X))]
                else:
                    X = X.cuda()
                y = y.cuda()
                Z = Z.cuda()
            
            if mode=='train':
                self.optimizer.zero_grad()

            output, H = self.model(X)
            is_rnn_output = len(output.shape)==3

            if mode=='train':
            
                if is_rnn_output:
                    tmpsize = len(output)
                    output = output.view(-1, output.shape[-1])
                    y = y.view(-1)
                    
                #if self.loss_function=='mse':
                #    #loss = th.pow(y - output, 2)
                #    loss = F.mse_loss(output, y, reduce=False)
                #    loss = th.mean(loss*Z)
                #elif self.loss_function=='bce':
                #    loss = F.binary_cross_entropy_with_logits(output, y, weight=Z, size_average=True)
                #    #loss = th.mean(loss*Z)
                if self.loss_function=='ce':
                    loss = F.cross_entropy(output, y, weight=weights)#, reduce=False)
                    #loss = th.mean(loss*Z)
                #elif self.loss_function=='huber':
                #    loss = F.smooth_l1_loss(output, y, reduce=False)
                #    loss = th.mean(loss*Z)
                    
                if is_rnn_output:
                    output = output.view(tmpsize, -1, output.shape[-1])
                    y = y.view(tmpsize, -1)
                
                running_loss +=float(loss.data.cpu().numpy())
                #if self.model_loss_function is not None:
                #    loss = loss + self.model_loss_function(self.model)
                loss.backward()
                self.optimizer.step()
                #if self.model_constraint is not None:
                #    self.model_constraint.apply(self.model)
                #self.batch_loss_history.append(float(loss.data.cpu().numpy()))
                if bi % verbosity == verbosity-1:
                    print('[%d, %d %s] loss: %g' % (epoch + 1, bi + 1, datetime.datetime.now(), running_loss / verbosity))
                    running_loss = 0.
            else:
                if is_rnn_output:
                    tmpsize = len(output)
                    output = output.view(-1, output.shape[-1])
                    y = y.view(-1)
                    
                #if evaluate_loss:
                #if self.loss_function=='mse':
                #    #loss = th.pow(y - output, 2)
                #    loss = F.mse_loss(output, y, reduce=False)
                #    loss = th.sum(loss*Z)
                #elif self.loss_function=='bce':
                #    loss = F.binary_cross_entropy_with_logits(output, y, weight=Z, size_average=False)
                if self.loss_function=='ce':
                    loss = F.cross_entropy(output, y, weight=weights)#, reduce=False)
                    loss = loss*batch_size
                    #loss = th.sum(loss*Z)
                #elif self.loss_function=='huber':
                #    loss = F.smooth_l1_loss(output, y, reduce=False)
                #    loss = th.sum(loss*Z)
                total_loss += float(loss.data.cpu().numpy())
                
                if is_rnn_output:
                    output = output.view(tmpsize, -1, output.shape[-1])
                    y = y.view(tmpsize, -1)

                total_outputs.append(var_to_np(output))
                total_Hs.append(var_to_np(H))
                
            del loss
            #time.sleep(0.1)
            del H
            #time.sleep(0.1)
            del output
            #time.sleep(0.1)
                
        if mode!='train':
            if N==0:
                N=1
            return total_loss/N, total_outputs, total_Hs
              
    def get_input_gradient(self, D):
        #if not self.fitted_:
        #    raise NotFittedError
        
        self.model.train()
        
        # set requires_grad = False and record the old value
        #req_grad_old = {}
        #for name, param in self.model.named_parameters():
        #    req_grad_old[name] = param.requires_grad
        #    #param.requires_grad = False  # RuntimeError: 'inconsistent range for TensorList output'
            
        batch_size = 1
        gen = DataLoader(D, batch_size=batch_size, shuffle=False,
                            num_workers=self.n_jobs, pin_memory=False)
            
        gradients = []
        for bi, batch in enumerate(gen):
            X = Variable(batch['X'])
            
            if self.n_gpu>0:
                X = X.cuda()
            X.requires_grad = True
            
            outputs = self.model(X)
            if type(outputs)==tuple:
                output = outputs[0]
            else:
                output = outputs
            grad = th.autograd.grad(output, X)
            gradients.extend(grad[0].data.cpu().numpy())
        gradients = np.array(gradients)
       
        # restore to old value
        #for name, param in self.model.named_parameters():
        #    param.requires_grad = req_grad_old[name]
        self.model.eval()
        
        return gradients
    
    def fit(self, Dtr, Dva=None, Dte=None, Dtrva=None):
        self.fitted_ = False
        #########TODO self.model.init()
        if self.n_gpu>0:
            self.model = self.model.cuda()
            if self.n_gpu>1:
                self.model = nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))
        else:
            self.model = self.model.cpu()
        
        self.train_history = {'loss':[]}
        if Dva is not None:
            self.train_history['valid_loss'] = []
        if Dte is not None:
            self.train_history['test_loss'] = []
        self.best_perf = np.inf
        self.best_epoch = self.max_epoch
        self.save_path = os.path.join(self.save_base_path,'current_best_model.pth')
        self.initial_path = os.path.join(self.save_base_path,'initial_model.pth')
        if self.n_jobs==-1:
            self.n_jobs = multiprocessing.cpu_count()
        
        gen_tr = DataLoader(Dtr, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.n_jobs, pin_memory=False)
        if Dva is not None:
            gen_va = DataLoader(Dva, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs, pin_memory=False)
        if Dte is not None:
            gen_te = DataLoader(Dte, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs, pin_memory=False)
                            
        # save initial model before training
        if not os.path.exists(self.save_base_path):
            os.mkdir(self.save_base_path)
        th.save(self.model, self.initial_path)
        
        st = timeit.default_timer()
        for epoch in range(self.max_epoch):
            self.run_one_epoch(epoch, gen_tr, 'train', use_gpu=self.n_gpu>0)
            if epoch==0:
                th.save(self.model, self.save_path)
            if Dva is not None:
                val_loss = self.evaluate(Dva, metrics=['loss'], use_gpu=self.n_gpu>0)#,'auc'
                current_perf = val_loss[self.remember_best_metric]
                self.train_history['valid_loss'].append(current_perf)
                print('[%d %s] val_loss: %g, current best: [epoch %d] %g' % (epoch + 1, datetime.datetime.now(), current_perf, np.argmin(self.train_history['valid_loss'])+1, np.min(self.train_history['valid_loss'])))
                
                if current_perf < self.best_perf:
                    self.best_epoch = epoch+1
                    self.best_perf = current_perf
                    if not os.path.exists(self.save_base_path):
                        os.mkdir(self.save_base_path)
                    th.save(self.model, self.save_path)
                if self.scheduler is not None:
                    self.scheduler.step(current_perf)
            if Dte is not None:
                te_loss = self.evaluate(Dte, metrics=['loss'], use_gpu=self.n_gpu>0)#,'auc'
                self.train_history['test_loss'].append(te_loss[self.remember_best_metric])
                print('[%d %s] te_loss: %g, current best [epoch %d] %g' % (epoch + 1, datetime.datetime.now(), te_loss[self.remember_best_metric], np.argmin(self.train_history['valid_loss'])+1, self.train_history['test_loss'][np.argmin(self.train_history['valid_loss'])]))
                #ypte = self.predict(Dte)
                #ypte = np.nanmean(ypte[:,30:],axis=1).flatten()
                #print(np.c_[Dte.y1d, ypte])

        et = timeit.default_timer()
        self.train_time = et-st

        if self.verbose:
            print('best epoch: %d, best val perf: %g'%(self.best_epoch, self.best_perf))
            print('training time: %gs'%self.train_time)
                
        if Dtrva is not None:
            print('\nTrain using combined tr and va...')
            gen_trva = gen_tr+gen_va
            gen_trva = DataLoader(Dtrva, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.n_jobs, pin_memory=False)
            self.model = th.load(self.intial_path)
            st = timeit.default_timer()
            
            for epoch in range(self.best_epoch):
                self.run_one_epoch(epoch, gen_trva, 'train', use_gpu=self.n_gpu>0)
                if Dte is not None:
                    te_loss = self.evaluate(Dte, metrics=['loss'], use_gpu=self.n_gpu>0)#,'auc'
                    print('[%d %s] te_loss: %g' % (epoch + 1, datetime.datetime.now(), te_loss[self.remember_best_metric]))
                
            et = timeit.default_timer()
            self.train_val_time = et-st
        else:
            self.model = th.load(self.save_path)
            
        self.fitted_ = True
        return self

    def save(self, save_path=None):
        if not self.fitted_:
            raise NotFittedError
        if save_path is None:
            save_path = self.save_path
        #th.save(self.model.state_dict(), save_path)
        th.save(self.model, save_path)

    def load(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        tmp = th.load(save_path)
        if type(tmp)==nn.DataParallel:
            tmp = tmp.module
        if type(tmp)==OrderedDict:
            self.model.load_state_dict(tmp)
        else:
            self.model = tmp
        #if self.n_gpu>0:
        #    self.model = self.model.cuda()
        #else:
        #    self.model = self.model.cpu()
        self.fitted_ = True
    
    def predict(self, D, output_id=0, return_only_loss=False, concatenate=True, apply_activation=True, use_gpu=None):
        #if not self.fitted_:
        #    raise NotFittedError
        #assert data_generator.shuffle==False

        if use_gpu is None:
            use_gpu = self.n_gpu>0
        if self.n_gpu>0:
            self.model = self.model.cuda()
            if self.n_gpu>1 and type(self.model)!=nn.DataParallel:
                self.model = nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))
        else:
            self.model = self.model.cpu()
        if self.n_jobs==-1:
            self.n_jobs = multiprocessing.cpu_count()
        
        gen = DataLoader(D, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs, pin_memory=False)
        loss, yp, Hp = self.run_one_epoch(0, gen, 'eval', use_gpu=use_gpu)#, evaluate_loss=return_only_loss
            
        if return_only_loss:
            return loss
        else:
            res = []
            if type(output_id)==int:
                output_id = [output_id]
            if 0 in output_id:
                if concatenate:
                    yp = np.concatenate(yp, axis=0)
                if apply_activation:
                    if self.loss_function=='bce':
                        yp = sigmoid(yp)
                    elif self.loss_function=='ce':
                        yp_3d = yp.ndim==3
                        if yp_3d:
                            nnn = yp.shape[0]
                            yp = yp.reshape(-1, yp.shape[-1])
                        yp = softmax(yp)
                        if yp_3d:
                            yp = yp.reshape(nnn,-1,yp.shape[-1])
                res.append(yp)
            if 1 in output_id:
                if concatenate:
                    Hp = np.concatenate(Hp, axis=0)
                res.append(Hp)
            if len(res)==1:
                res = res[0]
            return res
    
    def evaluate(self, D, metrics=['loss'], use_gpu=None):
        #if not self.fitted_:
        #    raise NotFittedError
        #assert data_generator.shuffle==False
        
        perf = {}
        perf['loss'] = self.predict(D, return_only_loss=True, use_gpu=use_gpu)
        return perf


def split_tr_va_te_assessment(records, assessments, n_tr=None, n_va=None, n_te=None):
    _, idx = np.unique(records, return_index=True)
    unique_records = records[np.sort(idx)]

    trids = []
    vaids = []
    teids = []
    for ur in unique_records:
        this_record_ids = np.where(records==ur)[0]
        this_record_assessments = assessments[this_record_ids]
        _, idx = np.unique(this_record_assessments, return_index=True)
        unique_assessments = this_record_assessments[np.sort(idx)]

        #if len(unique_assessments)<=n_tr+n_va:
        #    return [], []
        if n_tr is not None:
            n_tr_ = n_tr
        if n_va is not None:
            n_va_ = n_va
        if n_te is not None:
            n_te_ = n_te
        if n_tr is not None and n_va is not None:
            n_te_ = len(unique_assessments)-n_tr_-n_va_
        elif n_va is not None and n_te is not None:
            n_tr_ = len(unique_assessments)-n_va_-n_te_
        elif n_tr is not None and n_te is not None:
            n_va_ = len(unique_assessments)-n_tr_-n_te_
        if len(unique_assessments)==1 and n_tr_==0:
            n_tr_ = 1
            if n_va_>0:
                n_va_ -= 1
            elif n_te_>0:
                n_te_ -= 1
        if n_tr_+n_va_+n_te_>len(unique_assessments):
            print('%s removed'%ur)
            continue
        train_assessments = unique_assessments[:n_tr_]
        valid_assessments = unique_assessments[n_tr_:n_tr_+n_va_]
        test_assessments = unique_assessments[-n_te_:]#[np.sort(np.setdiff1d(np.arange(len(unique_assessments)),n_tr+n_va))]

        trids.extend(np.where(np.in1d(assessments, train_assessments))[0])
        vaids.extend(np.where(np.in1d(assessments, valid_assessments))[0])
        teids.extend(np.where(np.in1d(assessments, test_assessments))[0])

    trids = np.sort(trids)
    vaids = np.sort(vaids)
    teids = np.sort(teids)
            
    return trids, vaids, teids


"""
def split_tr_va_records(patients, records, n_va=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    unique_patients = np.unique(patients)
    trids = []
    vaids = []
    for up in unique_patients:
        ids = np.where(patients==up)[0]
        records_this_patient = records[ids]
        unique_records = np.unique(records_this_patient)
        np.random.shuffle(unique_records)
        n_records = len(unique_records)
        #n_va = max(1,int(round(n_records*(1-n_tr))))
        if n_records==1:
            tr_records = unique_records
            va_records = []
        else:
            tr_records = unique_records[:-n_va]
            va_records = unique_records[-n_va:]
        trids.extend(ids[np.in1d(records_this_patient, tr_records)])
        vaids.extend(ids[np.in1d(records_this_patient, va_records)])

    trids = np.sort(trids)
    vaids = np.sort(vaids)

    return trids, vaids
"""

