import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn_dropout = 0.5
        self.rnn_hidden_num = 32#hidden_num
        #self.L = 50
        self.n_output = 5#self.feedforward.n_output
        self.prior = None
        self.rnn_layer_num = 2#rnn_layer_num
        self.bidirectional = True
            
        #dummy = self._forward(Variable(th.ones(2,self.L,self.feedforward.n_channel, self.feedforward.n_timestep)))
        #D = dummy.shape[2]
        D = 1280
        self.rnn = nn.GRU(D, self.rnn_hidden_num, self.rnn_layer_num, bidirectional=self.bidirectional, dropout=self.rnn_dropout, batch_first=True)
        if self.bidirectional:
            rnn_hidden_num = self.rnn_hidden_num*2
        else:
            rnn_hidden_num = self.rnn_hidden_num
        self.fc_output = nn.Linear(rnn_hidden_num, self.n_output)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        h, _ = self.rnn(x)
        h = h.contiguous()
        
        N = h.shape[0]
        h = h.view(-1, h.shape[-1])
        y = self.fc_output(h)
        y = y.view(N, -1, y.shape[-1])
        h = h.view(N, -1, h.shape[-1])
        return y, h
    
    def init(self, method='orth'):
            
        for wn, w in self.rnn.named_parameters():
            if 'bias' in wn:
                n = w.size(0)
                """LSTM
                w.data[:n//4].fill_(0.)
                w.data[n//4:n//2].fill_(1.)
                w.data[n//2:].fill_(0.)
                """
                #GRU
                w.data[:n//3].fill_(-1.)
                w.data[n//3:].fill_(0.)
            else:
                if len(w.size())>=2:
                    if method=='orth':
                        nn.init.orthogonal(w)
                    else:
                        nn.init.xavier_normal(w)
                else:
                    nn.init.normal(w, std=0.01)
                    
        for wn, w in self.fc_output.named_parameters():
            if 'bias' in wn:
                nn.init.constant(w, 0.)
            else:
                if len(w.size())>=2:
                    if method=='orth':
                        nn.init.orthogonal(w)
                    else:
                        nn.init.xavier_normal(w)
                else:
                    nn.init.normal(w, std=0.01)
                    
        
class ResBlock2d(nn.Module):
    def __init__(self, Lin, Lout, filter_len, dropout, subsampling, momentum, maxpool_padding=0):
        assert filter_len%2==1
        super(ResBlock2d, self).__init__()
        self.Lin = Lin
        self.Lout = Lout
        self.filter_len = filter_len
        self.dropout = dropout
        self.subsampling = subsampling
        self.momentum = momentum
        self.maxpool_padding = maxpool_padding

        self.bn1 = nn.BatchNorm2d(self.Lin, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv2d(self.Lin, self.Lin, (self.filter_len,1), stride=(self.subsampling,1), padding=(self.filter_len//2,0), bias=False)
        self.bn2 = nn.BatchNorm2d(self.Lin, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv2d(self.Lin, self.Lout, (self.filter_len,1), stride=1, padding=(self.filter_len//2,0), bias=False)
        #self.bn3 = nn.BatchNorm2d(self.Lout, momentum=self.momentum, affine=True)
        if self.Lin==self.Lout and self.subsampling>1:
            self.maxpool = nn.MaxPool2d((self.subsampling,1), padding=(self.maxpool_padding,0))

    def forward(self, x):
        if self.Lin==self.Lout:
            res = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        if self.Lin==self.Lout:
            if self.subsampling>1:
                x = x+self.maxpool(res)
            else:
                x = x+res
        #x = self.bn3(x)
        return x


class ECGSleepNet2d(nn.Module):

    def __init__(self, to_combine=False):#, filter_len):
        super(ECGSleepNet2d, self).__init__()
        self.filter_len = 17#33
        self.filter_num = 64#16
        self.padding = self.filter_len//2
        self.dropout = 0.5
        self.momentum = 0.1
        self.subsampling = 4
        self.n_channel = 1
        self.n_timestep = 54000#//2
        self.n_output = 5
        self.to_combine = to_combine
        
        self.conv1 = nn.Conv2d(1, self.filter_num, (self.filter_len,1), stride=1, padding=(self.padding,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        
        self.conv2_1 = nn.Conv2d(self.filter_num, self.filter_num, (self.filter_len,1), stride=(self.subsampling,1), padding=(self.padding,0), bias=False)
        self.bn2 = nn.BatchNorm2d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2_2 = nn.Conv2d(self.filter_num, self.filter_num, (self.filter_len,1), stride=1, padding=(self.padding,0), bias=False)
        self.maxpool2 = nn.MaxPool2d((self.subsampling,1))
        #self.bn_input = nn.BatchNorm2d(self.filter_num, momentum=self.momentum, affine=True)

        self.resblock1 = ResBlock2d(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock2 = ResBlock2d(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock3 = ResBlock2d(self.filter_num, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
                
        self.resblock4 = ResBlock2d(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock5 = ResBlock2d(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock6 = ResBlock2d(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock7 = ResBlock2d(self.filter_num*2, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)
                
        self.resblock8 = ResBlock2d(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock9 = ResBlock2d(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock10 = ResBlock2d(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
        self.resblock11 = ResBlock2d(self.filter_num*3, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        
        self.resblock12 = ResBlock2d(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
        self.resblock13 = ResBlock2d(self.filter_num*4, self.filter_num*5, self.filter_len,
                self.dropout, 1, self.momentum)

        self.bn_output = nn.BatchNorm2d(self.filter_num*5, momentum=self.momentum, affine=True)
        self.relu_output = nn.ReLU()
        
        #if not self.to_combine:
        dummy = self._forward(Variable(th.ones(1,self.n_channel, self.n_timestep,1)))
        self.Hdim = dummy.size(1)#self.filter_num*5*4
        self.fc_output = nn.Linear(self.Hdim, self.n_output)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        res = x
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2_2(x)
        x = x+self.maxpool2(res)

        #x = self.bn_input(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        if hasattr(self, 'to_combine') and self.to_combine:
            return x
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        
        x = self.bn_output(x)
        x = self.relu_output(x)

        #x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        N = x.shape[0]
        ff_input = len(x.size())==3
        if ff_input:
        
            # to be compatible when combined with RNN
            # the last dimention is time step in RNN, which is always 1 here
            x = th.unsqueeze(x, -1)
            H = self._forward(x)
            #H = H.squeeze()
            
            H = H.view(N,-1)
            if not hasattr(self, 'to_combine') or not self.to_combine:
                output = self.fc_output(H)
        else:
            x = x.permute(0,2,3,1)
            H = self._forward(x)
            H = H.permute(0,3,1,2).contiguous()
            H = H.view(N, H.shape[1], -1)
            output = None
        
        return output, H
        
    def load_param(self, model_path):
        model = th.load(model_path)
        if type(model)==nn.DataParallel and hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'state_dict'):
            model = model.state_dict()
        self.load_state_dict(model)
        
    def fix_param(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfix_param(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def init(self, method='orth'):
        pass
        
        
class ResBlock(nn.Module):
    def __init__(self, Lin, Lout, filter_len, dropout, subsampling, momentum, maxpool_padding=0):
        assert filter_len%2==1
        super(ResBlock, self).__init__()
        self.Lin = Lin
        self.Lout = Lout
        self.filter_len = filter_len
        self.dropout = dropout
        self.subsampling = subsampling
        self.momentum = momentum
        self.maxpool_padding = maxpool_padding

        self.bn1 = nn.BatchNorm1d(self.Lin, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv1d(self.Lin, self.Lin, self.filter_len, stride=self.subsampling, padding=self.filter_len//2, bias=False)
        self.bn2 = nn.BatchNorm1d(self.Lin, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv1d(self.Lin, self.Lout, self.filter_len, stride=1, padding=self.filter_len//2, bias=False)
        #self.bn3 = nn.BatchNorm1d(self.Lout, momentum=self.momentum, affine=True)
        if self.Lin==self.Lout and self.subsampling>1:
            self.maxpool = nn.MaxPool1d(self.subsampling, padding=self.maxpool_padding)

    def forward(self, x):
        if self.Lin==self.Lout:
            res = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        if self.Lin==self.Lout:
            if self.subsampling>1:
                x = x+self.maxpool(res)
            else:
                x = x+res
        #x = self.bn3(x)
        return x
        

class ECGSleepNet(nn.Module):

    def __init__(self, to_combine=False):#, filter_len):
        super(ECGSleepNet, self).__init__()
        self.filter_len = 17#33
        self.filter_num = 64#16
        self.padding = self.filter_len//2
        self.dropout = 0.5
        self.momentum = 0.1
        self.subsampling = 4
        self.n_channel = 1
        self.n_timestep = 54000#//2
        self.n_output = 5
        self.to_combine = to_combine
        
        # input convolutional block
        # 1 x 54000
        self.conv1 = nn.Conv1d(1, self.filter_num, self.filter_len, stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        
        # 64 x 54000
        self.conv2_1 = nn.Conv1d(self.filter_num, self.filter_num, self.filter_len, stride=self.subsampling, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2_2 = nn.Conv1d(self.filter_num, self.filter_num, self.filter_len, stride=1, padding=self.padding, bias=False)
        self.maxpool2 = nn.MaxPool1d(self.subsampling)
        #self.bn_input = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)

        # 64 x 13500
        self.resblock1 = ResBlock(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock2 = ResBlock(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock3 = ResBlock(self.filter_num, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock4 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
                
        # 128 x 844
        self.resblock5 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock6 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock7 = ResBlock(self.filter_num*2, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)                
        self.resblock8 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
                
        # 192 x 53
        self.resblock9 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock10 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
        self.resblock11 = ResBlock(self.filter_num*3, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock12 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
                
        # 256 x 4
        self.resblock13 = ResBlock(self.filter_num*4, self.filter_num*5, self.filter_len,
                self.dropout, 1, self.momentum)

        # 320 x 4
        self.bn_output = nn.BatchNorm1d(self.filter_num*5, momentum=self.momentum, affine=True)
        self.relu_output = nn.ReLU()
        
        #if not self.to_combine:
        dummy = self._forward(Variable(th.ones(1,self.n_channel, self.n_timestep)))
        self.fc_output = nn.Linear(dummy.size(1), self.n_output)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        res = x
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2_2(x)
        x = x+self.maxpool2(res)

        #x = self.bn_input(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        if hasattr(self, 'to_combine') and self.to_combine:
            return x
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        
        x = self.bn_output(x)
        x = self.relu_output(x)

        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        h = self._forward(x)
        if not hasattr(self, 'to_combine') or not self.to_combine:
            x = self.fc_output(h)
        
        return x, h
        
    def load_param(self, model_path):
        model = th.load(model_path)
        if type(model)==nn.DataParallel and hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'state_dict'):
            model = model.state_dict()
        self.load_state_dict(model)
        
    def fix_param(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfix_param(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def init(self, method='orth'):
        pass


class SleepNet_RNN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden, n_layer, dropout=0.5, bidirectional=False):
        super(SleepNet_RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, n_hidden, n_layer, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            L = n_hidden*2
        else:
            L = n_hidden
        self.fc_output = nn.Linear(L, output_size)
        
    def forward(self, x):
        h, _ = self.rnn(x)
        y = self.fc_output(h)
        return y, h      


class CHESTSleepNet(nn.Module):#TODO rename to RespSleepNet

    def __init__(self, to_combine=False):#, filter_len):
        super(CHESTSleepNet, self).__init__()
        self.filter_len = 17#33
        self.filter_num = 64#16
        self.padding = self.filter_len//2
        self.dropout = 0.5
        self.momentum = 0.1
        self.subsampling = 4
        self.n_channel = 1
        self.n_timestep = 2700
        self.n_output = 5
        self.to_combine = to_combine
        
        # 1x2700
        self.conv1 = nn.Conv1d(1, self.filter_num, self.filter_len, stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        
        # 64x2700
        self.conv2_1 = nn.Conv1d(self.filter_num, self.filter_num, self.filter_len, stride=self.subsampling, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2_2 = nn.Conv1d(self.filter_num, self.filter_num, self.filter_len, stride=1, padding=self.padding, bias=False)
        self.maxpool2 = nn.MaxPool1d(self.subsampling)
        #self.bn_input = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)

        # 64x675
        self.resblock1 = ResBlock(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock2 = ResBlock(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock3 = ResBlock(self.filter_num, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock4 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
                
        # 128x43
        self.resblock5 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock6 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock7 = ResBlock(self.filter_num*2, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock8 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
                
        # 192x3
        self.resblock9 = ResBlock(self.filter_num*3, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        """        
        self.resblock10 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock11 = ResBlock(self.filter_num*3, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        
        self.resblock12 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock13 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock14 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock15 = ResBlock(self.filter_num*4, self.filter_num*5, self.filter_len,
                self.dropout, 1, self.momentum)
                
        self.resblock16 = ResBlock(self.filter_num*5, self.filter_num*5, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock17 = ResBlock(self.filter_num*5, self.filter_num*5, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock18 = ResBlock(self.filter_num*5, self.filter_num*5, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock19 = ResBlock(self.filter_num*5, self.filter_num*6, self.filter_len,
                self.dropout, 1, self.momentum)
                
        self.resblock20 = ResBlock(self.filter_num*6, self.filter_num*6, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock21 = ResBlock(self.filter_num*6, self.filter_num*6, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock22 = ResBlock(self.filter_num*6, self.filter_num*6, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        self.resblock23 = ResBlock(self.filter_num*6, self.filter_num*7, self.filter_len,
                self.dropout, 1, self.momentum)
        """

        # 256x3
        self.bn_output = nn.BatchNorm1d(self.filter_num*4, momentum=self.momentum, affine=True)
        self.relu_output = nn.ReLU()
        
        #if not self.to_combine:
        dummy = self._forward(Variable(th.ones(1,self.n_channel, self.n_timestep)))
        self.fc_output = nn.Linear(dummy.size(1), self.n_output)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        res = x
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2_2(x)
        x = x+self.maxpool2(res)

        #x = self.bn_input(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        if hasattr(self, 'to_combine') and self.to_combine:
            return x
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        
        x = self.bn_output(x)
        x = self.relu_output(x)

        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        h = self._forward(x)
        if not hasattr(self, 'to_combine') or not self.to_combine:
            x = self.fc_output(h)
        
        return x, h
        
    def load_param(self, model_path):
        model = th.load(model_path)
        if type(model)==nn.DataParallel and hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'state_dict'):
            model = model.state_dict()
        self.load_state_dict(model)
        
    def fix_param(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfix_param(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def init(self):
        pass
        
        
class CombinedSleepNet2(nn.Module):
    def __init__(self, pretrained_model_paths):
        super(CombinedSleepNet2, self).__init__()
        self.pretrained_model_paths = pretrained_model_paths
        self.n_channel1 = 1
        self.n_timestep1 = 54000
        self.n_channel2 = 1
        self.n_timestep2 = 2700
        self.n_output = 5
        
        self.filter_len = 17
        self.filter_num = 64
        self.padding = self.filter_len//2
        self.dropout = 0.5
        self.momentum = 0.1
        self.subsampling = 4
        
        self.ecgnet = ECGSleepNet()
        self.ecgnet.load_param(self.pretrained_model_paths[0])
        self.ecgnet.fix_param()
        self.ecgnet.to_combine = True
        
        self.respnet = CHESTSleepNet()
        self.respnet.load_param(self.pretrained_model_paths[1])
        self.respnet.fix_param()
        self.respnet.to_combine = True
        
        dummy = self._forward1([Variable(th.ones(1,self.n_channel1,self.n_timestep1)), Variable(th.ones(1,self.n_channel2,self.n_timestep2))])

        # 320x53
        self.resblock_comb = ResBlock(dummy.size(1), self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
                
        # 256x53
        self.resblock1 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
        # 256x14
        self.resblock2 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        # 256x14
        self.resblock3 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
        # 256x4
        self.resblock4 = ResBlock(self.filter_num*4, self.filter_num*5, self.filter_len,
                self.dropout, 1, self.momentum)
                
        # 320x4
        self.bn_output = nn.BatchNorm1d(self.filter_num*5, momentum=self.momentum, affine=True)
        self.relu_output = nn.ReLU()
        
        dummy = self._forward2(dummy)
        self.fc_output = nn.Linear(dummy.size(1), self.n_output)

    def _forward1(self, x):
        _, x1 = self.ecgnet(x[0])
        #TODO bn?
        
        _, x2 = self.respnet(x[1]/200.)
        #TODO bn?
        #x2 = F.pad(x2, (5,5), mode='constant', value=0)
        x2 = F.pad(x2.unsqueeze(3),(0,0,5,5), mode='constant', value=0).squeeze(3)
        
        x = th.cat((x1, x2), dim=1)
        
        return x
        
    def _forward2(self, x):
        x = self.resblock_comb(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
        x = self.bn_output(x)
        x = self.relu_output(x)

        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self._forward1(x)
        h = self._forward2(x)
        x = self.fc_output(h)
        
        return x, h
    
    def init(self):
        pass

