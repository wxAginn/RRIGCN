import sys
import time
import dgl
import numpy
import torch
from config import args, sk_preformance, Logger, newperformance
from sklearn.metrics import classification_report
from random import shuffle
from dataset_dgl import xdataset_
from dgl.nn.pytorch import GATConv,APPNPConv
import torch.nn as nn
import random
torch.manual_seed(1314)
random.seed(1314)
class APPNP(nn.Module):
    def __init__(self,in_feat,out_feat,k,alpha):
        super(APPNP, self).__init__()
        self.conv1=GATConv(in_feat,out_feat,3)
        self.conv2=APPNPConv(k=k,alpha=alpha)
        self.fc = nn.Sequential(nn.Flatten(0, -1),
                                nn.Linear(out_feat*50*3, 2))

    def forward(self,A,x):
        x=self.normalize(x)
        out=self.conv1(A,x)
        out=self.conv2(A,out)
        out=self.fc(out)
        return out

    def normalize(self, X: torch.TensorType):
        N = torch.nn.functional.normalize(X, p=1, dim=0)
        '''
        std = []
        for i in range(N.shape[1]):
            if math.sqrt(N[:, i].var()) <= 0. or self.var_F[i] <= 0.:
                std.append(1.)
            else:
                # std.append( 1. / math.sqrt(self.var_F[i]))
                std.append(1. / math.sqrt(N[:, i].var()))
        std = torch.diag(torch.tensor(std))
        M = torch.matmul(N, std)
        '''
        return N

train_len = int(0.8 * xdataset_.__len__())
test_len = xdataset_.__len__() - train_len
train_dataset, test_dataset = torch.utils.data.random_split(xdataset_, [train_len, test_len])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = APPNP(in_feat=166,out_feat=2,k=3,alpha=0.8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer_comp = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = torch.nn.CrossEntropyLoss()

aloss = float('Inf')
mark = 0
seq_tra, seq_tes = [s for s in range(train_dataset.__len__())], [s for s in
                                                                 range(
                                                                     test_dataset.__len__())]
shuffle(seq_tra)
shuffle(seq_tes)
print(train_dataset.__len__())
for epoch in range(args.epochs):
    avg_loss, total_loss = 0., []
    print('----train----,epoch:', epoch)
    i = 0
    while i < train_dataset.__len__():
        count = i % args.batch_size
        output_cache, L_cache = [], []
        optimizer.zero_grad()
        while count < args.batch_size:
            count += 1
            if i >= len(seq_tra):
                break
            g, F, L = train_dataset.__getitem__(seq_tra[i])
            output_cache.append(model(g.to(device), F.to(device)))
            L_cache.append(L)
            i += 1
        output = torch.stack(output_cache)
        loss = criterion(output, torch.tensor(L_cache).to(device))
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    avg_loss = numpy.mean(numpy.array(total_loss))
    print(avg_loss)
    print('----test----epoch:', epoch)
    i, L_list, L_pre = 0, [], []
    while i < test_dataset.__len__():
        g, F, L = test_dataset.__getitem__(seq_tes[i])
        _, l_pre = torch.max(model(g.to(device), F.to(device)), 0)
        L_pre.append(l_pre.item())
        L_list.append(L)
        i += 1
    print(classification_report(y_true=L_list, y_pred=L_pre, digits=6))
