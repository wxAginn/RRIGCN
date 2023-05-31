import sys
import time

import numpy
import torch
from sklearn.metrics import classification_report
from random import shuffle
from model import GCN_k
from config import Logger
import argparse
import dgl
from dataset_dgl import xdataset_
recordname = '../log/GCN_kaiming'+time.strftime('%Y_%m_%d_%H%M%S',time.localtime())+'.txt'
sys.stdout = Logger(recordname, sys.stdout)
train_len=int(xdataset_.__len__()*0.8)
test_len=xdataset_.__len__()-train_len
train_dataset,test_dataset=torch.utils.data.random_split(xdataset_,[train_len,test_len])
args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=150)
args.add_argument('--hidden', type=int, default=16)
args.add_argument('--dropedge', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--graph_size', type=int, default=50)
args.add_argument('--batch_size', type=int, default=24)
args.add_argument('--raw-dir', type=str,
                           default='../dataset/elliptic_bitcoin_dataset/',
                           help="Dir after unzip downloaded dataset, which contains 3 csv files.")
args.add_argument('--processed-dir', type=str,
                           default='../dataset/elliptic_bitcoin_dataset/processed/',
                           help="Dir to store processed raw data.")
args.add_argument('--layer_num',type=int,default=2)
#args.add_argument('--gpu', type=int, default=-1,
#                           help="GPU device ID. Use -1 for CPU training.")
#args.add_argument('--device', type=str, default='cpu')
#args.add_argument('--xdatacache_dir',type=str,default='../dataset/cache/xdatacache.pickle')
args = args.parse_args(args=[])
print(args)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_k(166, 2,args.layer_num).to(device)
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
            g = dgl.add_self_loop(g)
            output_cache.append(model(g.to(device), F.to(device))[0, :])
            L_cache.append(L)
            i += 1
        output = torch.stack(output_cache)
        loss = criterion(output.to(device), torch.tensor(L_cache).to(device))
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    avg_loss=numpy.mean(numpy.array(total_loss))
    print(avg_loss)
    print('----test----epoch:',epoch)
    i,L_list,L_pre=0,[],[]
    while i < test_dataset.__len__():
        g, F, L = test_dataset.__getitem__(seq_tes[i])
        g = dgl.add_self_loop(g)
        _,l_pre=torch.max(model(g.to(device), F.to(device))[0, :],0)
        L_pre.append(l_pre.item())
        L_list.append(L)
        i += 1
    print(classification_report(y_true=L_list,y_pred=L_pre,digits=6))
