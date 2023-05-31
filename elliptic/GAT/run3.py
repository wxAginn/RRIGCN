import sys
import numpy
import torch
from config import args, sk_preformance, Logger, newperformance
from sklearn.metrics import classification_report
from random import shuffle
from GAT import GAT

recordname = '../log/GAT_2021_11_14_LeaktReLU3.txt'
sys.stdout = Logger(recordname, sys.stdout)
print(args)
import dgl
from dataset import train_dataset, test_dataset

model = GAT(166, 2, num_heads=3)
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
            output_cache.append(model(g, F)[0, :])
            L_cache.append(L)
            i += 1
        output = torch.stack(output_cache)
        loss = criterion(output, torch.tensor(L_cache))
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
        _,l_pre=torch.max(model(g, F)[0, :],0)
        L_pre.append(l_pre)
        L_list.append(L)
        i += 1
    print(classification_report(y_true=L_list,y_pred=L_pre,digits=6))
