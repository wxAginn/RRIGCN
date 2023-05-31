import sys
import time
import dgl
import numpy
import torch
from config import args, sk_preformance, Logger, newperformance
from sklearn.metrics import classification_report
from random import shuffle
from GAT import GAT
from dataset_dgl import xdataset_

#recordname = '../log/GAT' + time.strftime('%Y_%m_%d_%H%M%S', time.localtime()) + '.txt'
# sys.stdout = Logger(recordname, sys.stdout)
train_len = int(0.8 * xdataset_.__len__())
test_len = xdataset_.__len__() - train_len
train_dataset, test_dataset = torch.utils.data.random_split(xdataset_, [train_len, test_len])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(166, 2, num_heads=3, graph_size=50).to(device)
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
