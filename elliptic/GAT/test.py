import sys
import numpy
import torch
from config import args, sk_preformance, Logger, newperformance
from sklearn.metrics import classification_report
from random import shuffle
from GAT import GAT

#recordname = '../log/GAT_2021_9_20.txt'
#sys.stdout = Logger(recordname, sys.stdout)
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
