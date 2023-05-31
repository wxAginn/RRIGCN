import torch
from torch.utils.data import DataLoader
from dataload.Xdataset import Xdataset

xdataset_ = Xdataset()
train_len=int(xdataset_.__len__()*0.8)
test_len=xdataset_.__len__()-train_len
train_dataset,test_dataset=torch.utils.data.random_split(xdataset_,[train_len,test_len])