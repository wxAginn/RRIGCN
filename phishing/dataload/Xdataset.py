import os
import pickle
from random import shuffle

import dgl

from dataload import XGraph, transform
from torch.utils.data import Dataset
import torch


class Xdataset(Dataset):
    def __init__(self, root='../dataset/'):
        self.root = root
        self.cache=os.path.join(self.root,'cache/phishing_cache.pickle')
        self.onehopdirs, self.onehopfiles = [], []
        for i in os.listdir(self.root):
            if i.find('one') >= 0:
                # string.find(substring,index=0) 从index开始往下查找子字符串，若找到则返回第一个子字符串所在下标，否则返回-1
                self.onehopdirs.append(i)
        for i in self.onehopdirs:
            self.onehopfiles.append(os.listdir(os.path.join(root,i)))
        if os.path.exists(self.cache):
            with open(self.cache,'rb+') as f:
                self.glist,self.L_list=pickle.load(f)
        else:
            self.glist,self.L_list=[],[]
            for dirs in self.onehopdirs:
                files=os.listdir(os.path.join(root,dirs))
                for i in range(len(files)):
                    if dirs.find('nonphishing')>= 0:
                        self.L_list.append(0)# 源点并非钓鱼节点
                    else:
                        self.L_list.append(1)# 源点是钓鱼节点
                    g = XGraph.create_detection_domain(
                        os.path.join(self.root, dirs),
                        files[i])
                    self.glist.append(g)
            with open(self.cache,'wb') as f:
                pickle.dump([self.glist,self.L_list],f)
        self.shuf = [w for w in range(len(self.glist))]
        shuffle(self.shuf)


    def __getitem__(self, item):
        """
        :param item:
        :return:A,F,L
        """
        item=self.shuf[item]
        g=self.glist[item]
        g=dgl.add_self_loop(g)
        F=g.ndata['feature']
        #F=torch.nn.functional.normalize(F, p=1, dim=0).to(torch.float32)
        L=self.L_list[item]
        return g, F, L

    def __len__(self):
        l = 0
        for i in self.onehopfiles:
            l += len(i)
        return l
