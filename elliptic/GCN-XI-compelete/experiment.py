import torch
import numpy
import dgl
import argparse
from GAT import GAT
from torch.utils.data import Dataset, random_split
from random import shuffle
from sklearn.metrics import classification_report


class Experiment:
    def __init__(self, model, optimizer, criterion, trainset,testset, args):
        '''

        :param model:
        :param optimizer:
        :param criterion:
        :param dataset:
        :param args: train_set_len 训练集在总的数据集占比
                     epochs
                     batch_size
                     shuffle
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.train_dataset, self.test_dataset = trainset,testset


    def train(self, epoch):
        self.model.train()
        seq_tra = [w for w in range(self.train_dataset.__len__())]
        if self.args.shuffle:
            shuffle(seq_tra)
        print('len of train set is :', self.train_dataset.__len__())
        avg_loss, total_loss = 0., []
        print('----train----,epoch:', epoch)
        i = 0
        while i < self.train_dataset.__len__():
            count = i % self.args.batch_size
            output_cache, L_cache = [], []
            self.optimizer.zero_grad()
            while count < self.args.batch_size:
                count += 1
                if i >= len(seq_tra):
                    break
                data = self.train_dataset.__getitem__(seq_tra[i])
                out, L = self._forward(data, self.model, self.args)
                # output_cache是0维的数字
                output_cache.append(out)
                L_cache.append(L)
                i += 1
            output = torch.stack(output_cache)
            loss = self.criterion(output.to(self.device), torch.tensor(L_cache).to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())
        avg_loss = numpy.mean(numpy.array(total_loss))
        print('avg_loss of epoch ', epoch, 'is ', avg_loss)

    def test(self):
        self.model.eval()
        seq_tes = [w for w in range(self.test_dataset.__len__())]
        if self.args.shuffle:
            shuffle(seq_tes)
        i, L_list, L_pre = 0, [], []
        while i < self.test_dataset.__len__():
            data = self.test_dataset.__getitem__(seq_tes[i])
            output_cache, L = self._forward(data, self.model, self.args)
            _, l_pre = torch.max(output_cache, 0)
            L_pre.append(l_pre.item())
            L_list.append(L)
            i += 1
        print(classification_report(y_true=L_list, y_pred=L_pre, digits=6))

    def _forward(self, data, model, args):
        '''
        此函数不同数据集和模型使用时需重写,返回不能变
        :param data:
        :param model:
        :return: output_cache [1,2] 各个分类的打分，L标签
        '''
        g, F, L = data
        output_cache=model(g.to(self.device), F.to(self.device))
        return output_cache, L
