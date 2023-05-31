import time

import torch
import numpy
from torch.utils.data import random_split
from random import shuffle
from sklearn.metrics import classification_report


class Experiment:
    def __init__(self, model, optimizer, criterion,trainset,testset, args):
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
        self.criterion = criterion.to(self.device)
        self.args = args
        self.train_dataset=trainset
        self.test_dataset=testset
        self.checkpoint_path='../model_saved/'


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
            output = torch.stack(output_cache).to(self.device)
            loss = self.criterion(output, torch.tensor(L_cache).to(self.device))
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
        metrics=classification_report(y_true=L_list, y_pred=L_pre, digits=6,output_dict=True)
        print(classification_report(y_true=L_list, y_pred=L_pre, digits=6))
        if metrics['accuracy']>0.85:
            torch.save({'state_dict': self.model.state_dict(),'optimizer': self.optimizer.state_dict()},
                           self.checkpoint_path +time.strftime('%Y_%m_%d_%H%M%S',time.localtime())+'.pth.tar')


    def _forward(self, data, model, args):
        '''
        此函数不同数据集和模型使用时需重写,返回不能变
        :param data:
        :param model:
        :return: output_cache [1,2] 各个分类的打分，L标签
        '''
        g, F, L = data
        output_cache=model(g.to(self.device), F.float().to(self.device))
        return output_cache, L

def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer
