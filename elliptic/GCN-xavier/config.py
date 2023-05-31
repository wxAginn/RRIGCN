import argparse
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import sys
import torch

args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=150)
args.add_argument('--hidden', type=int, default=16)
args.add_argument('--dropedge', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--graph_size', type=int, default=950)
args.add_argument('--feature_size', type=int, default=6)
args.add_argument('--batch_size', type=int, default=24)
args.add_argument('--raw-dir', type=str,
                           default='../dataset/elliptic_bitcoin_dataset/',
                           help="Dir after unzip downloaded dataset, which contains 3 csv files.")
args.add_argument('--processed-dir', type=str,
                           default='../dataset/elliptic_bitcoin_dataset/processed/',
                           help="Dir to store processed raw data.")
args.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training.")
args.add_argument('--device', type=str, default='cpu')
args.add_argument('--xdatacache_dir',type=str,default='../dataset/cache/xdatacache.pickle')
args = args.parse_args(args=[])
print(args)

def fscore(precision: float, recall: float, beta=1.):
    if (beta ** 2 * precision + recall) > 0:
        x = (1 + beta ** 2) * precision * recall / (
                beta ** 2 * precision + recall)
    else:
        x = 0
    return x

def get_performance(pre: torch.TensorType, L: torch.TensorType, beta=1.):
    """
    只适合二分类，输入为1维tensor
    :param pre: a tensor dim=1 composed of 0 and 1
    :param L: a tensor dim=1 composed of o and 1
    :return: accuracy,precision,recall,fscore
    """
    TP = pre[pre == L].sum().item()
    FP = pre[pre != L].sum().item()
    TN = len(pre[pre == L]) - pre[pre == L].sum().item()
    FN = len(pre[pre != L]) - pre[pre != L].sum().item()
    accuracy = (TP + TN) / (TP + TN + FP + FN + 0.)
    if TP + FP > 0.:
        precision = TP / (TP + FP + 0.)
    else:
        precision = 0.
    if TP + FN > 0.:
        recall = TP / (TP + FN + 0.)
    else:
        recall = 0.
    Fscore = fscore(precision, recall, beta)
    return accuracy, precision, recall, Fscore

def sk_preformance(y_pred: torch.TensorType, y_true: torch.TensorType, beta=1.):
    acc = accuracy_score(y_true=y_true,y_pred=y_pred)
    precision = precision_score(y_true=y_true,y_pred=y_pred)
    recall = recall_score(y_true=y_true,y_pred=y_pred)
    Fscore = f1_score(y_true=y_true,y_pred=y_pred)
    return acc, precision, recall, Fscore


def newperformance(rep: float, p: float, count: int):
    if count == 0:
        re = p
    else:
        count += 1
        re = (count-1) * rep / count + p / count
    return re
'''
def transform(A:torch.TensorType,mean:float):
    return A-mean
def transform(A:torch.TensorType,mean:float,var:float,num:int):
    return A/math.sqrt(var)
    '''
def data_transform(F: torch.TensorType,var:list):
    for i in range(F.shape[2]):
        F[:,:,i]=F[:,:,i]/math.sqrt(var[i])
    return F

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
