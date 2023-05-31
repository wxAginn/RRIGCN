import argparse
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import sys
import torch


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
