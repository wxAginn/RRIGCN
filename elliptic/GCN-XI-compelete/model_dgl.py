import torch as th
import torch
import math
import numpy as np
import dgl
from dgl import block_to_graph, DGLError, reverse
from dgl import function as fn
from dgl.heterograph import DGLBlock
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import init
from config import data_transform


class EdgeWeightNorm(nn.Module):
    def __init__(self, norm='both', eps=0.):
        super(EdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_weight):
        with graph.local_scope():
            if isinstance(graph, DGLBlock):
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError('Currently the normalization is only defined '
                               'on scalar edge weight. Please customize the '
                               'normalization for your high-dimensional weights.')
            if self._norm == 'both' and th.any(edge_weight <= 0).item():
                raise DGLError(
                    'Non-positive edge weight detected with `norm="both"`. '
                    'This leads to square root of zero or negative values.')

            dev = graph.device
            graph.srcdata['_src_out_w'] = th.ones(
                (graph.number_of_src_nodes())).float().to(dev)
            graph.dstdata['_dst_in_w'] = th.ones(
                (graph.number_of_dst_nodes())).float().to(dev)
            graph.edata['_edge_w'] = edge_weight

            if self._norm == 'both':
                reversed_g = reverse(graph)
                reversed_g.edata['_edge_w'] = edge_weight
                reversed_g.update_all(fn.copy_edge('_edge_w', 'm'),
                                      fn.sum('m', 'out_weight'))
                degs = reversed_g.dstdata['out_weight'] + self._eps
                norm = th.pow(degs, -0.5)
                graph.srcdata['_src_out_w'] = norm

            if self._norm != 'none':
                graph.update_all(fn.copy_edge('_edge_w', 'm'),
                                 fn.sum('m', 'in_weight'))
                degs = graph.dstdata['in_weight'] + self._eps
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata['_dst_in_w'] = norm

            graph.apply_edges(
                lambda e: {'_norm_edge_weights': e.src['_src_out_w'] * \
                                                 e.dst['_dst_in_w'] * \
                                                 e.data['_edge_w']})
            return graph.edata['_norm_edge_weights']


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_of_nodes,
                 mean_A,
                 var_A,
                 var_F: list,
                 dropout=0.,
                 norm='none',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.num_of_nodes = num_of_nodes
        self.mean = mean_A
        self.var = var_A
        self.var_F=var_F
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            self.weight = self.xinitialization(self.mean, self.var,self.var_F,
                                          self.num_of_nodes, self._out_feats)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        'External weight is provided while at the same time the'
                        ' module has defined its own weight parameter. Please'
                        ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

    def xinitialization(self,mean: float, var_A: float, var_F: list, n: int, d: int):
        std_list = []
        for i in var_F:
            std_list.append(
                math.sqrt(2. * i / (d * n * var_A * i + d * mean ** 2)))
        std=torch.tensor(std_list).mean().item()
        x = nn.Parameter(torch.normal(mean=0,std=std,size=(d,d)))
        return x


def statistical_information_X(dataset):
    nA, muA, sigma2A = [], [], []
    nF, muF, sigma2F = [], [], []
    total_meanF, total_varF = [], []
    for i in range(dataset.__len__()):
        g, F, _ = dataset.__getitem__(i)
        A = g.adj(scipy_fmt='coo').todense()
        A = torch.tensor(A) + torch.eye(A.shape[0])
        D = torch.sum(A, dim=0)
        D = 1. / torch.sqrt(D)
        # D=torch.tensor([1./torch.sqrt(w) for w in D])
        D = torch.where(torch.isinf(D), torch.tensor(0.), D)
        D = torch.diag(D)
        A = torch.matmul(torch.matmul(D, torch.tensor(A).float()), D)
        nA.append(A.shape[0])
        muA.append(A.mean().item())
        sigma2A.append(A.var().item())
        #if i == 0:
        col_num = F.shape[1]
        for j in range(col_num):
            if len(nF) < col_num:
                nF.append([])
            if len(muF) < col_num:
                muF.append([])
            if len(sigma2F) < col_num:
                sigma2F.append([])
            nF[j].append(F.shape[0])
            muF[j].append(
                torch.nn.functional.normalize(F[:, j], 1, 0).mean().item())
            sigma2F[j].append(
                torch.nn.functional.normalize(F[:, j], 1, 0).var().item())
    for i in range(col_num):
        np_nF = np.array(nF[i]).astype(float)
        np_muF = np.array(muF[i])
        np_sigma2F = np.array(sigma2F[i])
        sum_F = sum(nF[i])
        total_meanF.append(np.dot(np_nF, np_muF) / sum_F)
        total_varF.append((np.dot(np_nF, np_sigma2F) + np.dot(np_nF,
                                                              np_muF ** 2) - sum(
            np_sigma2F) - sum_F * total_meanF[-1] ** 2) / (sum_F - 1))
    np_nA = np.array(nA).astype(float)
    np_muA = np.array(muA)
    np_sigma2A = np.array(sigma2A)
    sum_A = sum(nA)
    total_mean = np.dot(np_nA, np_muA) / sum_A
    total_var = (np.dot(np_nA, np_sigma2A) + np.dot(np_nA, np_muA ** 2) - sum(
        sigma2A) - sum_A * total_mean ** 2) / (sum_A - 1)
    return total_mean, total_var, total_meanF, total_varF







class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_of_nodes,
                 mean_A,
                 var_A,
                 var_F,
                 layer_num=1,
                 LRM_layer_num=2,
                 LRM_dimension=120,
                 dropout=0.,
                 alpha=0.5,
                 beta=0.5,
                 gamma=0.5,
                 delta=0.5,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GCN, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.num_of_nodes = num_of_nodes
        self.mean = mean_A
        self.var = var_A
        self.var_F = var_F
        self.layer_num = layer_num
        self.layers = []
        self.LRM = []
        self.LRM_layer_num = LRM_layer_num
        self.LRM_dimension = LRM_dimension
        self.activation = nn.ReLU()
        self.alpha=math.sqrt(alpha)
        self.beta=math.sqrt(beta)
        self.gamma=math.sqrt(gamma)
        self.delta=math.sqrt(delta)
        self.leakyrelu=nn.LeakyReLU()
        self.show_var=[]
        if dropout>0.:
            self.dropout=nn.Dropout(dropout)
        else:
            self.dropout=None
        if self.LRM_layer_num > 0:
            self.LRM.append(nn.Linear(in_feats, self.LRM_dimension).to(self.device))
        for i in range(1, self.LRM_layer_num):
            self.LRM.append(nn.Linear(self.LRM_dimension, self.LRM_dimension).to(self.device))
        for _ in range(self.layer_num):
            self.layers.append(
                GraphConv(self.LRM_dimension, self.LRM_dimension,
                          num_of_nodes,
                          mean_A,
                          var_A,self.var_F,dropout, norm, weight, bias, activation,
                          allow_zero_in_degree).to(self.device))

        self.FC = nn.Sequential(nn.Linear(self.LRM_dimension, out_feats),
                                nn.Flatten(0, -1),
                                nn.Linear(self.num_of_nodes * out_feats,
                                          out_feats))

    def sequential_graphconv(self, g, F, weight=None, edge_weight=None):
        x = self.layers[0](g, F, weight=None, edge_weight=None)
        x=self.beta*x+self.gamma*torch.cat([F[0,:].unsqueeze(0),torch.zeros(F.shape[0]-1,F.shape[1]).to(self.device)],dim=0)
        #x = self.normalize(x)
        res=x
        #x = self.activation(x)
        self.show_var.append(x.var().item())
        for i in range(1, self.layer_num):
            x = self.layers[i](g, x, weight=None, edge_weight=None)
            x = self.alpha*(self.beta * x + self.gamma * torch.cat(
                [F[0,:].unsqueeze(0), torch.zeros(F.shape[0] - 1, F.shape[1]).to(self.device)],
                dim=0))+self.delta*res
            res=x
            #x=self.normalize(x)
            #x = self.activation(x)
            self.show_var.append((x.var().item()))
        return x

    def forward(self, g, F, weight=None, edge_weight=None):
        g = dgl.add_self_loop(g)
        # F = data_transform(F, self.var_F)
        self.show_var.append(F.var().item())
        x=self.normalize(F.float())
        self.show_var.append(x.var().item())
        for i in range(self.LRM_layer_num):
            x = self.LRM[i](x)
            if self.dropout is not None:
                x=self.dropout(x)
            #x=self.leakyrelu(x)
            self.show_var.append(x.var().item())
            #x = self.activation(x)
        x = self.sequential_graphconv(g, x, weight, edge_weight)
        x=self.activation(x)
        x = self.FC(x)
        if False:#是否显示传播方差
            print(self.show_var)
        self.show_var=[]
        return x

    def normalize(self, X: torch.TensorType):
        N = 10*torch.nn.functional.normalize(X, p=1, dim=0)
        '''
        std = [1. / math.sqrt(w) for w in self.var_F]
        std = torch.diag(torch.tensor(std))
        M = torch.matmul(N, std)
        '''
        return N
