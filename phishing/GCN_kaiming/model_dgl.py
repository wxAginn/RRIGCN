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
from dataload.Xdataset import Xdataset
#from config import data_transform
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
            self.weight = nn.Parameter(torch.nn.init.kaiming_normal_(th.Tensor(self._in_feats,self._out_feats)))
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

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_of_nodes,
                 layer_num=2,
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.num_of_nodes = num_of_nodes
        self.layer_num = layer_num
        self.layers = []
        self.activation=nn.LeakyReLU()
        for _ in range(self.layer_num):
            self.layers.append(GraphConv(in_feats, in_feats, num_of_nodes,norm, weight, bias, activation,
                                    allow_zero_in_degree).to(self.device))
        self.FC = nn.Sequential(nn.Linear(in_feats, out_feats),
                                nn.Flatten(0, -1),
                                nn.Linear(self.num_of_nodes * out_feats,
                                          out_feats))
    def sequential_graphconv(self,g,F, weight=None, edge_weight=None):
        x=self.layers[0](g,F, weight=None, edge_weight=None)
        x=self.activation(x)
        for i in range(1,self.layer_num):
            x=self.layers[i](g,x, weight=None, edge_weight=None)
            x=self.activation(x)
        return x

    def forward(self, g, F, weight=None, edge_weight=None):
        g = dgl.add_self_loop(g)
        x=torch.nn.functional.normalize(F,1,0)
        x = self.sequential_graphconv(g, x, weight, edge_weight)
        x = self.FC(x)
        return x


