import torch
from dgl import DGLError
import dgl
import torch.nn as nn
import torch as th
from dgl import block_to_graph, DGLError, reverse
from dgl import function as fn
from dgl.heterograph import DGLBlock
from dgl.utils import expand_as_pair
from torch.nn import init


class GCN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(GCN, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.gcn=dgl.nn.pytorch.GraphConv(in_dim,in_dim)
        self.fc=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(in_dim,out_dim))

    def forward(self,g,F):
        x=self.gcn(g,F)
        x=self.fc(x)
        return x

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
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
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            self.weight = th.nn.init.kaiming_normal_(self.weight)
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

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN-kaiming on bipartite.
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

class GCN_k(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 layer_num=2,
                 norm='none',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GCN_k, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.layer_num=layer_num
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._allow_zero_in_degree = allow_zero_in_degree
        self.layers=[]
        for i in range(layer_num):
            self.layers.append(GraphConv(in_feats, in_feats, norm, weight, bias, activation,
                                   allow_zero_in_degree).to(self.device))
        self.activation=nn.ReLU().to(self.device)
        self.FC = nn.Linear(in_feats, out_feats).to(self.device)

    def forward(self, g, F, weight=None, edge_weight=None):
        x = self.layers[0](g, F, weight, edge_weight)
        x=self.activation(x)
        for i in range(1,len(self.layers)):
            x=self.layers[i](g,x,weight,edge_weight)
            x = self.activation(x)

        x = self.FC(x)
        return x
