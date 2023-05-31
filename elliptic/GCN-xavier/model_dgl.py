import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim,graph_size,layer_num):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num=layer_num
        self.node_num=graph_size
        self.activation=nn.ReLU()
        self.gcn = []
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(layer_num):
            self.gcn.append(GraphConv(in_dim,in_dim).to(self.device))
        self.fc = nn.Sequential(nn.Flatten(start_dim=0),self.activation,
                                nn.Linear(in_dim*graph_size, out_dim)).to(self.device)

    def forward(self, g, F):
        g = dgl.add_self_loop(g).to(self.device)
        x=self.gcn[0](g,F.to(self.device))
        x=self.activation(x)
        for i in range(1,self.layer_num):
            x=self.gcn[i](g,x)
            x=self.activation(x)
        x = self.fc(x)
        return x
