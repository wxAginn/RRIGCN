from dgl.nn.pytorch  import GATConv
import torch.nn as nn
class GAT(nn.Module):
    def __init__(self,in_dim,out_dim,num_heads,graph_size):
        super(GAT, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.num_heads=num_heads
        self.gat=GATConv(in_dim,out_dim,num_heads=3)
        self.fc=nn.Sequential(nn.Flatten(start_dim=0,end_dim=-1),nn.Linear(graph_size*self.out_dim*self.num_heads,2))
        self.activation=nn.ReLU()

    def forward(self,g,F):
        x=self.gat(g,F)
        x=self.activation(x)
        x=self.fc(x)
        return x