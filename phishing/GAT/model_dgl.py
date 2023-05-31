import torch.nn.functional
from dgl.nn.pytorch import GATConv
import torch.nn as nn


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gat = GATConv(in_dim, out_dim, num_heads=3)
        self.fc = nn.Sequential(nn.Flatten(0, -1),
                                nn.Linear(self.out_dim * self.num_heads, 2))

    def forward(self, g, F):
        x = torch.nn.functional.normalize(F, 1, 0).float()
        x = self.gat(g, x)
        ans = x[0, :]
        ans = self.fc(ans)
        return ans
