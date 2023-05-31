import transform
import numpy as np
import os
import torch
import networkx as nx
import argparse
import dgl

args=argparse.ArgumentParser()
args.add_argument('--graph_size',type=int,default=3000)

args=args.parse_args(args=[])

def find_twohop_neighbours(root, onehop_file: str):
    """
    e.g. root='data\phishing_one_hop_nodes'
         onehop_file='0x002f0c8119c16d310342d869ca8bf6ace34d9c39.csv'
    return
    """
    r = root[0:20] + 'two' + root[23::]
    twohop_dir = onehop_file[0:-4]
    return os.path.join(r, twohop_dir)


def refineTable(table):
    """
    此函数用于原始数据，输出特征阵，其中特征阵去掉列名，0x格式数据转换为十进制数据，但依然是np.str
    :param table:
    :return:
    """
    table = table[:, 1:7]
    table = table[1:, :]
    return table


def readAGraph(filename: str, G: nx.Graph):
    """
    将filename内的图信息读入G中
    """
    f_info = np.loadtxt("{}".format(filename), delimiter=",", dtype=str)
    # ' '   TxHash	BlockHeight	TimeStamp	From	To	Value	ContractAddress(N)	Input(N)	isError(N)
    # print(f_info)
    feature = refineTable(f_info)
    nodesTo = feature[:, 4].copy()
    nodesFrom = feature[:, 3].copy()
    feature[:, [0, 3, 4]] = dataload.transform.npstrArraytoFloatArray2D(
        feature[:, [0, 3, 4]], base=16)
    feature = np.array(feature).astype(float)
    for i in range(len(nodesTo)):
        if nodesTo[i] in G.nodes :
            if G.nodes[nodesTo[i]]['feature'] == []:
                G.nodes[nodesTo[i]]['feature'] = feature[i]
            else:
                G.nodes[nodesTo[i]]['feature'] = G.nodes[nodesTo[i]][
                                        'feature'] + feature[i]
            if nodesFrom[i] in G.nodes:
                if G.nodes[nodesFrom[i]]['feature'] == []:
                    G.nodes[nodesFrom[i]]['feature'] = feature[i]
                else:
                    G.nodes[nodesFrom[i]]['feature'] = G.nodes[nodesFrom[i]][
                                                           'feature'] + feature[
                                                           i]
            elif G.number_of_nodes() < args.graph_size:
                G.add_node(nodesFrom[i], feature=feature[i])
                G.add_node(nodesTo[i], feature=feature[i])
                G.add_edge(nodesFrom[i], nodesTo[i])
            else:
                return G
        elif G.number_of_nodes()<args.graph_size:
            if nodesFrom[i] in G.nodes:
                if G.nodes[nodesFrom[i]]['feature'] == []:
                    G.nodes[nodesFrom[i]]['feature'] = feature[i]
                else:
                    G.nodes[nodesFrom[i]]['feature'] = G.nodes[nodesFrom[i]][
                                                           'feature'] + feature[
                                                           i]
                G.add_node(nodesTo[i], feature=feature[i])
                G.add_edge(nodesFrom[i], nodesTo[i])
            elif G.number_of_nodes() < args.graph_size-1:
                G.add_node(nodesFrom[i], feature=feature[i])
                G.add_node(nodesTo[i], feature=feature[i])
                G.add_edge(nodesFrom[i], nodesTo[i])
            else:
                return G
        else:
            return G



def get_features(G: nx.Graph):
    flist = []
    for i in G.nodes:
        flist.append(G.nodes[i]['feature'])
    return torch.tensor(flist).float()


def create_detection_domain(onehopdir: str, filename: str):
    """
    :param onehopdir: nonphishing_one_hop_nodes or phishing_one_hop_nodes
    :param filename: file in the  onehopdir
    :return: nx.Graph
    """
    G = nx.Graph()
    G.add_node(filename[:-4], feature=[])  # 保证G.nodes的第一个为源节点
    readAGraph(os.path.join(onehopdir, filename), G)
    for root, dirs, files in os.walk(
            find_twohop_neighbours(onehopdir, filename)):
        for i in range(len(files)):
            readAGraph(os.path.join(root, files[i]), G)
            if G.number_of_nodes() > args.graph_size:
                break
    g = dgl.from_networkx(G,node_attrs=['feature'])
    if g.num_nodes() < args.graph_size:
        g = dgl.add_nodes(g, args.graph_size - g.num_nodes())
    return g
