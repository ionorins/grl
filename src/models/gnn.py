from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GATv2Conv, SuperGATConv
from layers.aggregator import Aggregator
from layers.sampler import Sampler
import sys
import logging
import numpy as np

import torch
import torch.nn as nn

sys.path.append('..')


class GNN(nn.Module):

    def __init__(self, layers, in_features, adj_lists, args):
        super(GNN, self).__init__()

        self.layers = layers
        self.num_layers = len(layers) - 2
        self.in_features = torch.Tensor(in_features).to(args.device)
        self.adj_lists = adj_lists
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device

        if args.gnn_type not in ['gat', 'gatv2', 'supergat']:
            args.heads = 1

        layer = {
            'gat': GATConv,
            'gcn': GCNConv,
            'sage': SAGEConv,
            'gatv2': GATv2Conv,
            'supergat': SuperGATConv,
        }[args.gnn_type]

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                layer(
                    layers[i] * (args.heads if i > 0 else 1),
                    layers[i + 1],
                    heads=args.heads,
                )
            )
        self.sampler = Sampler(adj_lists)
        self.aggregator = Aggregator()

        self.weight = nn.Parameter(torch.Tensor(
            args.heads * layers[-2], layers[-1]))
        self.xent = nn.CrossEntropyLoss()

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            try:
                nn.init.xavier_uniform_(param)
            except:
                nn.init.zeros_(param)

    def forward(self, nodes):
        features = self.in_features
        edge_index = []
        for node in nodes:
            neighbors = [n for n in self.adj_lists[node] if n in nodes]
            for neighbor in neighbors:
                edge_index.append([node, neighbor])
        edge_index = torch.tensor(edge_index).t().to(self.device)

        for i in range(self.num_layers):
            features = self.convs[i].forward(
                x=features, edge_index=edge_index)

        return nn.functional.log_softmax(torch.matmul(features[nodes], self.weight), 1)

    def loss(self, nodes, labels=None):
        preds = self.forward(nodes)
        return self.xent(preds, labels.squeeze())

    def get_embeds(self, nodes):
        features = self.in_features
        edge_index = []
        for node in nodes:
            neighbors = [n for n in self.adj_lists[node] if n in nodes]
            for neighbor in neighbors:
                edge_index.append([node, neighbor])
        edge_index = torch.tensor(edge_index).t().to(self.device)

        for i in range(self.num_layers):
            features = self.convs[i].forward(
                x=features, edge_index=edge_index)
        return features[nodes].data.numpy()

    def get_attention(self, nodes):
        features = self.in_features
        edge_index = []
        for node in nodes:
            neighbors = [n for n in self.adj_lists[node] if n in nodes]
            for neighbor in neighbors:
                edge_index.append([node, neighbor])
        edge_index = torch.tensor(edge_index).t().to(self.device)

        for i in range(self.num_layers):
            features, att_weights = self.convs[i].forward(
                x=features, edge_index=edge_index, return_attention_weights=True)
            yield att_weights

    def get_attention_dict(self, nodes):
        attentions = {}

        for edges, weights in self.get_attention(nodes):
            edges = edges.t()
            weights = weights

            for edge, weight in zip(edges, weights):
                e0 = edge[0].item()
                e1 = edge[1].item()
                w = weight.sum().item()

                attentions[e0] = attentions.get(e0, 0) + w
                attentions[e1] = attentions.get(e1, 0) + w

        return attentions
