import sys
import os
import torch
import argparse
import random
import logging
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.gnn import GNN as GAT
from handlers.data_handler import DataHandler

parser = argparse.ArgumentParser(description='pytorch version of GAT')
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--aggr_func', type=str, default='MEAN')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--num_neg_samples', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gnn_type', default='gat',
                    choices=['gat', 'sage', 'gcn', 'gatv2'])
parser.add_argument('--heads', type=int, default=1)
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    args.device = torch.device("cuda" if args.cuda else "cpu")
    logging.info('Device:' + str(args.device))

    # Data
    data = DataHandler()
    data.load(args.data)

    # Model parameter
    num_layers = 2
    embed_size = 64
    layers = [data.feature_size] + [embed_size] * \
        num_layers + [data.label_size]

    # Model definition
    gat = GAT(layers, data.features, data.adj_lists, args).to(args.device)

    # Model optimizer
    gat.optimizer = torch.optim.SGD(gat.parameters(), lr=args.lr)

    # Model training
    times = []
    for epoch in range(args.num_epochs):
        losses = 0

        nodes = data.train_nodes
        np.random.shuffle(nodes)
        for batch in range(data.train_size // args.batch_size):
            batch_nodes = nodes[batch *
                                args.batch_size: (batch + 1) * args.batch_size]
            batch_labels = torch.LongTensor(
                data.labels[np.array(batch_nodes)]).to(args.device)

            start_time = time.time()

            gat.optimizer.zero_grad()
            loss = gat.loss(batch_nodes, batch_labels)
            loss.backward()
            gat.optimizer.step()

            end_time = time.time()
            times.append(end_time - start_time)

            loss_val = loss.data.item()
            losses += loss_val * len(batch_nodes)
            logging.debug(str(loss.data.item()))
            if (np.isnan(loss_val)):
                logging.error('Loss Val is NaN !!!')
                sys.exit()

        if epoch % 10 == 0:
            logging.info('Epoch: ' + str(epoch) + ' ' +
                         str(np.round(losses / data.train_size, 6)))

    gat.eval()
    val_output = gat.forward(data.valid_nodes)
    logging.info("Validation Macro F1:" + str(np.round(f1_score(
        data.labels[data.valid_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="macro"), 6)))
    logging.info("Validation Micro F1:" + str(np.round(f1_score(
        data.labels[data.valid_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="micro"), 6)))
    logging.info("Average batch time:" + str(np.round(np.mean(times), 6)))

    attentions = {}

    for edges, weights in gat.get_attention(data.train_nodes):
        edges = edges.t()
        weights = weights

        for edge, weight in zip(edges, weights):
            e0 = edge[0].item()
            e1 = edge[1].item()
            w = weight.item()

            attentions[e0] = attentions.get(e0, 0) + w
            attentions[e1] = attentions.get(e1, 0) + w

    print('Top 10 important nodes:')
    for node, weight in sorted(attentions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(node, weight)
