import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from sklearn.metrics import f1_score

from coo_graph import Parted_COO_Graph
from models import CachedGCN

from dist_utils.timer import *


def f1(y_true, y_pred, multilabel=True):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if multilabel:
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")


def train(g, env, total_epoch):
    logging.info("Rank={} Start training".format(env.rank))

    model = CachedGCN(g, env, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        loss_func = nn.CrossEntropyLoss()
    elif g.labels.dim()==2:
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    comp_dur, comm_dur, reduce_dur, wait_dur = [], [], [], []
    for epoch in range(total_epoch):
        with comp_timer.timer(f"train"):
            with autocast(env.half_enabled):
                with comp_timer.timer(f"forward"):
                    outputs = model(g.features)

                optimizer.zero_grad()

                with comp_timer.timer(f"backward"):
                    if g.local_labels[g.local_train_mask].size(0) > 0:
                        loss = loss_func(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
                    else:
                        loss = (outputs * 0).sum()

                    loss.backward()

                with comp_timer.timer(f"optimizer"):
                    optimizer.step()

        comp_dur.append(comp_timer.tot_time())
        reduce_dur.append(reduce_timer.tot_time())
        comm_dur.append(comm_timer.tot_time())
        wait_dur.append(wait_timer.tot_time())

        comm_timer.clear()
        wait_timer.clear()
        reduce_timer.clear()
        comp_timer.clear()

        logging.info(
            "Rank-{} | Epoch {:05d} | Comp(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Sum(s) {:.4f} | Loss {:.4f}".format(
                env.rank,
                epoch,
                comp_dur[-1],
                comm_dur[-1],
                reduce_dur[-1],
                (comp_dur[-1]) + (comm_dur[-1]) + (reduce_dur[-1]),
                loss.item(),
                )
        )

        if epoch % 10 == 0 or epoch == total_epoch-1:
            all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                mask = g.train_mask
                logging.info("Rank-{} | Epoch {:05d} | f1{:.4f}".format(
                    env.rank,
                    epoch,
                    f1(g.labels[mask], torch.sigmoid(all_outputs[mask]))
                ))
            else:
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                logging.info("Rank-{} | Epoch {:05d} | Train_Acc {:.4f} | Val_Acc {:.4f} | Test_Acc {:.4f}".format(
                    env.rank,
                    epoch,
                    acc(g.train_mask),
                    acc(g.val_mask),
                    acc(g.test_mask)
                ))


def main(env, args):
    g = Parted_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
    train(g, env, total_epoch=args.epoch)
