import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
from dist_utils.timer import *


try:
    from spmm_cpp import spmm_cusparse_coo, spmm_cusparse_csr
    def spmm(A,B,C): 
        if DistEnv.env.csr_enabled:
            spmm_cusparse_csr(A.crow_indices().int(), A.col_indices().int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
        else:
            spmm_cusparse_coo(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
except ImportError as e:
    print('no spmm cpp:', e)
    spmm = lambda A,B,C: C.addmm_(A,B)


from collections import defaultdict
g_cache = defaultdict(dict)
g_cache_enabled = {'ForwardL1': True, 'ForwardL2': True,
                   'BackwardL1': False, 'BackwardL2': False }

g_bcast_counter = defaultdict(lambda: defaultdict(int))
g_epoch_counter = defaultdict(int)

def use_cache(tag, src):
    F_L1 = tag == 'ForwardL1' and g_bcast_counter[tag][src]>0 # if there is enough gpu mem
    F_L2 = tag == 'ForwardL2' and (g_bcast_counter[tag][src]>50 and g_epoch_counter[tag]%2==0)
    use = g_cache_enabled[tag] and (F_L1 or F_L2)
    if use:
        assert(src in g_cache[tag])
    return use


def cached_broadcast(local_adj_parts, local_feature, tag):
    env = DistEnv.env
    z_loc = torch.zeros_like(local_feature)
    feature_bcast = torch.zeros_like(local_feature)
    g_epoch_counter[tag] += 1
    
    for src in range(env.world_size):
        if src == env.rank:
            feature_bcast = local_feature.clone()

        with comm_timer.timer(f"broadcast: {tag}"):
            if not use_cache(tag, src):
                with comm_timer.timer(f'broadcast: {tag} {src}'):
                    dist.broadcast(feature_bcast, src=src)
                    g_bcast_counter[tag][src] += 1
                    if g_cache_enabled[tag]:
                        g_cache[tag][src] = feature_bcast.clone()
            else:
                feature_bcast = g_cache[tag][src]

        with comp_timer.timer(f"spmm"):
            spmm(local_adj_parts[src], feature_bcast, z_loc)
    return z_loc


class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, adj_parts, tag):
        ctx.save_for_backward(features, weight)
        ctx.adj_parts = adj_parts
        ctx.tag = tag

        name = 'Forward' + tag
        z_local = cached_broadcast(adj_parts, features, name)

        with comp_timer.timer(f"forward_mm: {name}"):
            z_local = torch.mm(z_local, weight)

        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        features,  weight = ctx.saved_tensors

        name = 'Backward' + ctx.tag

        ag = cached_broadcast(ctx.adj_parts, grad_output, name)

        with comp_timer.timer(f"backward_mm: {name}"):
            grad_features = torch.mm(ag.to(dtype=torch.float), weight.t())
            grad_weight = torch.mm(features.t(), ag)

        with reduce_timer.timer(f"all_reduce: {name}"):
            DistEnv.env.all_reduce_sum(grad_weight)

        return grad_features, grad_weight, None, None


class CachedGCN(nn.Module):
    def __init__(self, g, env, hidden_dim=16):
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.features.size(1), g.num_classes
        torch.manual_seed(0)
        self.weight1 = nn.Parameter(torch.rand(in_dim, hidden_dim).to(env.device))
        self.weight2 = nn.Parameter(torch.rand(hidden_dim, hidden_dim).to(env.device))
        self.weight3 = nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device))

    def forward(self, features):
        hidden_features = F.relu(DistGCNLayer.apply(features, self.weight1, self.g.adj_parts, 'L1'))
        outputs = DistGCNLayer.apply(hidden_features, self.weight3, self.g.adj_parts,  'L2')
        return outputs
