import os
import argparse
import torch

import dist_utils
import dist_train


def process_wrapper(rank, args, func):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    # os.environ['NCCL_DEBUG']='INFO'
    # os.environ['NCCL_DEBUG_SUBSYS']='ALL'
    # os.environ['NCCL_P2P_DISABLE']='1'
    # os.environ['NCCL_ALGO'] = 'Ring'
    # os.environ['NCCL_MIN_NCHANNELS'] = '1'
    # os.environ['NCCL_MAX_NCHANNELS'] = '1'

    env = dist_utils.DistEnv(rank, args.nprocs, args.backend)
    env.half_enabled = False
    env.csr_enabled = False

    combined_log_path = os.path.join(args.log_dir, f"{args.dataset}_partition_{args.nprocs}.log")
    cur_log_path = os.path.join(args.log_dir, f"{args.dataset}_partition_{args.nprocs}_{rank}.log")
    dist_utils.create_logger(combined_log_path, cur_log_path)

    func(env, args)


if __name__ == "__main__":
    num_GPUs = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int, default=num_GPUs if num_GPUs>1 else 8)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--backend", type=str, default='nccl' if num_GPUs>1 else 'gloo')
    parser.add_argument("--dataset", type=str, default='ogbn-products')
    parser.add_argument("--log_dir", type=str, default='logs')
    args = parser.parse_args()

    # create the directory for log
    os.makedirs(args.log_dir, exist_ok=True)

    process_args = (args, dist_train.main)
    torch.multiprocessing.spawn(process_wrapper, process_args, args.nprocs)
