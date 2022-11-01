import time
import torch.distributed as dist
from contextlib import contextmanager
import logging


class CommTimer(object):

    def __init__(self):
        super(CommTimer, self).__init__()
        self._time = {}

    @contextmanager
    def timer(self, name):
        t0 = time.time()
        yield
        t1 = time.time()
        if name in self._time:
            (t0_ori, t1_ori) = self._time[name]
            self._time[name] = (t0_ori, t1_ori + t1 - t0)
        self._time[name] = (t0, t1)

    def tot_time(self, blacklist=None):
        tot = 0
        for name in self._time:
            if blacklist is not None and name in blacklist:
                continue
            (t0, t1) = self._time[name]
            tot += t1 - t0
        return tot

    def print_time(self):
        rank = dist.get_rank()
        for (k, (t0, t1)) in self._time.items():
            logging.info(f'Rank-{rank} | {k}: {t1 - t0} seconds.')

    def clear(self):
        self._time = {}
