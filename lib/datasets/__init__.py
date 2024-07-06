import threading
import sys
if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

import torch

from .cuhk_sysu import CUHK_SYSU
from .prw import PRW
from .jta import JTA

from ..utils.transforms import get_transform
from ..utils.group_by_aspect_ratio import create_aspect_ratio_groups,\
    GroupedBatchSampler
import torch.distributed as dist
import random
class PrefetchGenerator(threading.Thread):

    def __init__(self, generator, max_prefetch=1):
        super(PrefetchGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class PrefetchDataLoader(torch.utils.data.DataLoader):

    def __iter__(self):
        return PrefetchGenerator(
            super(PrefetchDataLoader, self).__iter__()
        )

def collate_fn(x):
    return x

def get_dataset(args, train=True):
    paths = {
        'CUHK': ('/root/dataset/PersonSearch/cuhk_sysu/', None, CUHK_SYSU),
        'PRW': ('/root/dataset/PersonSearch/PRW-v16.04.20/', None, PRW),
        'JTA': ('/root/workplace/PersonSearch/Dataset/JTA', None, JTA),
        
    }
    p, p_target, ds_cls = paths[args.dataset]
    import pdb
    if train:
        if args.debug:
            train_set = ds_cls(p, p_target, get_transform(args.train.use_flipped, resize=args.disable_resize),
                            mode='train', debug=args.debug, args=args)
            return train_set
        train_set = ds_cls(p, p_target, get_transform(args.train.use_flipped, resize=args.disable_resize),
                        mode='train', args=args)
        return train_set
    else:
        test_set = ds_cls(p, p_target, get_transform(False), mode='test')
        probe_set = ds_cls(p, p_target, get_transform(False), mode='probe')
        return test_set, probe_set


def get_data_loader(args, train=True, drop_last=True):

    dataset = get_dataset(args, train)
    # args.train.batch_size = 20
    if train:
        if torch.cuda.device_count() == 1:
            train_sampler = torch.utils.data.RandomSampler(dataset)

            if args.train.aspect_grouping >= 0:
                group_ids = create_aspect_ratio_groups(dataset, k=args.train.aspect_grouping)
                train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.train.batch_size)
            else:
                train_batch_sampler = torch.utils.data.BatchSampler(
                    train_sampler, args.train.batch_size, drop_last=False)
            
            data_loader = PrefetchDataLoader(
                dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers,
                collate_fn=collate_fn)

        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                                dataset,
                                num_replicas=dist.get_world_size(),
                                shuffle = True)
            
            data_loader = PrefetchDataLoader(
                dataset, sampler = train_sampler, batch_size = args.train.batch_size, num_workers=args.num_workers,
                collate_fn=collate_fn, pin_memory = True, drop_last=False)

        return data_loader

    else: # test
        test_sampler = torch.utils.data.SequentialSampler(dataset[0])
        probe_sampler = torch.utils.data.SequentialSampler(dataset[1])
        # random.seed(3)
        # random.shuffle(dataset[0].record)
        data_loader_test = PrefetchDataLoader(
            dataset[0], batch_size=args.test.batch_size,
            sampler=test_sampler, num_workers=args.num_workers,
            collate_fn=collate_fn)
        data_loader_probe = PrefetchDataLoader(
            dataset[1], batch_size=args.test.batch_size,
            sampler=probe_sampler, num_workers=args.num_workers,
            collate_fn=collate_fn)
        return data_loader_test, data_loader_probe
