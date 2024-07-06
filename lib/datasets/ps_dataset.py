import os.path as osp
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import huepy as hue
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pdb
class PersonSearchDataset(object):
    def __init__(self, root, root_target, transforms, mode='train', debug = 0, args=None):
        super(PersonSearchDataset, self).__init__()
        
        if args is not None:        
            self.args = args
            
        self.brisque = {}

        self.unique_id = {}
        self.unique_seq = {}
        self.cnt = 1
        self.cnt_s = 1
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.data_path = self.get_data_path()
        # test = gallery + probe
        assert self.mode in ('train', 'test', 'probe')

        self.imgs = self._load_image_set_index()
        if self.mode in ('train', 'test'):
            if root.split('/')[-1] == 'JTA' and args.tsne:
                self.record = self.gt_roidb_tsne()
            else:
                self.record = self.gt_roidb()
                
        else:
            self.record = self.load_probes()
        if self.mode in ('train'):
            if debug:
                print(hue.info(hue.bold(hue.lightgreen(
                'Loading 1 image sample for debugging'))))
                self.imgs=self.imgs[:30]
                self.record=self.record[:30]
        
    def get_data_path(self):
        raise NotImplementedError

    def _load_image_set_index(self):
        raise NotImplementedError

    def gt_roidb(self):
        raise NotImplementedError

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, idx):
        # since T.to_tensor() is in transforms, the returned image pixel range is in (0, 1)
        # label_pids.min() = 1 if mode = 'train', else label_pids unchanged.
        sample = self.record[idx]
        im_name = sample['im_name']
        img_path = osp.join(self.data_path, im_name)
        img = Image.open(img_path).convert('RGB')

        boxes = torch.as_tensor(sample['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(sample['gt_pids'], dtype=torch.int64)

        if self.mode == 'train': 
            cnt = torch.as_tensor(sample['cnt'], dtype=torch.int64)
            brisque = []
            for c in cnt:
                brisque.append(self.brisque[c.item()])
            brisque = torch.as_tensor(brisque, dtype=torch.float32)
            
            sequences = torch.as_tensor(sample['seq'], dtype=torch.int64)
            imcnt = torch.as_tensor(sample['imcnt'], dtype=torch.int64)
            
            target = dict(
                    im_name=im_name,
                    boxes=boxes,
                    brisque=brisque,
                    labels=labels,
                    seq=sequences,
                    flipped='True',
                    imcnt=imcnt,
                    cnt=cnt,
                    )
        else:
            target = dict(boxes=boxes,
                    labels=labels,
                    flipped='True',
                    im_name=im_name,
                    )

        if self.transforms != None:
            img, target = self.transforms(img, target)

        if self.mode == 'train':
            target['labels'] = \
                self._adapt_pid_to_cls(target['labels'])

        return img, target

    def __len__(self):
        return len(self.record)

    def _adapt_pid_to_cls(self, label_pids, upid=5555):
        raise NotImplementedError

    def load_probes(self):
        raise NotImplementedError