# -*- coding: utf-8 -*-
# Reference:
# https://github.com/pytorch/vision/blob/fe3b4c8f2c/references/detection/utils.py

import argparse
import sys
import torch
import huepy as hue
import json
import numpy as np
from collections import OrderedDict

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    try:
        os.makedirs(osp.dirname(fpath))
    except:
        pass
    _obj = obj.copy()
    for k, v in _obj.items():
        if isinstance(v, np.ndarray):
            _obj.pop(k) 
    with open(fpath, 'w') as f:
        json.dump(_obj, f, indent=4, separators=(',', ': '))


class Nestedspace(argparse.Namespace):

    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group, name = name.split('.', 1)
            try:
                ns = self.__dict__[group]
            except KeyError:
                raise AttributeError
            return getattr(ns, name)
        else:
            raise AttributeError

    def to_dict(self, args=None, prefix=None):
        out = {}
        args = self if args is None else args
        for k, v in args.__dict__.items():
            if isinstance(v, Nestedspace):
                out.update(self.to_dict(v, prefix=k))
            else:
                if prefix != None:
                    out.update({prefix + '.' + k: v})
                else:
                    out.update({k: v})
        return out

    def from_dict(self, dic):
        for k, v in dic.items():
            self.__setattr__(k, v)

    def export_to_json(self, file_path):
        write_json(self.to_dict(), file_path)

    def load_from_json(self, file_path):
        self.from_dict(read_json(file_path))


def lazy_arg_parse(parser):
    '''
    Only parse the given flags.
    '''
    def parse_known_args():
        args = sys.argv[1:]
        namespace = Nestedspace()

        try:
            namespace, args = parser._parse_known_args(args, namespace)
            if hasattr(namespace, '_unrecognized_args'):
                args.extend(getattr(namespace, '_unrecognized_args'))
                delattr(namespace, '_unrecognized_args')
            return namespace, args
        except argparse.ArgumentError:
            err = sys.exc_info()[1]
            parser.error(str(err))

    args, argv = parse_known_args()
    if argv:
        msg = _('unrecognized arguments: %s')
        parser.error(msg % ' '.join(argv))
    return args


def ship_data_to_cuda(batch, device):
    f = lambda sample: ship_data_to_cuda_singe_sample(
        sample[0], sample[1], device=device)
    return tuple(map(list, zip(*map(f, batch))))


def ship_data_to_cuda_singe_sample(img, target, device):
    img = img.cuda(device, non_blocking = True)
    if target != None:
        target['boxes'] = target['boxes'].cuda(device, non_blocking = True)
        target['labels'] = target['labels'].cuda(device, non_blocking = True)
        if 'seq' in target:
            target['seq'] = target['seq'].cuda(device, non_blocking = True)
        if 'occlusion' in target:
            target['occlusion'] = target['occlusion'].cuda(device, non_blocking = True)
        if 'heatmaps' in target:
            target['heatmaps'] = target['heatmaps'].cuda(device, non_blocking = True)
    return img, target


def resume_from_checkpoint(args, model, optimizer=None, lr_scheduler=None):
    load_name = args.test.checkpoint_name
    checkpoint = torch.load(load_name)
    if 'roi_heads.dlut2' in checkpoint['model']:
        checkpoint['model'].pop('roi_heads.dlut2')

    args.train.start_epoch = checkpoint['epoch']
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        checkpoint['model']['roi_heads.dlut'] = torch.zeros_like(model.roi_heads.dlut)
        checkpoint['model']['roi_heads.domain_proj.weight'] = torch.zeros_like(model.roi_heads.domain_proj.weight)
        checkpoint['model']['roi_heads.domain_proj.bias'] = torch.zeros_like(model.roi_heads.domain_proj.bias)
        checkpoint['model']["roi_heads.FE.conv_1.weight"] = torch.zeros_like(model.roi_heads.FE.conv_1.weight)
        checkpoint['model']["roi_heads.FE.conv_1.bias"] = torch.zeros_like(model.roi_heads.FE.conv_1.bias)
        checkpoint['model']["roi_heads.FE.bn1.weight"] = torch.zeros_like(model.roi_heads.FE.bn1.weight)
        checkpoint['model']["roi_heads.FE.bn1.bias"] = torch.zeros_like(model.roi_heads.FE.bn1.bias)
        checkpoint['model']["roi_heads.FE.bn1.running_mean"] = torch.zeros_like(model.roi_heads.FE.bn1.running_mean)
        checkpoint['model']["roi_heads.FE.bn1.running_var"] = torch.zeros_like(model.roi_heads.FE.bn1.running_var)
        checkpoint['model']["roi_heads.FE.conv_2.weight"] = torch.zeros_like(model.roi_heads.FE.conv_2.weight)
        checkpoint['model']["roi_heads.FE.conv_2.bias"] = torch.zeros_like(model.roi_heads.FE.conv_2.bias)
        checkpoint['model']["roi_heads.FE.bn2.weight"] = torch.zeros_like(model.roi_heads.FE.bn2.weight)
        checkpoint['model']["roi_heads.FE.bn2.bias"] = torch.zeros_like(model.roi_heads.FE.bn2.bias)
        checkpoint['model']["roi_heads.FE.bn2.running_mean"] = torch.zeros_like(model.roi_heads.FE.bn2.running_mean)
        checkpoint['model']["roi_heads.FE.bn2.running_var"] = torch.zeros_like(model.roi_heads.FE.bn2.running_var)
        checkpoint['model']["roi_heads.FE.conv_3.weight"] = torch.zeros_like(model.roi_heads.FE.conv_3.weight)
        checkpoint['model']["roi_heads.FE.conv_3.bias"] = torch.zeros_like(model.roi_heads.FE.conv_3.bias)
        checkpoint['model']["roi_heads.FE.bn3.weight"] = torch.zeros_like(model.roi_heads.FE.bn3.weight)
        checkpoint['model']["roi_heads.FE.bn3.bias"] = torch.zeros_like(model.roi_heads.FE.bn3.bias)
        checkpoint['model']["roi_heads.FE.bn3.running_mean"] = torch.zeros_like(model.roi_heads.FE.bn3.running_mean)
        checkpoint['model']["roi_heads.FE.bn3.running_var"] = torch.zeros_like(model.roi_heads.FE.bn3.running_var)
        checkpoint['model']["roi_heads.FE.conv_4.weight"] = torch.zeros_like(model.roi_heads.FE.conv_4.weight)
        checkpoint['model']["roi_heads.FE.conv_4.bias"] = torch.zeros_like(model.roi_heads.FE.conv_4.bias)
        checkpoint['model']["roi_heads.FE.fc.weight"] = torch.zeros_like(model.roi_heads.FE.fc.weight)
        checkpoint['model']["roi_heads.FE.fc.bias"] = torch.zeros_like(model.roi_heads.FE.fc.bias)
        model.load_state_dict(checkpoint['model'])
    # except:
    #     new_state_dict = OrderedDict()
    #     for n, v in checkpoint['model'].items():
    #         name = n.replace("module.","") 
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler != None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    print(hue.good('loaded checkpoint %s' % (load_name)))
    print(hue.info('model was trained for %s epochs' % (args.train.start_epoch)))
    return args, model, optimizer, lr_scheduler


def get_optimizer(args, model):
    lr = args.train.lr
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': args.train.weight_decay}]

    optimizer = torch.optim.SGD(params, momentum=args.train.momentum)
    return optimizer


def get_lr_scheduler(args, optimizer):
    if args.train.lr_decay_milestones != None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.train.lr_decay_milestones,
            gamma=args.train.lr_decay_gamma)
    else:
        if args.train.lr_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max=3200, eta_min=0)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train.lr_decay_step,
            gamma=args.train.lr_decay_gamma)


    return scheduler


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def lucky_bunny(i):
    print('')
    print('|￣￣￣￣￣￣￣￣|')
    print('|    TRAINING    |')
    print('|     epoch      |')
    print('|       ' + hue.bold(hue.green(str(i))) + '        |')
    print('| ＿＿＿_＿＿＿＿|')
    print(' (\__/) ||')
    print(' (•ㅅ•) || ')
    print(' / 　 づ')
    print('')
