import os 
import sys
sys.path.append('./')

from datetime import datetime
import numpy as np
import huepy as hue
import torch
import builtins

from configs import args_faster_rcnn
from lib.model.faster_rcnn import get_model
from lib.datasets import get_data_loader
from lib.utils.misc import Nestedspace, get_optimizer, get_lr_scheduler
from lib.utils.trainer2 import Trainer
from lib.model.resnet_backbone import resnet_backbone

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

CUDA_LAUNCH_BLOCKING=1

def main(gpu, args, get_model_fn):

    if torch.cuda.device_count() > 1:
        # print once
        if gpu != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    device = f'cuda:{gpu}'
    args.device = device
    args.gpu = gpu

    ## Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ## Determine checkpoint path and files
    print("Memo :", args.memo)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.path = os.path.join(args.path)
    working_path = args.path
    if args.debug == 0:
        try:
            os.makedirs(working_path)
        except:
            pass 
        print(hue.info(hue.bold(hue.lightgreen('Train mode'))))
        print(hue.info(hue.bold(hue.lightgreen('Working directory: {}'.format(working_path)))))
    
        args.export_to_json(os.path.join(working_path, 'args.json'))
        ## Load test set
    elif args.debug == 1:
        print(hue.info(hue.bold(hue.lightgreen('Debug mode'))))
        try:
            os.makedirs(working_path)
        except:
            pass 
            print(hue.info(hue.bold(hue.lightgreen('Working directory: {}'.format(working_path)))))
            args.export_to_json(os.path.join(working_path, 'args.json'))


    ## Load model
    model = get_model_fn(args, training=True, pretrained_backbone=True)
    
               
    if torch.cuda.device_count() > 1:
        # gpu grouping                         
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1115', world_size=torch.cuda.device_count(), rank=gpu)
        dist.barrier()
        
        # sync batchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        
        args.train.batch_size = args.train.batch_size // dist.get_world_size()
        args.train.lr = args.train.lr 
        args.train.lr_decay_gamma = args.train.lr_decay_gamma 
    
    # single gpu
    else:
        model.cuda(gpu)
    
    if torch.cuda.device_count() > 1:
        # ddp model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu], output_device = gpu)

    print(f"Using { torch.cuda.device_count() } devices")
    print(f"Batch Size per gpu: {args.train.batch_size}")

    train_loader = get_data_loader(args, train=True)
    
     
    args.dataset = 'CUHK'
    gallery_loader1, probe_loader1 = get_data_loader(args, train=False)
    args.dataset = 'PRW'
    gallery_loader2, probe_loader2 = get_data_loader(args, train=False)
    gallery_loader3, probe_loader3 = "None", "None"

    ## Set optimizer and scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    
    ## Load the existing models if possible
    
    if (args.train.resume_name != None) :
        if os.path.exists(args.train.resume_name):
            print(f"Loading from {args.train.resume_name}")
            checkpoint = torch.load(args.train.resume_name)
            args.train.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
                    
            if optimizer != None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler != None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print(hue.good('loaded checkpoint %s' % (args.train.resume_name)))
            print(hue.info('model was trained for %s epochs' % (args.train.start_epoch)))
        else:
            print("check resume_name")

    ## Define and run trainer
    trainer = Trainer(args, model, train_loader, gallery_loader1, probe_loader1, gallery_loader2, probe_loader2, gallery_loader3, probe_loader3, optimizer, lr_scheduler, device, working_path)
    trainer.run()

if __name__ == '__main__':
    arg_parser = args_faster_rcnn()
    args = arg_parser.parse_args(namespace=Nestedspace())

    ngpus_per_node = torch.cuda.device_count()

    if ngpus_per_node > 1:
        mp.spawn(main, nprocs = ngpus_per_node, args = (args,get_model))
    else:
        main(0, args, get_model)
