import os.path as osp
import huepy as hue
import json
import numpy as np
import torch
from torch.backends import cudnn

import sys
sys.path.append('./')
from configs import args_faster_rcnn

from lib.datasets import get_data_loader
from lib.model.faster_rcnn import get_model
from lib.utils.misc import lazy_arg_parse, Nestedspace, \
    resume_from_checkpoint
from lib.utils.evaluator import inference, detection_performance_calc, get_detection_results


def main(new_args, get_model_fn):

    args = Nestedspace()
    
    args.load_from_json(osp.join(osp.dirname(new_args.test.checkpoint_name), 'args.json'))
    args.from_dict(new_args.to_dict())  # override previous args

    device = torch.device('cuda')
    cudnn.benchmark = False

    print(hue.info(hue.bold(hue.lightgreen(
        'Working directory: {}'.format(args.path)))))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # dataloader
    # args.dataset = 'JTA'
    # gallery_loader_JTA, probe_loader_JTA = get_data_loader(args, train=False)
    args.dataset = 'CUHK'
    gallery_loader_CUHK, probe_loader_CUHK = get_data_loader(args, train=False)    
    args.dataset = 'PRW'
    gallery_loader_PRW, probe_loader_PRW = get_data_loader(args, train=False)

    gallery_loaders = [gallery_loader_CUHK, gallery_loader_PRW]
    probe_loaders = [probe_loader_CUHK, probe_loader_PRW]
    datasets = ['CUHK', 'PRW']

    # model
    model = get_model_fn(args, training=False,
                         pretrained_backbone=False)
    model.to(device)
    
    args.resume = osp.join(args.path, new_args.test.checkpoint_name)
    
    if osp.exists(args.test.checkpoint_name):
        # if not osp.exists(args.test.checkpoint_name.replace('.pth', '.json')):
            args, model, _, _ = resume_from_checkpoint(args, model)
            performance_all = {}
            for name, gallery_loader, probe_loader in zip(datasets, gallery_loaders, probe_loaders):   
                    
                    # print(hue.info(hue.bold(hue.lightgreen(
                    #     'Evaluating Model in epoch {}!!!'.format(epoch)))))
                    name_to_boxes, all_feats, probe_feats = \
                            inference(model, gallery_loader, probe_loader, device)
                    
                    print(hue.run('Evaluating detections:'))
                    precision, recall = detection_performance_calc(gallery_loader.dataset,
                                                                    name_to_boxes.values(),
                                                                    det_thresh=0.01)

                    print(hue.run('Evaluating search: '))
                    if name == "CUHK": gallery_size = 100
                    if name == "PRW": gallery_size = -1
                    if name == "JTA": gallery_size = 500

                    ret = gallery_loader.dataset.search_performance_calc(
                        gallery_loader.dataset, probe_loader.dataset,
                        name_to_boxes.values(), all_feats, probe_feats,
                        det_thresh=0.9, gallery_size=gallery_size,
                        ignore_cam_id=True,
                        remove_unlabel=True)

                    performance = {}
                    performance['mAP'] = ret['mAP']
                    performance['top_k'] = ret['accs'].tolist()
                    performance['precision'] = precision
                    performance['recall'] = recall
                    print(performance)



if __name__ == '__main__':
    arg_parser = args_faster_rcnn()
    new_args = lazy_arg_parse(arg_parser)
    main(new_args, get_model)
