import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch.nn as nn
import torch
from .misc import ship_data_to_cuda

from PIL import Image, ImageDraw
import torchvision.transforms as T
import os.path as osp
import os

@torch.no_grad()
def get_both_features(model, data_loader, device):
    
    features_id = []
    features_dom = []
    
    sequences = []
    pids = []
    with torch.no_grad():
        for data in tqdm(data_loader, ncols=0):
            images, targets = ship_data_to_cuda(data, device)
            
            if data_loader.dataset.__class__.__name__ == "PRW":
                sequence = torch.tensor(
                    [int(t['im_name'][1]) for t in targets for i in range(len(t['boxes']))])
            else:
                sequence = torch.cat([t['seq'] for t in targets])
                
            pid = torch.cat([t['labels'] for t in targets])
            
            feat_id, feat_dom = model.ex_both_feat(images, targets)
            
            features_id.append(torch.cat(feat_id))
            features_dom.append(feat_dom)
            sequences.append(sequence)
            pids.append(pid)
    
    features_id = torch.cat(features_id).cpu().numpy()
    features_dom = torch.cat(features_dom).cpu().numpy()
    sequences = torch.cat(sequences).cpu().numpy()
    pids = torch.cat(pids).cpu().numpy()
        
    return features_id, features_dom, sequences, pids

@torch.no_grad()
def get_domain_features(model, data_loader, device):
    
    features = []
    sequences = []
    with torch.no_grad():
        for data in tqdm(data_loader, ncols=0):
            images, targets = ship_data_to_cuda(data, device)
            sequence = torch.cat([t['seq'] for t in targets])
            
            feat_dom = model.ex_domain_feat(images, targets)
            
            features.append(feat_dom)
            sequences.append(sequence)
    
    features = torch.cat(features).cpu().numpy()
    sequences = torch.cat(sequences).cpu().numpy()
        
    return features, sequences

@torch.no_grad()
def get_id_features(model, data_loader, device):
    
    features = []
    sequences = []    
    pids = []

    with torch.no_grad():
        for data in tqdm(data_loader, ncols=0):
            images, targets = ship_data_to_cuda(data, device)
            
            if data_loader.dataset.__class__.__name__ == "PRW":
                sequence = torch.tensor(
                    [int(t['im_name'][1]) for t in targets for i in range(len(t['boxes']))])
            else:
                sequence = torch.cat([t['seq'] for t in targets])
            pid = torch.cat([t['labels'] for t in targets])
            
            
            feat_id = model.ex_feat(images, targets, mode='det')
            
            features.append(torch.cat(feat_id))
            sequences.append(sequence)
            pids.append(pid)
    
    features = torch.cat(features).cpu().numpy()
    sequences = torch.cat(sequences).cpu().numpy()
    pids = torch.cat(pids).cpu().numpy()
        
    return features, sequences, pids

@torch.no_grad()
def get_detection_results(model, gallery_loader, probe_loader, device, src_dir):
    model.eval()

    dir = osp.join('materials', 'det', osp.dirname(src_dir))
    
    os.makedirs(dir, exist_ok=True)
    for data in tqdm(gallery_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        # Target != used in inference mode.
        outputs = model(images)

        img = images[0]
        output = outputs[0]
        target = targets[0]
        
        img_pil = T.ToPILImage()(img)
        draw = ImageDraw.Draw(img_pil)
        for bbox in output['boxes']:
            draw.rectangle((bbox.cpu().numpy()), outline=(255,0,0), width=3)
        img_pil.save(osp.join(dir, target['im_name']))    
        
            
            
@torch.no_grad()
def inference(model, gallery_loader, probe_loader, device):
    model.eval()

    # # test time bn unfreeze bn
    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         if hasattr(module, 'weight'):
    #             module.weight.requires_grad_(True)
    #         if hasattr(module, 'bias'):
    #             module.bias.requires_grad_(True)
    #         module.train()

    cpu = torch.device('cpu')

    im_names, all_boxes, all_feats = [], [], []
    for data in tqdm(gallery_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        # Target != used in inference mode.
        outputs = model(images)

        for o, t in zip(outputs, targets):
            im_names.append(t['im_name'])
            box_w_scores = torch.cat([o['boxes'],
                                      o['scores'].unsqueeze(1)],
                                     dim=1)
            all_boxes.append(box_w_scores.to(cpu).numpy())
            all_feats.append(o['embeddings'].to(cpu).numpy())
    
    probe_feats = []
    for data in tqdm(probe_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        try: embeddings = model.ex_feat(images, targets, mode='det')
        except: embeddings = model.module.ex_feat(images, targets, mode='det')
        for em in embeddings:
            probe_feats.append(em.to(cpu).numpy())


    name_to_boxes = OrderedDict(zip(im_names, all_boxes))

    return name_to_boxes, all_feats, probe_feats


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def detection_performance_calc(dataset, gallery_det, det_thresh=0.5, iou_thresh=0.5,
                               labeled_only=False):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image

    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert len(dataset) == len(gallery_det)
    gt_roidb = dataset.record

    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for gt, det in zip(gt_roidb, gallery_det):
        gt_boxes = gt['boxes']
        if labeled_only:
            inds = np.where(gt['gt_pids'].ravel() > 0)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        if det != []:
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
        else:
            num_det = 0
        if num_det == 0:
            count_gt += num_gt
            continue
        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
        tfmat = (ious >= iou_thresh)
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            if tfmat[:, j].any():
                y_true.append(True)
            else:
                y_true.append(False)
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / count_gt
    ap = average_precision_score(y_true, y_score) * det_rate
    precision, recall, __ = precision_recall_curve(y_true, y_score)
    recall *= det_rate

    print('{} detection:'.format('labeled only' if labeled_only else
                                 'all'))
    print('  recall = {:.2%}'.format(det_rate))
    if not labeled_only:
        print('  ap = {:.2%}'.format(ap))
    return ap, det_rate
    # return precision, recall
