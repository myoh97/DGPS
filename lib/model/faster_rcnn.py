from collections import OrderedDict

import numpy as np
import copy
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torchvision import transforms as tf
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from .generalized_rcnn import GeneralizedRCNN

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork, concat_box_prediction_layers
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from typing import List, Optional, Dict, Tuple

from .resnet_backbone import resnet_backbone
from torch import nn, autograd

from ..loss import OIMLoss
# from ..loss import OIMLossPart
from torch import autograd
import math

from PIL import Image
import pdb
class FasterRCNN(GeneralizedRCNN):
    """
    See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L26
    """
    def __init__(self, args,
                 backbone,
                 num_classes=None, 
                 # transform parameters
                 min_size=900, max_size=1500,
                 image_mean=[0.485, 0.456, 0.406], 
                 image_std=[0.229, 0.224, 0.225],
                 # RPN parameters
                 rpn_anchor_generator=None, 
                 rpn_head=None,
                 rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, 
                 feat_head=None, 
                 box_predictor=None,
                 box_score_thresh=0.0, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 embedding_head=None, 
                 embedding_head_dom=None, 
                 reid_regressor=None,
                 ):
        self.args = args
        
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                'backbone should contain an attribute out_channels '
                'specifying the number of output channels (assumed to be the '
                'same for all the levels)')
        
        if rpn_anchor_generator is None:
            raise ValueError('rpn_anchor_generator should be specified manually.')
        
        if rpn_head is None:
            raise ValueError('rpn_head should be specified manually.')
        
        if box_roi_pool is None:
            raise ValueError('box_roi_pool should be specified manually.')
        
        if feat_head is None:
            raise ValueError('feat_head should be specified manually.')
        
        if box_predictor is None:
            raise ValueError('box_predictor should be specified manually.')

        if embedding_head is None:
            raise ValueError('embedding_head should be specified manually.')


        # Construct RPN
        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = self._set_rpn(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # Construct ROI head 
        roi_heads = self._set_roi_heads(
            args,
            embedding_head, embedding_head_dom, reid_regressor,
            box_roi_pool, feat_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        # Construct image transformer
        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(
            backbone, rpn, roi_heads, transform)

    def _set_rpn(self, *args):
        return RegionProposalNetwork(*args)

    def _set_roi_heads(self, *args):
        return OrthogonalRoiHeads(*args)
        
    def ex_both_feat(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x['boxes'] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)

        roi_pooled_features, box_domain = self.roi_heads.disnet(roi_pooled_features)
        
        # domain guided normalization
        domain_mean = box_domain.mean(dim=(2,3), keepdim=True)
        domain_std = box_domain.std(dim=(2,3), keepdim=True, unbiased=False) + 1e-12
        roi_pooled_features = (roi_pooled_features - domain_mean) / domain_std
                
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)

        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
            
        feat_dom = self.roi_heads.domnet(box_domain)
        feat_dom = feat_dom/feat_dom.norm(dim=1, keepdim=True).clamp(min=1e-12)
        
        embeddings, _ = self.roi_heads.embedding_head(rcnn_features)
        embeddings = embeddings.squeeze(3).squeeze(2)
        return embeddings.split(1, 0), feat_dom
       
    def ex_domain_feat(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals = [x['boxes'] for x in targets]
        
        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)

        _, box_domain = self.roi_heads.disnet(roi_pooled_features)
        feat_dom = self.roi_heads.domnet(box_domain)
        feat_dom = feat_dom/feat_dom.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return feat_dom
    
    def ex_feat(self, images, targets, mode='det'):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == 'det':
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == 'reid':
            return self.ex_feat_by_img_crop(images, targets)

    def ex_feat_by_roi_pooling(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x['boxes'] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)

        roi_pooled_features, box_domain = self.roi_heads.disnet(roi_pooled_features)
            
        # domain guided normalization
        domain_mean = box_domain.mean(dim=(2,3), keepdim=True)
        domain_std = box_domain.std(dim=(2,3), keepdim=True, unbiased=False) + 1e-12
        roi_pooled_features = (roi_pooled_features - domain_mean) / domain_std
                
                
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)

        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])

        embeddings, _ = self.roi_heads.embedding_head(rcnn_features)
        embeddings = embeddings.squeeze(3).squeeze(2)
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, 'Only support batch_size 1 in this mode'

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        
        embeddings, _ = self.roi_heads.embedding_head(rcnn_features)
        embeddings = embeddings.squeeze(3).squeeze(2)
        return embeddings.split(1, 0)

class DomainMemory(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, flut, momentum):
        ctx.save_for_backward(inputs, targets, flut, momentum)
        outputs_labeled = inputs.mm(flut.t())

        return torch.cat([outputs_labeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, flut, momentum = ctx.saved_tensors

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([flut], dim=0))
            
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for idx, (x, y) in enumerate(zip(inputs, targets)):
            # y = int(y.item())
            if (0 <= y) and (y < len(flut)):

                flut[y] = momentum * flut[y] + (1-momentum)*x
                flut[y] /= flut[y].norm()

        return grad_inputs, None, None, None

def dm(inputs, targets, flut, momentum = 1/3):
    return DomainMemory.apply(inputs, targets, flut, torch.tensor(momentum))

class OrthogonalRoiHeads(RoIHeads):

    def __init__(self, arguments, embedding_head, embedding_head_dom, reid_regressor, *args, **kwargs):
        super(OrthogonalRoiHeads, self).__init__(*args, **kwargs)
        
        self.args = arguments
        self.embedding_head = embedding_head
        self.embedding_head_dom = embedding_head_dom
        self.reid_regressor = reid_regressor
        
        self.dim = 1024
        self.num_features = 256
        self.key = 22

        # FAT param
        self.FE_b = FidelityEstimator(self.dim, k=1, ce=False)
        self.K = 22
        
        # DIL param
        self.seq = 226
        self.disnet = ChannelDisentangleNet(in_channels = 1024)
        self.register_buffer('dlut', torch.zeros(self.seq, self.num_features))
            
        self.domnet = DomainClassifier(in_channels=1024, out_channels=self.num_features)

    @property
    def feat_head(self):  # re-name
        return self.box_head

    def select_training_samples(self,
                                proposals,  
                                targets):

        self.check_targets(targets)
        assert targets != None
        dtype = proposals[0].dtype

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        if self.training: 
            gt_cnts = [t["cnt"].cuda(non_blocking = True) for t in targets]
            gt_labels = [t["labels"].cuda(non_blocking = True) for t in targets]
            sequences = [target['seq'].cuda(non_blocking = True) for target in targets]
            brisque = [target['brisque'].cuda(non_blocking = True) for target in targets]
            
        else: 
            gt_labels = [t["labels"] for t in targets]
            sequences = [target['seq'].cuda(non_blocking = True) for target in targets]
        
        d_s = {}
        for label, seq in zip(gt_labels, sequences):
            for x, y in zip(label, seq):
                d_s[x.item()] = y
                 
        d_c = {}
        for label, cnt in zip(gt_labels, gt_cnts):
            for x, y in zip(label, cnt):
                d_c[x.item()] = y

        d_brisque = {}
        for label, v in zip(gt_labels, brisque):
            for x, y in zip(label, v):
                d_brisque[x.item()] = y
        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, cnts = self.assign_targets_to_proposals(proposals, gt_boxes, gt_cnts)
        _, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        
        # sample a fixed proportion of positive-negative proposals
        # sampled_inds1 = self.subsample(labels)
        sampled_inds = self.subsample(cnts)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            cnts[img_id] = cnts[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype).cuda()
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        seq = []
        for label in labels:
            tmp = torch.zeros_like(label, dtype = torch.float32).cuda(non_blocking= True)
            for idx, item in enumerate(label):
                if item > 0:
                    tmp[idx] = d_s[item.item()]
            seq.append(tmp.long())

        cnts = []
        for label in labels:
            tmp = torch.zeros_like(label, dtype = torch.float32).cuda(non_blocking= True)
            for idx, item in enumerate(label):
                if item > 0:
                    tmp[idx] = d_c[item.item()]
            cnts.append(tmp)
            
        brisque = []
        for label in labels:
            tmp = torch.zeros_like(label, dtype = torch.float32).cuda(non_blocking= True)
            for idx, item in enumerate(label):
                if item > 0:
                    tmp[idx] = d_brisque[item.item()]
            brisque.append(tmp)

        return proposals, matched_idxs, labels, regression_targets, seq, cnts, brisque

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)
    
    def predict_fidelity(self, input, target, cls_scores, mask):
        pred_brisque, _= self.FE_b(input)
        pred_brisque = pred_brisque.view(pred_brisque.size(0))
        loss_fid = F.mse_loss(pred_brisque[mask], target[mask]) * self.args.scale_fid
    
        fidelity = pred_brisque.clamp(min=1e-12).detach()
    
        loss_con = F.mse_loss(cls_scores[mask], fidelity[mask].clone().detach())
        return loss_fid, loss_con, fidelity
    
    def apply_dil(self, box_domain, embeddings_, sequences):
        feat_dom = self.domnet(box_domain)
        feat_dom = feat_dom / feat_dom.norm(dim=1, keepdim=True).clamp(min=1e-12)
        
        projected = dm(feat_dom, sequences, self.dlut)
        
        projected_labeled = projected[sequences >=0]
        seq_labeled = sequences[sequences >=0]
        val = projected_labeled * self.args.scalar_dom
        
        loss_dom = F.cross_entropy(val, seq_labeled.long().detach()) * self.args.scale_dom
        loss_sep = torch.exp(-self.mmd_gaussian(embeddings_, feat_dom, sigma=1.0)) * self.args.scale_sep
        
        return loss_dom, loss_sep
    
    def gaussian_kernel(self, x, y, sigma=1.0):
        """
        Compute the Gaussian kernel between two sets of samples.

        Parameters:
            x (Tensor): Tensor of shape (N, D) representing the first set of samples.
            y (Tensor): Tensor of shape (M, D) representing the second set of samples.
            sigma (float): The bandwidth of the Gaussian kernel (default=1.0).

        Returns:
            Tensor: Gaussian kernel matrix of shape (N, M).
        """
        assert x.shape[1] == y.shape[1], "Both input tensors should have the same number of features (D)."

        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        dist = x_norm + y_norm.t() - 2.0 * xy
        kernel_matrix = torch.exp(-dist / (2.0 * sigma ** 2))

        return kernel_matrix
    
    def mmd_gaussian(self, x, y, sigma=1.0):
        """
        Compute the Gaussian kernel based Maximum Mean Discrepancy (MMD) distance between two sets of samples.

        Parameters:
            x (Tensor): Tensor of shape (N, D) representing the first set of samples.
            y (Tensor): Tensor of shape (M, D) representing the second set of samples.
            sigma (float): The bandwidth of the Gaussian kernel (default=1.0).

        Returns:
            float: The MMD distance.
        """
        kernel_xx = self.gaussian_kernel(x, x, sigma)
        kernel_yy = self.gaussian_kernel(y, y, sigma)
        kernel_xy = self.gaussian_kernel(x, y, sigma)

        mmd = torch.mean(kernel_xx) - 2.0 * torch.mean(kernel_xy) + torch.mean(kernel_yy)

        return mmd

    def forward(self, features, proposals, image_shapes, targets=None, sampling=False, use_fe=None, epoch=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        
        if self.training:
            proposals, matched_idxs, labels, regression_targets, sequences, cnts, brisque= \
                self.select_training_samples(proposals, targets)
            sequences = torch.cat(sequences) - 1 # bg to -1
            brisque = torch.cat(brisque)
            
            fg_mask = sequences > -1
            
        box_pooled = self.box_roi_pool(features, proposals, image_shapes) # 1024 x 24 x 8
        
        ############################################
        ## ------           DIL            ------ ##
        ############################################
        box_pooled, box_domain = self.disnet(box_pooled)
        
        domain_mean = box_domain.mean(dim=(2,3), keepdim=True)
        domain_std = box_domain.std(dim=(2,3), keepdim=True, unbiased=False) + 1e-12
        box_pooled = (box_pooled - domain_mean) / domain_std
        ############################################

        rcnn_features = self.feat_head(box_pooled)
        
        if self.training:
            result, losses = [], {}
            det_labels = [(y != 0).long() for y in labels]
            box_regression = self.box_predictor(rcnn_features['feat_res5'])
            embeddings_, class_logits = self.embedding_head(rcnn_features, det_labels)

            if self.args.use_sigmoid:
                cls_scores = torch.sigmoid(class_logits[:, 1].squeeze())
            else:
                cls_scores = F.softmax(class_logits, dim=1)[:,[1]].squeeze()
            
            #! apply fat
            loss_fid, loss_con, fidelity = \
                self.predict_fidelity(
                    input=rcnn_features['feat_res4'], target=brisque, cls_scores=cls_scores, mask=fg_mask)
                
            embeddings_ = embeddings_.squeeze(3).squeeze(2)

            #! apply dil
            loss_dom, loss_sep = \
                self.apply_dil(box_domain, embeddings_, sequences)
                
            loss_detection, loss_box_reg = \
                rcnn_loss(class_logits.squeeze(3).squeeze(2), box_regression, det_labels, \
                    regression_targets, fidelity)
            
            loss_reid = self.reid_regressor(embeddings_, labels, cls_scores, fidelity)
            ###############################################

            losses = dict(loss_detection=loss_detection,
                        loss_box_reg=loss_box_reg,
                        loss_reid=loss_reid,
                        loss_fid=loss_fid,
                        loss_con=loss_con,
                        loss_dom=loss_dom,
                        loss_sep=loss_sep
                        )
                        
        else:
            box_regression = self.box_predictor(rcnn_features['feat_res5'])
            embeddings_, class_logits = self.embedding_head(rcnn_features, None)
            embeddings_ = embeddings_.squeeze(3).squeeze(2)

            result, losses = [], {}
            boxes, scores, embeddings, labels = \
                self.postprocess_detections(class_logits, box_regression, embeddings_,
                                            proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted
        return result, losses

    def postprocess_detections(self, class_logits, box_regression, embeddings_, proposals, image_shapes):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        class_logits = F.softmax(class_logits, dim=1)
        pred_scores = class_logits[:,1]
        # print('pred_scores.shape', pred_scores.shape)
        embeddings_ = embeddings_ * pred_scores.view(-1, 1)  # CWS

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0)).cuda(device, non_blocking = True)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = boxes[
                inds], scores[inds], labels[inds], embeddings[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

class DomainClassifier(nn.Module):
    def __init__(self, in_channels=1024, out_channels=256):
        super(DomainClassifier, self).__init__()
        self.net = \
            nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels,in_channels*2,bias=False),
                nn.BatchNorm1d(in_channels*2),
                nn.ReLU(),
                nn.Linear(in_channels*2, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, out_channels)
            )
        
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.net(x)
        return x
    
class ChannelDisentangleNet(nn.Module):
    def __init__(self, in_channels=1024, reduction=16):
        super(ChannelDisentangleNet, self).__init__()
        self.mask_generator_id = \
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            
        self.mask_generator_dom = \
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
    def forward(self, x):
        id_mask = self.mask_generator_id(x)
        dom_mask = self.mask_generator_dom(x)
        feat_id = x * id_mask
        feat_dom = x * dom_mask
        return feat_id, feat_dom
    
class OrthogonalEmbeddingProj(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256,
                 cls_scalar=1.0,
                 mode=None):
        super(OrthogonalEmbeddingProj, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = list(map(int, in_channels))
        self.dim = int(dim)
        self.cls_scalar = cls_scalar
        self.mode = mode
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            proj = nn.Sequential(
                nn.Conv2d(in_chennel, indv_dim, 1, 1, 0),
                nn.BatchNorm2d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.projectors_reid = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            proj = nn.Sequential(
                nn.Conv2d(in_chennel, indv_dim, 1, 1, 0),
                nn.BatchNorm2d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors_reid[ftname] = proj
        if self.mode==None:
            self.pred_class = nn.Conv2d(self.dim, 2, 1,1,0, bias=False)
            init.normal_(self.pred_class.weight, std=0.01)

    def forward(self, featmaps, targets=None):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        
        if targets != None:
            # Train mode
            targets = torch.cat(targets,dim=0)
        
        outputs = []
        for k, v in featmaps.items():
            v = F.adaptive_max_pool2d(v, 1)
            outputs.append(
                self.projectors[k](v)
            )
        embeddings = torch.cat(outputs, dim=1)  
        if self.mode == None:
            projected = self.pred_class(embeddings) * self.cls_scalar
        else:
            projected = None
        outputs_reid = []
        for k, v in featmaps.items():
            v = F.adaptive_max_pool2d(v, 1)
            outputs_reid.append(
                self.projectors_reid[k](v)
            )
        
        embeddings_reid = torch.cat(outputs_reid, dim=1)
        embeddings_reid = embeddings_reid / embeddings_reid.norm(dim=1, keepdim=True).clamp(min=1e-12)
        
        return embeddings_reid, projected

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            x = F.adaptive_max_pool2d(x, 1)
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp

class BboxRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection
    """
    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(BboxRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes),
                nn.BatchNorm1d(4 * num_classes))
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)
        
    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas

def rcnn_loss(class_logits, box_regression, labels, regression_targets, fidelity):

    """
    Computes the loss for Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, )
        box_regression (Tensor)
        
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
        
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = wBCE(
            class_logits, labels.long(), fidelity)
    
    
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)
    
    box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction="sum",
        )

    box_loss = box_loss / labels.numel()
        
    return classification_loss, box_loss

class FidelityEstimator(nn.Module):
    def __init__(self, in_channels = 1024, k = 1, ce=False):
        super(FidelityEstimator, self).__init__()
        self.conv_1= nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, k)
        self.ce = ce
        if self.ce == False:
            self.sigmoid = nn.Tanh()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        feat = self.conv_4(x)
        x = feat.squeeze()
        x = self.fc(x)
        if self.ce == False:
            x = self.sigmoid(x) +1
        return x, feat

def wBCE(input, target, fidelity):
    fg_mask = target>0
    bg_mask = ~fg_mask

    val = input[F.one_hot(target) > 0] + 1e-8
    val = torch.sigmoid(val)

    loss_fg = torch.mean(-torch.log(val[fg_mask]) * fidelity[fg_mask].squeeze(), dim = 0)
    loss_bg = torch.mean(-torch.log(val[bg_mask]), dim = 0)
    loss = (loss_fg + loss_bg) / 2.0

    return loss

def get_model(args, training=True, pretrained_backbone=True):
    phase_args = args.train if training else args.test
    # Resnet50
    resnet_part1, resnet_part2 = resnet_backbone('resnet50', 
                                        pretrained_backbone, 
                                        GAP=True
                                        )

    ##### Region Proposal Network ######
    # Anchor generator (Default)
    rpn_anchor_generator = AnchorGenerator(
                            (tuple(args.anchor_scales),), 
                            (tuple(args.anchor_ratios),))
    # 2D embedding head
    backbone = resnet_part1
    # 1D embedding head
    rpn_head = RPNHead(
                resnet_part1.out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])

    ########## Bbox Network #########
    # Bbox regressor
    box_predictor = BboxRegressor(
                    2048, num_classes=2,
                    RCNN_bbox_bn=args.rcnn_bbox_bn)
    
    # Bbox pooler
    box_roi_pool = MultiScaleRoIAlign(
                    featmap_names=['feat_res4'],
                    output_size=[24,8],
                    sampling_ratio=2)
                    
    # 2D embedding head
    feat_head = resnet_part2
    # 1D embedding head
    embedding_head = OrthogonalEmbeddingProj(
                    featmap_names=['feat_res4', 'feat_res5'],
                    in_channels=[1024, 2048],
                    dim=args.num_features,
                    cls_scalar=args.cls_scalar)
    if args.DSDI:
        embedding_head_dom = OrthogonalEmbeddingProj(
                    featmap_names=['feat_res4', 'feat_res5'],
                    in_channels=[1024, 2048],
                    dim=args.num_features,
                    cls_scalar=args.cls_scalar,
                    mode='dom')
    else:
        embedding_head_dom = None      

    # ReID regressor
    reid_regressor = OIMLoss(
                        args.num_features, args.num_pids, 
                        args.train.oim_momentum, 
                        args.oim_scalar, 
                        )

    model = FasterRCNN( args,
                        # Region proposal network
                        backbone=backbone,
                        min_size=phase_args.min_size, max_size=phase_args.max_size,
                        # Anchor generator
                        rpn_anchor_generator=rpn_anchor_generator,
                        # RPN parameters
                        rpn_head=rpn_head,
                        rpn_pre_nms_top_n_train=args.train.rpn_pre_nms_top_n,
                        rpn_post_nms_top_n_train=args.train.rpn_post_nms_top_n,
                        rpn_pre_nms_top_n_test=args.test.rpn_pre_nms_top_n,
                        rpn_post_nms_top_n_test=args.test.rpn_post_nms_top_n,
                        rpn_nms_thresh=phase_args.rpn_nms_thresh,
                        rpn_fg_iou_thresh=args.train.rpn_positive_overlap,
                        rpn_bg_iou_thresh=args.train.rpn_negative_overlap,
                        rpn_batch_size_per_image=args.train.rpn_batch_size,
                        rpn_positive_fraction=args.train.rpn_fg_fraction,
                        # Bbox network
                        box_predictor=box_predictor,
                        box_roi_pool=box_roi_pool,
                        feat_head=feat_head,
                        embedding_head=embedding_head,
                        embedding_head_dom=embedding_head_dom,
                        box_score_thresh=args.train.fg_thresh,
                        box_nms_thresh=args.test.nms,  # inference only
                        box_detections_per_img=phase_args.rpn_post_nms_top_n,  # use all
                        box_fg_iou_thresh=args.train.bg_thresh_hi,
                        box_bg_iou_thresh=args.train.bg_thresh_lo,
                        box_batch_size_per_image=args.train.rcnn_batch_size,
                        box_positive_fraction=args.train.fg_fraction,  # for proposals
                        bbox_reg_weights=args.train.box_regression_weights,
                        reid_regressor=reid_regressor,
                        )
    if training:
        model.train()
    else:
        model.eval()
    
    return model
