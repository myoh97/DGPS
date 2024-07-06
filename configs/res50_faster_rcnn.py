import argparse
import torchvision as tv


def args_faster_rcnn():
    parser = argparse.ArgumentParser(
        add_help=False,
        description='Plain Faster R-CNN')

    parser.add_argument('-p', '--path', dest='path',
                        help='directory to save models', default='logs/')
    
    parser.add_argument('--memo', default= 'hello')

    #! preprocessing w.r.t keypoint
    parser.add_argument('--occ_k', default=12, type=int, help='preprocess w.r.t # of occluded keypoints')

    #! rezie
    parser.add_argument('--disable_resize', action='store_false', default=True)

    ###! DIL
    parser.add_argument('--DIL', action='store_true', default=False)
    #! Domain Guided Normalization
    parser.add_argument('--DGN', action='store_true', default=False) 
    #! Separation Loss
    parser.add_argument('--SEP', action='store_true', default=False) 
    #! DGN => IN
    parser.add_argument('--IN', action='store_true', default=False) 
    parser.add_argument('--IGN', action='store_true', default=False) 
    parser.add_argument('--scale_sep', default=1.0, type=float, help='scale of fidelity loss')
    parser.add_argument('--scale_dom', default=1.0, type=float, help='scale of fidelity loss')
    parser.add_argument('--scalar_dom', default=1.0, type=float, help='scale of fidelity loss')
    
    
    ###! FAT
    parser.add_argument('--FAT', action='store_true', default=False, help='apply FAT')
    parser.add_argument('--FWDL_CLS', action='store_true', default=False, help='Fidelity Weighted Detection Loss')
    parser.add_argument('--FWDL_REG', action='store_true', default=False, help='Fidelity Weighted Detection Loss')
    parser.add_argument('--FGCL', action='store_true', default=False, help='Fidelity Guided Confidence Loss')
    parser.add_argument('--FWFU', action='store_true', default=False, help='Fidelity Weighted Feature Update')
    
    parser.add_argument('--use_sigmoid', action='store_true', default=False, help='')
    #! classification loss type
    parser.add_argument('--cls_loss_type', choices=['sigmoid', 'softmax', 'softlabel'], default='sigmoid', type=str)
    
    #! Brisque
    parser.add_argument('--direct', action='store_true', default=False, help='')
    
    parser.add_argument('--brisque', action='store_true', default=False)
    parser.add_argument('--brisque_k', type=float, default=1.0)
    parser.add_argument('--brisque_gt', action='store_true', default=False)
    parser.add_argument('--scale_fid', default=10.0, type=float, help='scale of fidelity loss')
    
    
    parser.add_argument('--scale_fid_penalty', default=10.0, type=float, help='scale of fidelity loss')
    parser.add_argument('--penalty', action='store_true', default=False, help='Fidelity Weighted Feature Update')
    parser.add_argument('--k_sharpness', default=1.0, type=float, help='k_sharpness')
    parser.add_argument('--k_contrast', default=1.0, type=float, help='k_contrast')

    
    parser.add_argument('--fidelity_averaging', action='store_true', default=False, help='')
    #! contrast
    parser.add_argument('--contrast', action='store_true', default=False)
    parser.add_argument('--contrast_gt', action='store_true', default=False)
    parser.add_argument('--contrast_use_prob', action='store_true', default=False)
    parser.add_argument('--cont_gt_k', default=0., type=float)
    parser.add_argument('--scale_fid_contrast', default=1.0, type=float, help='scale of fidelity loss')
    #! sharpness
    parser.add_argument('--sharpness', action='store_true', default=False)
    parser.add_argument('--sharpness_gt', action='store_true', default=False)
    parser.add_argument('--sharpness_use_prob', action='store_true', default=False)
    parser.add_argument('--sharp_gt_k', default=0., type=float)
    parser.add_argument('--scale_fid_sharpness', default=1.0, type=float, help='scale of fidelity loss')
    #! occlusion
    parser.add_argument('--occlusion', action='store_true', default=False)
    parser.add_argument('--naive_occ', action='store_true', default=False)
    parser.add_argument('--occlusion_gt', action='store_true', default=False)
    parser.add_argument('--occlusion_use_prob', action='store_true', default=False)
    parser.add_argument('--threshold_occ', default=22, type=int)
    parser.add_argument('--tau', default=1.0, type=float, help='tau')
    parser.add_argument('--scale_fid_occlusion', default=1.0, type=float, help='scale of fidelity loss')
    
    
    parser.add_argument('--tsne', action='store_true', default=False)
    
    parser.add_argument('--exp2', action='store_true', default=False)
    parser.add_argument('--exp3', action='store_true', default=False)
    parser.add_argument('--exp4', action='store_true', default=False) 
    parser.add_argument('--exp5', action='store_true', default=False)
    parser.add_argument('--exp6', action='store_true', default=False)
    parser.add_argument('--exp7', action='store_true', default=False)
    parser.add_argument('--FAT_exp', action='store_true', default=False, help='apply FAT')
    parser.add_argument('--FAT_seperate_mean', action='store_true', default=False) 
    parser.add_argument('--FAT_seperate', action='store_true', default=False) 
    parser.add_argument('--FAT_underfit', action='store_true', default=False) 
    parser.add_argument('--FAT_underfit_sampling', action='store_true', default=False) 
    parser.add_argument('--pseudo', action='store_true', default=False) 
    parser.add_argument('--FAT_seperate_gt', action='store_true', default=False) 
    parser.add_argument('--FAT_occ', action='store_true', default=False) 
    parser.add_argument('--FAT_area', action='store_true', default=False) 
    parser.add_argument('--FAT_occ_area', action='store_true', default=False) 
    parser.add_argument('--FAT_otherfactor', action='store_true', default=False) 
    parser.add_argument('--FAT_factor', type=str, default=None,
                        choices=['sharp', 'sharp_g', 'blur', 'blur_g', 'contrast', 'area'])   
    parser.add_argument('--FAT_mapping_func', type=str, default=None,
                        choices=['cosine', 'sigmoid5', 'sigmoid10', 'exponential']) 
    parser.add_argument('--FAT_usegt', action='store_true', default=False) 
    parser.add_argument('--FAT_occ_ce', action='store_true', default=False) 
    parser.add_argument('--FAT_otherfactor_mul', action='store_true', default=False)

    parser.add_argument('--DSDI', action='store_true', default=False)

    parser.add_argument('--oim_fidelity', action='store_true', default=False)

    
    parser.add_argument('--fid_loss', default='l1', type=str, help='fidelity loss : l1 or l2')
       # Data
    parser.add_argument('--dataset',
                        help='training dataset', type=str, default='JTA',
                        choices=['JTA'])
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--test_data',
                        help='test dataset', type=str, default='CUHK_PRW',
                        choices=['CUHK', 'PRW', 'JTA','JTA_PRW', 'JTA_CUHK', 'CUHK_PRW','ALL'])
    # Net architecture
    parser.add_argument('--net',
                        default='resnet50', type=str,
                        choices=tv.models.resnet.__all__)

    parser.add_argument('--rm_rcnn_bbox_bn', dest='rcnn_bbox_bn',
                        help='whether to use batch normalization for dc_rcc_box_regression',
                        action='store_false')
    parser.add_argument('--anchor_scales', type=float, nargs='+', default=(32, 64, 128, 256, 512),
                        help='ANCHOR_SCALES w.r.t. image size.')
    parser.add_argument('--anchor_ratios', type=float, nargs='+', default=(0.5, 1., 2.),
                        help='ANCHOR_RATIOS: anchor height/width')

    # resume trained model
    parser.add_argument('--resume',
                        help='resume file path',
                        default=None, type=str)

    # Device
    parser.add_argument('--device', default='cuda', help='device')
    
    # Random Seed
    parser.add_argument('--seed', type=int, default=1)

    #
    # Training
    #
    parser.add_argument('--wo_pretrained', dest='train.wo_pretrained',
                        help='whether to disable ImageNet pretrained weights.',
                        action='store_true')
    parser.add_argument('--start_epoch', dest='train.start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='train.epochs',
                        help='number of epochs to train',
                        default=16, type=int)
    parser.add_argument('--disp_interval', dest='train.disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)

    # Training.Optimization
    parser.add_argument('--lr', dest='train.lr',
                        help='starting learning rate', default=0.003, type=float)
    parser.add_argument('--momentum', dest='train.momentum',
                        help='Momentum',
                        default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='train.weight_decay',
                        help='Weight Decay',
                        default=0.0005, type=float)
    parser.add_argument('--lr_decay_gamma', dest='train.lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # Training.Optimization.lr_decay
    parser.add_argument('--lr_cosine', dest='train.lr_cosine', action='store_true')
    parser.add_argument('--lr_decay_step', dest='train.lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=3, type=int)
    parser.add_argument('--lr_decay_milestones', type=int, dest='train.lr_decay_milestones',
                        nargs='+', default=None)
    parser.add_argument('--lr_warm_up', dest='train.lr_warm_up',
                        action='store_true', default=True)
    parser.add_argument('--clip_gradient', dest='train.clip_gradient',
                        type=float, default=10.0)

    # Training.data
    parser.add_argument('--aspect_grouping', dest='train.aspect_grouping',
                        type=int, default=-1,
                        help='Whether to use aspect-ratio grouping of training images, \
                              introduced merely for saving GPU memory')
    parser.add_argument('--min_size', dest='train.min_size',
                        type=int, default=900,
                        help='Minimum size of the image to be rescaled before feeding \
                              it to the backbone')
    parser.add_argument('--max_size', dest='train.max_size',
                        type=int, default=1500,
                        help='Max pixel size of the longest side of a scaled input image')
    parser.add_argument('--batch_size', dest='train.batch_size',
                        type=int, default=4,
                        help='batch_size, __C.TRAIN.IMS_PER_BATCH')
    parser.add_argument('--no_flip', dest='train.use_flipped',
                        action='store_false',
                        help='Use horizontally-flipped images during training?')

    # Training.data.rcnn/rpn.sampling
    parser.add_argument('--rcnn_batch_size', dest='train.rcnn_batch_size',
                        type=int, default=128,
                        help='Minibatch size (number of regions of interest [ROIs])\
                              __C.TRAIN.BATCH_SIZE')
    parser.add_argument('--fg_fraction', dest='train.fg_fraction',
                        type=float, default=0.7,
                        help='Fraction of minibatch that is labeled foreground (i.e. class > 0)')
    parser.add_argument('--fg_thresh', dest='train.fg_thresh',
                        type=float, default=0.5,
                        help='Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)')
    parser.add_argument('--bg_thresh_hi', dest='train.bg_thresh_hi',
                        type=float, default=0.5,
                        help='Overlap threshold for a ROI to be considered background')
    parser.add_argument('--bg_thresh_lo', dest='train.bg_thresh_lo',
                        type=float, default=0.1,
                        help='Overlap threshold for a ROI to be considered background')
    parser.add_argument('--box_regression_weights', dest='train.box_regression_weights',
                        type=float, nargs=4, default=[10., 10., 5., 5.],
                        help='Weights for the encoding/decoding of the bounding boxes')

    # Training.RPN
    parser.add_argument('--rpn_positive_overlap', dest='train.rpn_positive_overlap',
                        type=float, default=0.7,
                        help='IOU >= thresh: positive example')
    parser.add_argument('--rpn_negative_overlap', dest='train.rpn_negative_overlap',
                        type=float, default=0.3,
                        help='IOU < thresh: negative example')
    parser.add_argument('--rpn_fg_fraction', dest='train.rpn_fg_fraction',
                        type=float, default=0.5,
                        help='Max ratio of foreground examples.')
    parser.add_argument('--rpn_batch_size', dest='train.rpn_batch_size',
                        type=int, default=256,
                        help='Total number of examples')
    parser.add_argument('--rpn_nms_thresh', dest='train.rpn_nms_thresh',
                        type=float, default=0.7,
                        help='NMS threshold used on RPN proposals')
    parser.add_argument('--rpn_pre_nms_top_n', dest='train.rpn_pre_nms_top_n',
                        type=int, default=12000,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_post_nms_top_n', dest='train.rpn_post_nms_top_n',
                        type=int, default=2000,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_min_size', dest='train.rpn_min_size',
                        type=int, default=8,
                        help='Proposal height and width both need to be greater than RPN_MIN_SIZE (at\
                              orig image scale)')

    # Training.checkpointing
    parser.add_argument('--checkpoint_interval', dest='train.checkpoint_interval',
                        type=int, default=1,
                        help='Epochs between snapshots.')
    parser.add_argument('--resume_name', dest='train.resume_name',
                        type=str, default=None,
                        help='Name of checkpoint file')
                        
    # Training.loss weights
    parser.add_argument('--w_RPN_loss_cls', dest='train.w_RPN_loss_cls',
                        default=1.0, type=float)
    parser.add_argument('--w_RPN_loss_box', dest='train.w_RPN_loss_box',
                        default=1.0, type=float)
    parser.add_argument('--w_RCNN_loss_bbox', dest='train.w_RCNN_loss_bbox',
                        default=10.0, type=float)
    parser.add_argument('--w_RCNN_loss_cls', dest='train.w_RCNN_loss_cls',
                        default=1.0, type=float)
    parser.add_argument('--w_loss_fid', dest='train.w_loss_fid',
                        default=1.0, type=float)
    parser.add_argument('--w_loss_con', dest='train.w_loss_con',
                        default=1.0, type=float)
    parser.add_argument('--w_loss_dom', dest='train.w_loss_dom',
                        default=1.0, type=float)
    parser.add_argument('--w_loss_dst', dest='train.w_loss_dst',
                        default=10.0, type=float)
    #
    # Test
    #
    parser.add_argument('--checkpoint_name', dest='test.checkpoint_name',
                        type=str, default='checkpoint.pth',
                        help='Name of checkpoint file')
    parser.add_argument('--min_size_test', dest='test.min_size',
                        type=int, default=900,
                        help='Minimum size of the image to be rescaled before feeding \
                              it to the backbone')
    parser.add_argument('--max_size_test', dest='test.max_size',
                        type=int, default=1500,
                        help='Max pixel size of the longest side of a scaled input image')
    parser.add_argument('--batch_size_test', dest='test.batch_size',
                        default=1, type=int,
                        help='batch_size')
    parser.add_argument('--nms_test', dest='test.nms',
                        type=float, default=0.4,
                        help='NMS threshold used on RCNN output')
    parser.add_argument('--rpn_nms_thresh_test', dest='test.rpn_nms_thresh',
                        type=float, default=0.7,
                        help='NMS threshold used on RPN proposals')
    parser.add_argument('--rpn_pre_nms_top_n_test', dest='test.rpn_pre_nms_top_n',
                        type=int, default=6000,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_post_nms_top_n_test', dest='test.rpn_post_nms_top_n',
                        type=int, default=300,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_min_size_test', dest='test.rpn_min_size',
                        type=int, default=16,
                        help='Proposal height and width both need to be greater than RPN_MIN_SIZE (at\
                              orig image scale)')

    # sizes
    parser.add_argument('--num_features', type=int, default=256,
                        help='Embedding dimension.')
    parser.add_argument('--num_pids', type=int, default=10912,
                        choices=[11296, 10912],
                        help='Labeled person ids in each dataset.')
    
    parser.add_argument('--oim_scalar', type=float, default=30.0,
                        help='OIM scalar')
    parser.add_argument('--cls_scalar', type=float, default=1.0,
                        help='Person classification scalar')
    parser.add_argument('--cls_weight', type=float, default=1.0,
                        help='Person classification weight')

    # training
    parser.add_argument('--w_OIM_loss_oim', dest='train.w_OIM_loss_oim',
                        default=1.0, type=float)
    parser.add_argument('--oim_momentum', dest='train.oim_momentum',
                        default=0.67, type=float)
    parser.add_argument('--reid_loss', dest='reid_loss', default='oim',
                        type=str)
    parser.add_argument('--debug',
                        default=0, type=int,
                        help='Mode for debug')
                        
    return parser
