import time
import os.path as osp
import huepy as hue
import torch
import json
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

from lib.utils.evaluator import inference, detection_performance_calc
from .logger import MetricLogger
from .misc import ship_data_to_cuda, warmup_lr_scheduler, resume_from_checkpoint, get_optimizer,  get_lr_scheduler

class Trainer():

    def __init__(self, args, model, train_loader, gallery_loader1, probe_loader1, gallery_loader2,  probe_loader2, gallery_loader3,  probe_loader3, optimizer, lr_scheduler, device, working_path):

        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.gallery_loader1 = gallery_loader1
        self.gallery_loader2 = gallery_loader2
        self.gallery_loader3 = gallery_loader3
        self.probe_loader1 = probe_loader1
        self.probe_loader2 = probe_loader2
        self.probe_loader3 = probe_loader3
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.working_path = working_path

    def run(self):
        steps = 0
        for epoch in range(self.args.train.start_epoch, self.args.train.epochs):
            if torch.cuda.device_count()>1:
                self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            
            if epoch == 0 and self.args.train.lr_warm_up:
                warmup_factor = 1. / 1000
                warmup_iters = len(self.train_loader) - 1
                sub_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
            metric_logger = MetricLogger()
            print(hue.info(hue.bold(hue.green("Start training from %s epoch"%str(epoch)))))

            start_time = time.time()
            
            for iteration, data in enumerate(self.train_loader):
                ## Initial iterations
                steps = epoch*len(self.train_loader) + iteration
                if steps % self.args.train.disp_interval == 0:
                    start = time.time()

                # Load data
                images, targets = ship_data_to_cuda(data, self.device)
                # Pass data to model
                loss_dict = self.model(images, targets, sampling=False, use_fe=None, epoch=epoch)

                # Total loss
                losses = self.args.train.w_RPN_loss_cls * loss_dict['loss_objectness'] \
                        + self.args.train.w_RPN_loss_box * loss_dict['loss_rpn_box_reg'] \
                        + self.args.train.w_RCNN_loss_bbox * loss_dict['loss_box_reg'] \
                        + self.args.train.w_RCNN_loss_cls * loss_dict['loss_detection'] \
                        + self.args.train.w_OIM_loss_oim * loss_dict['loss_reid'] \
                        + self.args.train.w_loss_fid * loss_dict['loss_fid'] \
                        + self.args.train.w_loss_con * loss_dict['loss_con'] \
                        + self.args.train.w_loss_dom * loss_dict['loss_dom'] \
                        + loss_dict['loss_sep']

                self.optimizer.zero_grad()
                losses.backward()
                if self.args.train.clip_gradient > 0:
                    clip_grad_norm_(self.model.parameters(), self.args.train.clip_gradient)
                self.optimizer.step()

                ## Post iteraions
                if epoch == 0 and self.args.train.lr_warm_up:
                    sub_scheduler.step()

                if steps % self.args.train.disp_interval == 0:
                    # Print 
                    loss_value = losses.item()
                    state = dict(loss_value=loss_value,
                                lr=self.optimizer.param_groups[0]['lr'])
                    state.update(loss_dict)

                    # Update logger
                    batch_time = time.time() - start
                    metric_logger.update(batch_time=batch_time)
                    metric_logger.update(**state)
                        
                    # Print log on console
                    metric_logger.print_log(epoch, iteration, len(self.train_loader))
                else:
                    state = None

            ## Post epochs
            self.lr_scheduler.step()

            if (self.args.gpu == 0 and torch.cuda.device_count() > 1) or (torch.cuda.device_count() == 1): # only master
                save_name = osp.join(self.args.path, 'checkpoint_epoch%d.pth'%epoch)

                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }, save_name)

                print(hue.good('save model: {}'.format(save_name)))
                if epoch == self.args.train.epochs - 1: continue
                if epoch >= 2:
                    performance = self.eval_model(epoch, self.args.test_data)
                    with open(save_name.replace('.pth', '-regular.json'), 'w') as f:
                            json.dump(performance, f)

            print(hue.info(hue.bold(hue.lightgreen(
            'Working directory: {}'.format(self.working_path)))))
            total_time = time.time() - start_time
            epoch_time(total_time)

        return None

    def eval_model(self, epoch, test_data):
        gallery_loader_all = self.gallery_loader1, self.gallery_loader2, self.gallery_loader3
        probe_loader_all = self.probe_loader1, self.probe_loader2, self.probe_loader3
        if test_data in ['CUHK', 'PRW', 'JTA'] :
            data1 = test_data
            data2 = "None"
            data3 = "None"
        if test_data in ['JTA_PRW', 'JTA_CUHK', 'CUHK_PRW'] :
            data1, data2 = test_data.split("_")
            data3 = "None"
        if test_data == 'ALL':
            data1 = "JTA"
            data2 = "CUHK"
            data3 = "PRW"
        data_name = [data1, data2, data3]
        performance_all = {}
        for name, gallery_loader, probe_loader in zip(data_name, gallery_loader_all, probe_loader_all):
            if name !='JTA':
                if gallery_loader != "None":
                    print(hue.info(hue.bold(hue.lightgreen(
                        'Evaluating Model in epoch {}!!!'.format(epoch)))))
                    name_to_boxes, all_feats, probe_feats = \
                            inference(self.model, gallery_loader, probe_loader, self.device)
                    
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
                    performance_all[name] = performance
        return performance_all

def epoch_time(total_time):
    total_time_str = str(timedelta(seconds=int(total_time)))
    print("Total training time {}".format(total_time_str))
