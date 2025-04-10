from copy import deepcopy
import os
import shutil
import time
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .dataset import make_dataset, make_dataloader
from .model import ResNet, SSM, SSM_contention, SSM_contention_two_branches
from .metric import spearman_rank_correlation, top_k_recall
from .optim import make_optimizer, make_scheduler
from .train_utils import (
    Logger, AverageMeter, fix_random_seed, time_str, worker_init_reset_seed, convert_all_predictions
)
from libs.ql_utils import NEW_BRANCHES, FILTER_HIGH_LATENCY_BRANCHES, PARETO_FRONTIER_BRANCHES, ENLARGE79, ENLARGE82, CONTENTION55, CONTENTION12

old_branches = ['centerpoint_dyn_pillar032_4x',
            'centerpoint_dyn_pillar036_4x',
            'centerpoint_dyn_pillar040_4x',
            'centerpoint_dyn_pillar044_4x',
            'centerpoint_dyn_pillar048_4x',
            'centerpoint_dyn_pillar052_4x',
            'centerpoint_dyn_pillar056_4x',
            'centerpoint_dyn_pillar060_4x',
            'centerpoint_without_resnet_dyn_voxel100',
            'centerpoint_without_resnet_dyn_voxel150',
            'centerpoint_without_resnet_dyn_voxel200',
            'centerpoint_without_resnet_dyn_voxel250',
            'centerpoint_without_resnet_dyn_voxel300',
            'centerpoint_without_resnet_dyn_voxel350',
            'centerpoint_without_resnet_dyn_voxel400',
            'centerpoint_without_resnet_dyn_voxel450',
            'dsvt_sampled_pillar020',
            'dsvt_sampled_pillar030',
            'dsvt_sampled_pillar040',
            'dsvt_sampled_pillar050',
            'dsvt_sampled_pillar060',
            'dsvt_sampled_pillar070',
            'dsvt_sampled_pillar080',
            'dsvt_sampled_pillar090',
            'dsvt_sampled_pillar100',
            'dsvt_sampled_pillar110',
            'dsvt_sampled_pillar120',
            'dsvt_sampled_pillar130',
            'dsvt_sampled_voxel020',
            'dsvt_sampled_voxel030',
            'dsvt_sampled_voxel040',
            'dsvt_sampled_voxel050',
            'dsvt_sampled_voxel060',
            'dsvt_sampled_voxel070',
            'dsvt_sampled_voxel080',
            'dsvt_sampled_voxel090',
            'dsvt_sampled_voxel100',
            'dsvt_sampled_voxel110',
            'dsvt_sampled_voxel120',
            'dsvt_sampled_voxel130']

filter_hl_branches_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

pareto_frontier_branches_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 30, 31, 32]


class Trainer:

    def __init__(self, opt):

        self.opt = opt
        # set random seed
        rng = fix_random_seed(opt.get('seed', 2023))

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        dataset = make_dataset(opt['train']['data'])
        self.dataloader = make_dataloader(
            dataset, rng, opt['train']['batch_size'], opt['train']['num_workers']
        )

        # build model and EMA
        if 'model_name' not in opt:
            model_class = ResNet
        elif opt['model_name']== 'SSM':
            model_class = SSM
            #opt['model'].pop('name')
            
        # opt['model']['num_classes'] = dataset.num_classes
        self.model = model_class(**opt['model']).cuda()
        #self.model = nn.DataParallel(temp_model, device_ids=['cuda:0',])
        
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
        self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
        self.loss_meter = AverageMeter()
        self.timer = AverageMeter()

        # load model weights and training states
        if opt['_resume']:
            self.load()
        
        self.loss_fn = F.mse_loss
        #self.loss_fn = F.mse_loss if opt['train']['loss'] == 'mse' else spearmanr_loss

    def run(self):
        print("Training started.")
        real_start_time = time.time()
        while self.epoch < self.num_epochs:
            for data_tuple in self.dataloader:
                data, acc = data_tuple['feature'],  data_tuple['reward']
                          
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.forward_backward(data, acc)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                self.loss_meter.update(loss)
                self.timer.update(time.time() - start_time)
                if self.itr == 1 or self.itr % self.log_interval == 0:
                    self.log()
                    curr_time = time.time()
                    print('time pass:', curr_time-real_start_time)
                    real_start_time = curr_time
            self.epoch += 1
            self.checkpoint()
        print("Training completed.")

    def forward_backward(self, data, acc):
        data = data.cuda(non_blocking=True)
        acc = acc.cuda(non_blocking=True)
        # if extra_info is not None:
        #     extra_info = extra_info.cuda(non_blocking=True)

        pred = self.model(data)
        loss = self.loss_fn(pred, acc)
        loss.backward()
        return loss.detach()

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))
        for (bn, b), (_, b_ema) in zip(
            self.model.named_buffers(), self.model_ema.named_buffers()
        ):
            if 'running' in bn: # copy over running statistics of batchnorm
                b_ema.copy_(b.detach())

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self.model.state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        log_str += f"loss {self.loss_meter.item():.3f} | "
        self.tb_writer.add_scalar("loss", self.loss_meter.item(), self.itr)
        self.loss_meter.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += str(self.timer.item())
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()


class Evaluator:

    def __init__(self, opt, usepolicy):

        self.opt = opt
        self.usepolicy = usepolicy

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2023))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader = make_dataloader(
            dataset, rng, batch_size=1, num_workers=0, is_training=False
        )

        # load model
        # opt['model']['num_classes'] = dataset.num_classes
        if 'model_name' not in opt or opt['model_name'] == 'ResNet':
            model_class = ResNet
        elif opt['model_name'] == 'SSM':
            model_class = SSM
        elif opt['model_name'] == 'SSM_contention':
            model_class = SSM_contention
        elif opt['model_name'] == 'SSM_contention_two_branches':
            model_class = SSM_contention_two_branches
        else:
            raise NotImplementedError
        
        self.model = model_class(**opt['model']).cuda()
        # self.critic = model_class(**opt['model'], critic=True).cuda() 
        self.load_model()
        self.model.eval().requires_grad_(False)
        
        # whether eval use the new branches
        self.use_new_branches = opt['eval']['use_new_branches'] if 'use_new_branches' in opt['eval'] else False
        self.branch_filter_type = opt['eval']['branch_filter_type'] if 'branch_filter_type' in opt['eval'] else 'default'
        self.punishment_buffer_type = opt['train']['punishment_buffer_type'] if 'punishment_buffer_type' in opt['train'] else 'ratio'
        
        # the root for the profiling result 
        self.profiling_root = opt['eval']['profiling_root'] if 'profiling_root' in opt['eval'] else '/anvil/projects/x-cis230283/datasets/waymo_new_profiling/det/test/'

        # metrics
        self.corr = AverageMeter()
        self.r1, self.r5 = AverageMeter(), AverageMeter()

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        if 'model_ema' in ckpt: # for the non rl model
            self.model.load_state_dict(ckpt['model_ema'])
            print('loading ema model')
        elif 'policy_net' in ckpt and self.usepolicy: # for the qlearning
            self.model.load_state_dict(ckpt['policy_net'])
            print('loading ema model (policy_net)')
        elif 'actor' in ckpt: # for the ppo
            self.model.load_state_dict(ckpt['actor'])
            print('loading actor model')
            # self.critic.load_state_dict(ckpt['critic'])
            # print('loading critic model')            
            
        else:
            self.model.load_state_dict(ckpt['model'])
            print('loading normal model')
        print(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        print("Evaluation started function run.")
        start_time = time.time()
        for i, data_tuple in enumerate(self.dataloader):
            if len(data_tuple) == 2:
                data, acc = data_tuple
                extra_info = None
            else:
                data, extra_info, acc = data_tuple
                extra_info = extra_info.cuda(non_blocking=True)
            
            data = data.cuda(non_blocking=True)
            pred = self.model(data, extra_info=extra_info).cpu()
            # if using window dataset and is not the first window
            # just pick the last prediction
            if len(pred.shape) == 3:
                pred = pred.squeeze(dim=0) # [bs, length, cls_dim]
            if len(acc.shape) == 3:
                acc = acc.squeeze(dim=0)                
            corr = spearman_rank_correlation(pred, acc)
            r1 = top_k_recall(pred, acc, k=1)
            r5 = top_k_recall(pred, acc, k=5)
            self.corr.update(corr)
            self.r1.update(r1)
            self.r5.update(r5)
            if i % 100 == 0:
                print('curr iteration:', i, 'total iteration:', len(self.dataloader))
        print(f"Spearman rank correlation: {self.corr.item():.2f}")
        print(f"Recall@1: {self.r1.item():.2f}")
        print(f"Recall@5: {self.r5.item():.2f}")
        
    @torch.no_grad()
    def run_rl(self):
        print("Evaluation started function run_rl.")
        conv_state, ssm_state = None, None
        start_time = time.time()
        for i, data_tuple in enumerate(self.dataloader):
            data, acc = data_tuple
            
            data = data.cuda(non_blocking=True)
            pred, conv_state, ssm_state = self.model(data, conv_state, ssm_state, data.device)
            pred = pred.cpu() # [bs, cls_dim]
            # if using window dataset and is not the first window
            # just pick the last prediction
            #pred = pred.squeeze(dim=0) 
            #acc = acc.squeeze(dim=0)                
            corr = spearman_rank_correlation(pred, acc)
            r1 = top_k_recall(pred, acc, k=1)
            r5 = top_k_recall(pred, acc, k=5)
            self.corr.update(corr)
            self.r1.update(r1)
            self.r5.update(r5)
            if i % 100 == 0:
                print('curr iteration:', i, 'total iteration:', len(self.dataloader))
        print(f"Spearman rank correlation: {self.corr.item():.2f}")
        print(f"Recall@1: {self.r1.item():.2f}")
        print(f"Recall@5: {self.r5.item():.2f}")
        
    @torch.no_grad()
    def run_rlv2(self, latency_filter_thresh=None, filter_pred=False, filter_gt=False):
        print("Evaluation started function run_rlv2.")
        
        for i, data_tuple in enumerate(self.dataloader):
            if len(data_tuple) == 3:
                data, acc, latency = data_tuple
            else:
                data, acc = data_tuple
                latency = None
                
            if filter_gt:
                assert latency is not None
                latency_mask = latency > latency_filter_thresh
                acc[latency_mask] = float('-inf')
            
            data = data.cuda(non_blocking=True)
            pred = self.model(data)
            pred = pred.cpu() # [bs, cls_dim]
            
            if filter_pred:
                assert latency is not None
                latency_mask = latency > latency_filter_thresh
                pred[latency_mask] = float('-inf')
            
            # if using window dataset and is not the first window
            # just pick the last prediction
            pred = pred.squeeze(dim=0) 
            acc = acc.squeeze(dim=0)                
            corr = spearman_rank_correlation(pred, acc)
            r1 = top_k_recall(pred, acc, k=1)
            r5 = top_k_recall(pred, acc, k=5)
            self.corr.update(corr)
            self.r1.update(r1)
            self.r5.update(r5)
            if i % 100 == 0:
                print('curr iteration:', i, 'total iteration:', len(self.dataloader))
        print(f"Spearman rank correlation: {self.corr.item():.2f}")
        print(f"Recall@1: {self.r1.item():.2f}")
        print(f"Recall@5: {self.r5.item():.2f}")
    
    @torch.no_grad()
    def run_rlv3(self, lat_threshold=None, opt=None, filter_pred=True, filter_ratio=1.0, interval_schedule=False, contention_level=None):
        # lat_threshold is the latency threshold we send into the model
        # opt is the config file
        # filter_pred means it filter the predictions using the latency threshold
        
        print("Evaluation started function run_rlv3.")
        
        ## load all the 40 branches
        if self.use_new_branches:
            if self.branch_filter_type == 'default':
                branches = NEW_BRANCHES
            elif self.branch_filter_type == 'filter_hl':
                branches = FILTER_HIGH_LATENCY_BRANCHES
            elif self.branch_filter_type == 'pf_only':
                branches = PARETO_FRONTIER_BRANCHES
            elif self.branch_filter_type == 'enlarge79':
                branches = ENLARGE79                
            elif self.branch_filter_type == 'enlarge82':
                branches = ENLARGE82          
            elif self.branch_filter_type == 'contention55':
                branches = CONTENTION55     
            elif self.branch_filter_type == 'contention12':
                branches = CONTENTION12  
            else:
                raise NotImplementedError
        else:
            branches = old_branches
        # assert the model classification head is the same dim
        assert self.opt['model']['num_classes'] == len(branches)
        # get the model 
        two_branches_model = self.opt['train']['two_branches_model'] if 'two_branches_model' in self.opt['train'] else False
        
        
        all_detection_res = []
        import pickle
        for b in branches: 
            curr_filename = os.path.join(self.profiling_root, (b + '_det.pkl'))
            with open(curr_filename, 'rb') as f:
                x = pickle.load(f)
                all_detection_res.append(x)
        
        all_selected_prediction = []
        all_selected_index = []
        all_latency = []
        
        for i, data_tuple in enumerate(self.dataloader):
            data = data_tuple['feature']
            latency = data_tuple['latency_value'] if 'latency_value' in data_tuple else None # (1, seqence_len, branches)
            average_latency = data_tuple['avg_latency_value'] if 'avg_latency_value' in data_tuple else None 
            
            # select the latency for the special case
            if self.branch_filter_type == 'filter_hl':
                #print('before filter:', latency.shape)
                idx = torch.tensor(filter_hl_branches_idx)
                latency = latency[..., idx]
                #print('after filter:', latency.shape)
            elif self.branch_filter_type == 'pf_only':
                idx = torch.tensor(pareto_frontier_branches_idx)
                latency = latency[..., idx]       
                
            # send the data into the model and the latency(TODO)
            data = [ele.cuda(non_blocking=True) for ele in data]
            
            # create the latency requirement
            if lat_threshold is not None:
                latency_input = torch.tensor([[lat_threshold]]).repeat(data[0].shape[0], data[0].shape[1]).to(data[0].device) # (B, L)
            else:
                latency_input = torch.tensor([[500]]).repeat(data[0].shape[0], data[0].shape[1]).to(data[0].device) # (B, L)
                
            if contention_level is not None:
                latency_input = torch.tensor([[contention_level]]).repeat(data[0].shape[0], data[0].shape[1]).to(data[0].device) # (B, L)
            
            # get the currenet sequence start index
            curr_seq_start_idx = self.dataloader.dataset.seq_range[i][0]
            curr_seq_end_idx = self.dataloader.dataset.seq_range[i][1]
            scheduler_call_interval = opt['train']['data']['scheduler_call_interval'] if 'scheduler_call_interval' in opt['train']['data'] else None
            use_det_res = opt['train']['use_det_res'] if 'use_det_res' in opt['train'] else False
            
            if interval_schedule and scheduler_call_interval is None: # the old version of the interval scheduling (only eval use interval)
                # we assume we have a content window (which the scheduler make the decision base on the content in the window)
                # and a apply window, which the sheduler prediction will be apply
                
                # determin the total window size
                content_wind_size = opt['train']['loss_start_idx'] if 'loss_start_idx' in opt['train'] else 0
                total_wind_size = opt['train']['data']['wind_size']
                apply_wind_size = total_wind_size - content_wind_size
                assert total_wind_size > 0
                assert apply_wind_size > 0
                
                for wind_start_idx in range(0, data[0].shape[1], apply_wind_size):
                    # the content wind is (wind_start_idx: wind_start_idx + content_wind_size)
                    # the apply wind is: (wind_start_idx + content_wind_size: wind_start_idx + content_wind_size + apply_wind_size)
                    
                    # handle the edge case
                    if wind_start_idx + content_wind_size > data.shape[1]: # no apply window
                        break
                    # determine the data and the lantency
                    curr_data = [ele[:, wind_start_idx : wind_start_idx + total_wind_size] for ele in data]# (B, content_wind_size, x, y, z)
                    curr_lat_thres = latency_input[:, wind_start_idx : wind_start_idx + total_wind_size] # (B, content_wind_size)  
                    #curr_profile_lat = latency[:, wind_start_idx : wind_start_idx + total_wind_size] # (B, content_wind_size)  
                    curr_avg_lat = average_latency[:, wind_start_idx : wind_start_idx + total_wind_size] # (B, content_wind_size)  
                    
                    # ipdb.set_trace()
                    # start_time = time.time()
                    pred = self.model(curr_data, curr_lat_thres)
                    # print('inference time:', time.time() - start_time)
                    pred = pred.cpu() # [bs, T , cls_dim]
                    pred = pred.squeeze(dim=0) #[T, cls_dim]

                    # prefilter the prediction using the latency
                    if lat_threshold is not None and filter_pred:
                        assert average_latency is not None  # latency shoud be (B, L, cls_dim)
                        # if self.punishment_buffer_type == 'ratio':
                        #     mask = curr_profile_lat > lat_threshold * 1.1
                        # else:
                        #     mask = curr_profile_lat > lat_threshold + 10
                        mask = curr_avg_lat > lat_threshold * filter_ratio
                        mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)
                        pred[mask] = float('-inf')
                    
                    # select the max idx
                    #print('pred:', pred.shape, pred)
                    all_idxes = torch.argmax(pred, dim=-1)
                    # print('all_idxes:', all_idxes)

                    # handle the special case for the first sequence
                    for idx_in_curr_wind, curr_branch in enumerate(all_idxes):
                        if wind_start_idx != 0 and idx_in_curr_wind < content_wind_size: # handle the special case for the first window
                            continue
                        # print('in edge case:', idx_in_curr_seq)
                        idx_in_curr_seq = idx_in_curr_wind + wind_start_idx
                        # print('idx_in_curr_seq:', idx_in_curr_seq)
                        idx_in_whole_dataset = curr_seq_start_idx + idx_in_curr_seq
                        all_selected_prediction.append(all_detection_res[curr_branch.item()][idx_in_whole_dataset])
                        # save the per-frame mAP, 
                        all_latency.append(latency[0][idx_in_curr_seq][curr_branch]) # latency (1, seq_len, num_branches)
                        all_selected_index.append(curr_branch.item())
                    
                # ipdb.set_trace()
            elif interval_schedule and scheduler_call_interval is not None and not use_det_res: # new version of interval scheduling (training with interval)
                assert scheduler_call_interval > 0
                # pick the idxes which should be sent in and forward
                scheduler_called_idx = torch.tensor([called_idx for called_idx in range(0, data[0].shape[1], scheduler_call_interval)]).to(data[0].device)
                scheduler_called_feat = [ele[:, scheduler_called_idx] for ele in data]
                scheduler_called_lat_thresh = latency_input[:, scheduler_called_idx]
                average_latency = average_latency.cuda()
                curr_avg_lat = average_latency[:, scheduler_called_idx] # (B, content_wind_size)  
                
                pred = self.model(scheduler_called_feat, scheduler_called_lat_thresh)
                pred = pred.cpu() # [bs, T , cls_dim]
                pred = pred.squeeze(dim=0) #[T, cls_dim]
                # ipdb.set_trace() # check : lat_threshold is not None and filter_pred:
                # prefilter the prediction using the latency
                if lat_threshold is not None and filter_pred:
                    assert average_latency is not None  # latency shoud be (B, L, cls_dim)
                    # if self.punishment_buffer_type == 'ratio':
                    #     mask = curr_profile_lat > lat_threshold * 1.1
                    # else:
                    #     mask = curr_profile_lat > lat_threshold + 10
                    mask = curr_avg_lat > lat_threshold * filter_ratio
                    mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)
                    pred[mask] = float('-inf')   
                
                all_idxes = torch.argmax(pred, dim=-1)
                
                # reconsturction the whole prediction
                for curr_call_idx, curr_branch in enumerate(all_idxes):
                    curr_call_start_idx_in_seq = curr_call_idx * scheduler_call_interval
                    curr_call_end_idx_in_seq = (curr_call_idx + 1) * scheduler_call_interval # this is not include the end
                    if curr_call_end_idx_in_seq > len(data[0][0]): # determine the last index
                        curr_call_end_idx_in_seq = len(data[0][0])
                        
                    for idx_in_curr_seq in range(curr_call_start_idx_in_seq, curr_call_end_idx_in_seq):
                        idx_in_whole_dataset = curr_seq_start_idx + idx_in_curr_seq
                        # save the predition, the latency, and the idx
                        all_selected_prediction.append(all_detection_res[curr_branch.item()][idx_in_whole_dataset])
                        all_latency.append(latency[0][idx_in_curr_seq][curr_branch]) # latency (1, seq_len, num_branches)
                        all_selected_index.append(curr_branch.item())   
                        # print('idx_in_curr_seq:', idx_in_curr_seq, 'idx_in_whole_dataset:', idx_in_whole_dataset)
            elif interval_schedule and scheduler_call_interval is not None and use_det_res: # most of the newesting method will go throught this
                # select the feature on the called index
                assert scheduler_call_interval > 0
                scheduler_called_idx = torch.tensor([called_idx for called_idx in range(0, data[0].shape[1], scheduler_call_interval)]).to(data[0].device)
                scheduler_called_feat = [ele[:, scheduler_called_idx] for ele in data]
                scheduler_called_lat_thresh = latency_input[:, scheduler_called_idx]
                average_latency = average_latency.cuda()
                scheduler_called_avg_lat = average_latency[:, scheduler_called_idx] # (B, content_wind_size)  
                
                prepared_det_info = (None, None, None)
                action_state = [(None, None)] * len(self.model.layers)
                called_idx_len = scheduler_called_feat[0].shape[1]
                # use the for loop to step over all the called index
                for curr_call_idx in range(called_idx_len):
                    curr_feat = [ele[:, curr_call_idx].unsqueeze(dim=1) for ele in scheduler_called_feat]
                    curr_lat_thres = scheduler_called_lat_thresh[:, curr_call_idx].unsqueeze(dim=1)
                    curr_avg_lat = scheduler_called_avg_lat[:, curr_call_idx]
                    # foward get the action
                    # ipdb.set_trace() # check contention level
                    # start_time = time.time()
                    if two_branches_model:
                        (map_x, lat_x), pred, action_state = \
                                self.model.step(curr_feat, curr_lat_thres, prepared_det_info, action_state)
                                     
                    else:
                        pred, action_state = \
                                self.model.step(curr_feat, curr_lat_thres, prepared_det_info, action_state)
                    # print('inference time:', time.time() - start_time)
                    pred = pred.cpu() # [bs, T , cls_dim]
                    pred = pred.squeeze(dim=0) #[T, cls_dim] (1, dim)
                    
                    if lat_threshold is not None and filter_pred:
                        assert average_latency is not None  # latency shoud be (B, L, cls_dim)
                        mask = curr_avg_lat > lat_threshold * filter_ratio # (B, cls_dim)
                        # mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)
                        pred[mask] = float('-inf')                       
                    
                    all_idxes = torch.argmax(pred, dim=-1)     
                    
                    # pick the detection result for current step and following steps
                    curr_call_start_idx_in_seq = curr_call_idx * scheduler_call_interval
                    curr_call_end_idx_in_seq = (curr_call_idx + 1) * scheduler_call_interval # this is not include the end
                    if curr_call_end_idx_in_seq > len(data[0][0]): # determine the last index
                        curr_call_end_idx_in_seq = len(data[0][0])
                    # ipdb.set_trace()
                    for idx_in_curr_seq in range(curr_call_start_idx_in_seq, curr_call_end_idx_in_seq):
                        idx_in_whole_dataset = curr_seq_start_idx + idx_in_curr_seq
                        # save the predition, the latency, and the idx
                        all_selected_prediction.append(all_detection_res[all_idxes[0].item()][idx_in_whole_dataset])
                        all_latency.append(latency[0][idx_in_curr_seq][all_idxes[0].item()]) # latency (1, seq_len, num_branches)
                        all_selected_index.append(all_idxes[0].item())   
                        # print('idx_in_curr_seq:', idx_in_curr_seq, 'idx_in_whole_dataset:', idx_in_whole_dataset)
                    
                    # convert the information
                    det_info = all_selected_prediction[-(curr_call_end_idx_in_seq - curr_call_start_idx_in_seq)]
                    prepared_det_info = convert_all_predictions(det_info)
                
            else:
                pred = self.model(data, latency_input)
                # value_pred = self.critic(data, latency_input)
                pred = pred.cpu() # [B, L, cls_dim]
                pred = pred.squeeze(dim=0) # [L, cls_dim]
            
                # create a mask and mask the predicted q value
                if lat_threshold is not None and filter_pred:
                    assert latency is not None  # latency shoud be (B, L, cls_dim)
                    mask = average_latency > lat_threshold * filter_ratio
                    mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)             
                    
                    pred[mask] = float('-inf')
                
                # select the max idx
                all_idxes = torch.argmax(pred, dim=-1)
                all_selected_index.append(all_idxes)

                for idx_in_curr_seq, curr_branch in enumerate(all_idxes):
                    idx_in_whole_dataset = curr_seq_start_idx + idx_in_curr_seq
                    all_selected_prediction.append(all_detection_res[curr_branch.item()][idx_in_whole_dataset])
                    # print('infenence:', latency[0][idx_in_whole_dataset][curr_branch])
                    all_latency.append(latency[0][idx_in_curr_seq][curr_branch]) # latency (1, seq_len, num_branches)

        print('final average latency:', sum(all_latency) / len(all_latency))

        # dump the result
        return all_selected_prediction, all_selected_index
                
        