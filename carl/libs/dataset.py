from copy import deepcopy
import os
import glob
import random
import numpy as np
import json
import math
import ipdb

import torch
from torch.utils.data import Dataset, DataLoader

from .train_utils import worker_init_reset_seed


class WaymoWindowRL(Dataset):

    def __init__(
        self,
        root,               # dataset root folder
        split='val',        # split name (val | test)
        # occupancy grid config
        size=(300, 300),    # input spatial size (X, Y)
        use_occ=False,       # whether to use occupancy grid (X, Y, C)
        use_intensity=False, # whether to use intensity map (X, Y)
        use_elongation=False,# whether to use elongation map (X, Y)
        load_downsampled=False,
        # feature choice
        use_bev_feat=False,
        use_gd_mae_feat=False,
        gd_feat_path=None,
        # config of the interval scheduling
        max_scheduler_call=5,
        min_scheduler_call=2,
        scheduler_call_interval=10,        
        # for loading the profiling data
        wind_size=10,
        random_sample_len=False, # random sample len for augmentation
        load_acc=True,
        acc_path=None,
        load_latency=False,
        latency_path=None,
        load_loss=False,
        loss_path=None,
        # reward prepare
        return_reward=True,
        reward_type='acc',
        normalize_acc=False,
        soft_adjust=False,
        rescale_reward=True,
        # latency value prepare
        latency_adjusted_reward=False,
        return_latency_value=True,
        mean_latency_value=False, # return the mean of the lantency value
        # return latency threshold (randomly generated)
        latency_threshold=200,
        latency_per_wind=False,
        return_latency_thresh=True,
        latency_threshold_type='fixed',
        # for online mAP
        return_idx=False,
        # for some special experiments
        random_feat=False,
    ):
        super().__init__()
        print('in dataset init')
        '''
            This dataset aims to prepare the data for training a scheduler which will be call every
            a few time step.
        '''
        self.root = root
        self.split = split
        # occupancy grid feature config
        self.size = size
        if not use_bev_feat and not use_gd_mae_feat: # if using occupancy grid, make sure we load at least one data
            assert use_occ or use_intensity or use_elongation
        # for occ feature
        self.use_occ = use_occ                  
        self.use_intensity = use_intensity
        self.use_elongation = use_elongation
        self.load_downsampled = load_downsampled
        # for bev feature
        self.use_bev_feat = use_bev_feat
        # for gd-mae feature
        self.use_gd_mae_feat = use_gd_mae_feat
        self.gd_feat_path = gd_feat_path        
        
        # config of the interval scheduling
        self.max_scheduler_call = max_scheduler_call # include the first idx
        self.min_scheduler_call = min_scheduler_call # include the first idx
        self.scheduler_call_interval = scheduler_call_interval
        
        # handle the profiling data
        self.wind_size = wind_size
        self.random_sample_len = random_sample_len
        self.load_acc = load_acc
        self.acc_path = acc_path
        self.load_latency = load_latency
        self.latency_path = latency_path
        self.load_loss = load_loss
        self.loss_path = loss_path
        
        # for reward
        self.return_reward = return_reward
        self.reward_type = reward_type
        self.normalize_acc = normalize_acc
        self.soft_adjust = soft_adjust # if soft_adjust is False then we mask the value to 0, other wise we divide the value by half
        self.rescale_reward = rescale_reward
        
        # for latency input
        self.latency_adjusted_reward = latency_adjusted_reward
        self.return_latency_value = return_latency_value
        self.mean_latency_value = mean_latency_value    
            
        # for latency threshold
        self.latency_threshold = latency_threshold
        self.latency_per_wind = latency_per_wind
        self.return_latency_thresh = return_latency_thresh
        self.latency_threshold_type = latency_threshold_type # if 'random' then random select a value , if it is 'continuous' then select a value from the interval

        # for online eval mAP
        self.return_idx = return_idx        
        
        # for some special experiments
        self.random_feat = random_feat    
        
        # some assertion
        if self.latency_threshold_type == 'continuous':
            assert len(self.latency_threshold) == 2 # for continuous only give the lower bound and the higher bound
        
        # data list
        data_folder = os.path.join(root, split)
        
        # load the feature list or the occ grid datalist
        if self.use_gd_mae_feat:
            gd_mae_file_name_list = sorted(glob.glob(os.path.join(self.gd_feat_path, '*_downsampled.pt'))) # 28351.pt
            intensity_list = None
            elongation_list = None 
            # from_idx_to_gd_mae_file = {ele.split('/')[-1].rstrip('_downsampled.pt'):ele for ele in gd_mae_file_name_list}
            self.gd_mae_file_name_list = gd_mae_file_name_list
            self.gd_mae_file_idx_list = [i for i in range(len(gd_mae_file_name_list))]
        else:
            self.gd_mae_file_idx_list = None
        
        if self.use_bev_feat:
            bev_feat_file_name_list = sorted(glob.glob(os.path.join(data_folder, '*.pt'))) # 28351.pt
            intensity_list = None
            elongation_list = None 
            self.bev_feat_file_name_list = bev_feat_file_name_list
            self.bev_feat_file_idx_list = [i for i in range(len(bev_feat_file_name_list))]
        else:
            self.bev_feat_file_idx_list = None
        
        # ipdb.set_trace()
        if self.use_occ: # use occ feature
            if self.load_downsampled:
                self.occ_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'occ', '*_downsampled_bitpacked.npy')))
                self.intensity_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'intensity', '*_downsampled.npy')))
                self.elongation_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'elongation', '*_downsampled.npy')))        
                self.occ_file_idx_list = [i for i in range(len(self.occ_list))]
            else:
                occ_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'occ', '*.npy')))
                self.occ_list = [ele for ele in occ_list if not ele.endswith('_downsampled_bitpacked.npy')]
                intensity_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'intensity', '*.npy')))
                self.intensity_list = [ele for ele in intensity_list if not ele.endswith('_downsampled.npy')]
                elongation_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'elongation', '*.npy')))
                self.elongation_list = [ele for ele in elongation_list if not ele.endswith('_downsampled.npy')]
                self.occ_file_idx_list = [i for i in range(len(self.occ_list))]
        else:
            self.occ_file_idx_list = None

        # handle the final index
        # ipdb.set_trace() # check whether the len of different list is the same, the order of different items in the list are the same
        if self.gd_mae_file_idx_list is not None:
            self.index_list = self.gd_mae_file_idx_list
        elif self.bev_feat_file_idx_list is not None:
            self.index_list = self.bev_feat_file_idx_list
        else:
            self.index_list = self.occ_file_idx_list

        # load the profiled acc (the per-frame per-branch mAP)
        if self.load_acc:
            if self.acc_path is None:
                path = os.path.join(data_folder, 'accuracy.npy')
            else:
                path = self.acc_path
            acc = np.load(path) # (#samples, #branches)
            assert acc.shape[0] == len(self.index_list)
            self.acc = torch.from_numpy(acc.astype(np.float32))
        else:
            self.acc = None
        
        # load the profiled latency (the per-frame per branch latency)
        if self.load_latency:
            if self.latency_path is None:
                path = os.path.join(data_folder, 'latency.npy')
            else:
                path = self.latency_path
            latency = np.load(path) # (#samples, #branches)
            assert latency.shape[0] == len(self.index_list)
            self.latency = torch.from_numpy(latency.astype(np.float32))
            self.mean_latency = torch.mean(self.latency, dim=0)
        else:
            self.latency = None
        
        # load the profiled loss
        if self.load_loss:
            if self.loss_path is None:
                path = os.path.join(data_folder, 'loss.npy')
            else:
                path = self.loss_path            
            loss_val = np.load(path) # (#samples, #branches)
            assert loss_val.shape[0] == len(self.index_list)
            self.loss_val = torch.from_numpy(loss_val.astype(np.float32))
        else:
            self.loss_val = None
        
        # load sequence range
        seq_content = json.load(open(os.path.join(data_folder, 'sequences_first_occurrence.json'))) # (#samples, #branches)
        self.seq_range = {}
        start = 0
        for i, key in enumerate(seq_content):
            if i == 0:
                continue
            elif i == len(seq_content.keys()) -1:
                prev_range = [start, seq_content[key]-1]
                self.seq_range[i-1] = prev_range
                start = seq_content[key]
                curr_range = [start, 31894]
                self.seq_range[i] = curr_range
            else:
                prev_range = [start, seq_content[key]-1]
                self.seq_range[i-1] = prev_range
                start = seq_content[key]

    def __len__(self):
        return len(self.index_list)

    def _load_occ(self, idx):
        '''For occupancy grid, load the elongation for a timestamp.'''
        if self.load_downsampled:
            occ = np.unpackbits(np.load(self.occ_list[idx]))
            occ = occ.reshape(150, 150, 30).astype(np.float32)
        else:     
            occ = np.unpackbits(np.load(self.occ_list[idx]))
            occ = occ.reshape(300, 300, 60).astype(np.float32)
        h, w = occ.shape[:2]
        if self.size[0] < h:
            margin = int((h - self.size[0]) // 2)
            occ = occ[margin:margin + self.size[0]]
        if self.size[1] < w:
            occ = occ[:, :self.size[1]]
        assert occ.shape[0] == self.size[0] and occ.shape[1] == self.size[1]
        return occ              # (h, w, c)

    def _load_intensity(self, idx):
        '''For occupancy grid, load the elongation for a timestamp.'''
        intensity = np.load(self.intensity_list[idx]).astype(np.float32)
            
        h, w = intensity.shape[:2]
        if self.size[0] < h:
            margin = int((h - self.size[0]) // 2)
            intensity = intensity[margin:margin + self.size[0]]
        if self.size[1] < w:
            intensity = intensity[:, :self.size[1]]
        assert intensity.shape[0] == self.size[0] and intensity.shape[1] == self.size[1]
        return intensity[..., None]   # (h, w, 1)

    def _load_elongation(self, idx):
        '''For occupancy grid, load the elongation for a timestamp.'''
        elongation = np.load(self.elongation_list[idx]).astype(np.float32)
        h, w = elongation.shape[:2]
        if self.size[0] < h:
            margin = int((h - self.size[0]) // 2)
            elongation = elongation[margin:margin + self.size[0]]
        if self.size[1] < w:
            elongation = elongation[:, :self.size[1]]
        assert elongation.shape[0] == self.size[0] and elongation.shape[1] == self.size[1]
        return elongation[..., None]   # (h, w, 1)

    def get_one_data_point(self, idx):
        '''
            Load the occupancy grid feature, 
            GD-MAE feature or BEV features for one timestamps.
        '''
        all_features = []
        
        if self.use_gd_mae_feat: # GD-MAE feature
            if hasattr(self, 'random_feat') and self.random_feat:
                rand_idx = torch.randint(low=0, high=len(self.occ_list), size=(1,))
                data = torch.load(self.gd_mae_file_name_list[rand_idx], map_location=torch.device('cpu')) # torch.Size([192, 256, 256]) / torch.Size([192, 128, 128])
            else:
                # load the feature 
                data = torch.load(self.gd_mae_file_name_list[idx], map_location=torch.device('cpu')) # torch.Size([192, 256, 256]) / torch.Size([192, 128, 128])
            all_features.append(data.unsqueeze(dim=0))
        if self.use_bev_feat: # BEV features
            data = torch.load(self.bev_feat_file_name_list[idx], map_location=torch.device('cpu')) # torch.Size([192, 256, 256]) / torch.Size([192, 128, 128])
            # print('data:', data.shape)
            all_features.append(data.unsqueeze(dim=0))
        
        if self.use_occ: # occupancy grid feature
            data = np.zeros((*self.size, 0), dtype=np.float32)
            if self.use_occ:
                occ = self._load_occ(idx)
                data = np.concatenate((data, occ), axis=-1)
            if self.use_intensity:
                intensity = self._load_intensity(idx)
                data = np.concatenate((data, intensity), axis=-1)
            if self.use_elongation:
                elongation = self._load_elongation(idx)
                data = np.concatenate((data, elongation), axis=-1)
            data = np.ascontiguousarray(data.transpose(2, 0, 1))
            data = torch.from_numpy(data)   # (c, h, w) in [0, 1] # data: torch.Size([32, 150, 150])

            all_features.append(data.unsqueeze(dim=0))
        
        return all_features
    
    def load_reward(self, idx, latency_thresh=None):
        '''
            Load the acc or the loss as rewards of a timestamp, 
            and use latency threshold (requirement) to adjust the rewards.
        '''
        
        if self.reward_type == 'acc': # use acc as reward
            assert self.acc is not None
            reward = self.acc[idx]
            # normalize the target (test - test.min(1, keepdim=True)[0]) / (test.max(1, keepdim=True)[0] - test.min(1, keepdim=True)[0])
            # if self.normalize_acc:
            if self.rescale_reward:
                reward = 2 * torch.exp((reward - reward.max()))
        elif self.reward_type == 'loss': # use loss as rewards
            assert self.loss_val is not None
            reward = self.loss_val[idx]
            # define the reward exp(-(ele-minval)) 
            #print('reward before:', reward)
            # the reward will between 0 to 2
            if self.rescale_reward:
                reward = 2 * torch.exp(-(reward - reward.min()))
            #print('reward after:', reward)
        else:
            raise NotImplementedError
    
        if self.latency_adjusted_reward: # ensure the adjustment about -2 to 0?
            assert latency_thresh is not None
            assert self.latency is not None
            curr_all_latency = self.latency[idx]
            reward_mask = curr_all_latency > latency_thresh
            #print('before mask:', reward)
            if self.soft_adjust:
                reward[reward_mask] /= 2
            else:
                reward[reward_mask] = 0
            #print('after mask:', reward)

        return reward.unsqueeze(dim=0)
    
    def load_latency(self, idx):
        '''
            Load the profiling latency value for on time stamps.
        '''        
        
        latency = self.latency[idx]
        return latency.unsqueeze(dim=0)
    
    def __getitem__(self, idx):
        # judge the index in which sequence
        if self.wind_size > 1:
            if self.random_sample_len: # if the wind len is 15 then the random sample should range from 8 to 23
                curr_window_size = int(self.wind_size * random.uniform(0.5, 1.5))
            else:
                curr_window_size = self.wind_size
            
            range_idx = math.floor(idx / 200)
            curr_seq_range = self.seq_range[range_idx]
            # ensure the range index
            while idx > curr_seq_range[-1]:
                range_idx += 1
                curr_seq_range = self.seq_range[range_idx]
            assert idx >= curr_seq_range[0] and idx <= curr_seq_range[1]
            
            # make sure the index + window smaller than the end of the sequence end
            # if the end of the current range large, adjust the range
            if idx + curr_window_size - 1 >= curr_seq_range[1]:
                idx = curr_seq_range[1] - curr_window_size + 1
        
        all_idx = np.arange(idx, idx + curr_window_size) # the most naive setting
        
        # load 10 sample
        data_list = []
        reward_list = []
        latency_thresh_list = []
        latency_value_list = []
        curr_lat_thresh = None
        
        for curr_idx in all_idx:
            # add the new result
            #curr_idx = idx + i
            temp_data = self.get_one_data_point(curr_idx)
            data_list.append(temp_data)
            
            # prepare latency threshold
            if self.return_latency_thresh:
                if self.latency_per_wind and curr_lat_thresh is not None:# the threshold number is continuous and but keep for the same for the whole window
                    curr_lat_thresh = curr_lat_thresh
                else: # if the latency threshold for each time stamp, or it is the first time stamp of latency_per_wind
                    if self.latency_threshold_type == 'random':
                        curr_lat_thresh = self.latency_threshold[random.randint(0, len(self.latency_threshold)-1)]
                    elif self.latency_threshold_type == 'continuous':
                        assert len(self.latency_threshold) == 2
                        high = self.latency_threshold[1] - 1
                        low = self.latency_threshold[0]
                        curr_lat_thresh = random.randint(low, high)
                    else:
                        raise NotImplementedError
                latency_thresh_list.append(torch.tensor([curr_lat_thresh]))     
            else:
                curr_lat_thresh = None
                
            if self.return_latency_value:
                if self.mean_latency_value:
                    latency_value_list.append(self.mean_latency_value.unsqueeze(dim=0)) 
                else:
                    latency_value_list.append(self.latency[curr_idx].unsqueeze(dim=0)) 
            
            # load the reward
            temp_reward = self.load_reward(curr_idx, curr_lat_thresh)
            reward_list.append(temp_reward)

        # handle the concatenation of the feature
        if len(data_list[0]) == 1: # if there is only one features
            feature = [torch.cat([ele[0] for ele in data_list], dim=0)]
        else: # if there is multiple features
            # ipdb.set_trace() # check the feature
            num_of_feat = len(data_list[0])
            feature = []
            for i in range(num_of_feat):
                curr_feat_set = [ele[i] for ele in data_list]
                curr_feat_set = torch.cat(curr_feat_set, dim=0)
                feature.append(curr_feat_set)
            # ipdb.set_trace() # check the feature

        # the acc list should be the mAP or reward
        all_infos = {
            'feature': feature,
            'reward': torch.cat(reward_list, dim=0) if len(reward_list) > 1 else reward_list[0].squeeze(dim=0),
        }
        if self.return_latency_thresh:
            all_infos['latency_thres'] = torch.cat(latency_thresh_list, dim=0)
        
        if self.return_idx:
            all_infos['idx'] = torch.tensor(idx)
        
        if self.return_latency_value:
            all_infos['latency_value'] = torch.cat(latency_value_list, dim=0)
        
        return all_infos


class WaymoWindowRLInterval(WaymoWindowRL):    
    def __getitem__(self, idx):
        '''
        
            Since in this setting, the scheduler is called every couple timestamps:
            
            1. determine the scheduler called index
            2. get the feature on this index
            3. get the lantency and the acc during this call        
        '''
        
        # find which sequence current index in
        range_idx = math.floor(idx / 200)
        curr_seq_range = self.seq_range[range_idx]
        # ensure the range index
        while idx > curr_seq_range[-1]:
            range_idx += 1
            curr_seq_range = self.seq_range[range_idx]
        assert idx >= curr_seq_range[0] and idx <= curr_seq_range[1]
        
        # make sure index to the end of the sequence is (min_scheduler_call - 1) * scheduler_call_interval + 2 (we can take the last index)
        if idx + (self.min_scheduler_call - 1) * self.scheduler_call_interval + 1 >= curr_seq_range[1]:
            idx = curr_seq_range[1] - ((self.min_scheduler_call - 1) * self.scheduler_call_interval + 1)
            called_idx = [idx, idx + self.scheduler_call_interval]
            #called_sequence_range = [idx, curr_seq_range[1] + 1] # not include the end (we should not select the end)
            train_sequence_end = curr_seq_range[1] + 1 # not include the end (we should not select the end)
            
        else: # if current index to the end of the sequence is more than 2 call, then random sample
            distance_to_end = curr_seq_range[1] - idx
            num_of_call_to_end = math.ceil(distance_to_end / self.scheduler_call_interval)
            max_call = num_of_call_to_end if num_of_call_to_end <= self.max_scheduler_call else self.max_scheduler_call 
            # random sample a call number
            random_call_num = random.randint(self.min_scheduler_call, max_call)
            # get the call index and the sequence range
            called_idx = [self.scheduler_call_interval*i + idx for i in range(random_call_num)]
            # determine the end with the end of the sequence
            # called_sequence_range = [idx, min(curr_seq_range[1], called_idx[-1] + self.scheduler_call_interval - 1) + 1] # not include the end (we should not select the end)
            train_sequence_end = min(curr_seq_range[1], called_idx[-1] + self.scheduler_call_interval - 1) + 1
            
        # load feature, rewards, latency on the called timestamps
        data_list = []
        reward_list = []
        latency_thresh_list = []
        latency_value_list = []
        curr_lat_thresh = None
        
        for curr_idx in called_idx:
            # get the feature
            temp_data = self.get_one_data_point(curr_idx)
            data_list.append(temp_data)
            curr_end_idx = min(curr_seq_range[1], curr_idx + self.scheduler_call_interval - 1) + 1
            
            # prepare latency threshold
            if self.return_latency_thresh:
                if self.latency_per_wind and curr_lat_thresh is not None:# the threshold number is continuous and but keep for the same for the whole window
                    curr_lat_thresh = curr_lat_thresh
                else: # if the latency threshold for each time stamp, or it is the first time stamp of latency_per_wind
                    if self.latency_threshold_type == 'random':
                        curr_lat_thresh = self.latency_threshold[random.randint(0, len(self.latency_threshold)-1)]
                    elif self.latency_threshold_type == 'continuous':
                        assert len(self.latency_threshold) == 2
                        high = self.latency_threshold[1] - 1
                        low = self.latency_threshold[0]
                        curr_lat_thresh = random.randint(low, high)
                    else:
                        raise NotImplementedError
                latency_thresh_list.append(torch.tensor([curr_lat_thresh]))     
            else:
                curr_lat_thresh = None
            
            ### Although we only collect the feature only on the called indexes
            ### The profiled latency and the accuracy of all branches are collected over the whole window
            if self.return_latency_value:
                if self.mean_latency_value:
                    latency_value_list.append(self.mean_latency_value.unsqueeze(dim=0)) 
                else:
                    latency_value_list.append(torch.mean(self.latency[curr_idx:curr_end_idx], dim=0).unsqueeze(dim=0)) 
            
            # load the reward (acc), after averaged over the interval
            temp_reward = torch.mean(self.acc[curr_idx:curr_end_idx], dim=0).unsqueeze(dim=0)
            reward_list.append(temp_reward)

        # handle the concatenation of the feature
        if len(data_list[0]) == 1: # if there is only one features
            feature = [torch.cat([ele[0] for ele in data_list], dim=0)]
        else: # if there is multiple features
            # ipdb.set_trace() # check the feature
            num_of_feat = len(data_list[0])
            feature = []
            for i in range(num_of_feat):
                curr_feat_set = [ele[i] for ele in data_list]
                curr_feat_set = torch.cat(curr_feat_set, dim=0)
                feature.append(curr_feat_set)
            # ipdb.set_trace() # check the feature ipdb> feature[0].shape torch.Size([2, 128]) ipdb> feature[1].shape torch.Size([2, 32, 150, 150])

        # the acc list should be the mAP or reward
        all_infos = {
            'feature': feature,
            'reward': torch.cat(reward_list, dim=0) if len(reward_list) > 1 else reward_list[0].squeeze(dim=0),
        }
        if self.return_latency_thresh:
            all_infos['latency_thres'] = torch.cat(latency_thresh_list, dim=0)
        
        if self.return_idx:
            called_idx.append(train_sequence_end)
            all_infos['idx'] = torch.tensor(called_idx)
        
        if self.return_latency_value:
            all_infos['latency_value'] = torch.cat(latency_value_list, dim=0)
        
        return all_infos


class WaymoWindowRLEval(WaymoWindowRL):

    def __init__(
        self,
        root,               # dataset root folder
        split='test',        # split name (val | test)
        size=(300, 300),    # input spatial size (X, Y)
        use_occ=False,       # whether to use occupancy grid (X, Y, C)
        use_intensity=False, # whether to use intensity map (X, Y)
        use_elongation=False,# whether to use elongation map (X, Y)
        load_downsampled=False, 
        # use feature
        use_bev_feat = False,
        use_gd_mae_feat = False,
        gd_feat_path = None,
        
        # for loading
        load_acc=False,
        acc_path=None,
        load_latency=False,
        latency_path=None,
        load_loss=False,
        loss_path=None,
        # reward prepare
        return_reward=True,
        reward_type='acc',
        latency_adjusted_reward=False,
        latency_threshold_type='fixed',
        latency_threshold=200,

    ):
        self.root = root
        self.split = split
        self.size = size
        # if using occupancy grid, make sure we load at least one data
        if not use_bev_feat and not use_gd_mae_feat:
            assert use_occ or use_intensity or use_elongation
        self.use_occ = use_occ
        self.use_intensity = use_intensity
        self.use_elongation = use_elongation
        self.load_downsampled = load_downsampled
        # for bev feature
        self.use_bev_feat = use_bev_feat
        # for gd-mae feature
        self.use_gd_mae_feat = use_gd_mae_feat
        self.gd_feat_path = gd_feat_path            
        
        # for reward
        self.return_reward = return_reward
        self.reward_type = reward_type
        self.latency_adjusted_reward = latency_adjusted_reward
        self.latency_threshold_type = latency_threshold_type
        self.latency_threshold = latency_threshold

        # data list
        data_folder = os.path.join(root, split)
        # load the feature list or the occ grid datalist
        if self.use_gd_mae_feat:
            gd_mae_file_name_list = sorted(glob.glob(os.path.join(self.gd_feat_path, '*_downsampled.pt'))) # 28351.pt
            intensity_list = None
            elongation_list = None 
            # ipdb.set_trace()
            # from_idx_to_gd_mae_file = {ele.split('/')[-1].rstrip('_downsampled.pt'):ele for ele in gd_mae_file_name_list}
            self.gd_mae_file_name_list = gd_mae_file_name_list
            self.gd_mae_file_idx_list = [i for i in range(len(gd_mae_file_name_list))]
        else:
            self.gd_mae_file_idx_list = None
        
        if self.use_bev_feat:
            bev_feat_file_name_list = sorted(glob.glob(os.path.join(data_folder, '*.pt'))) # 28351.pt
            intensity_list = None
            elongation_list = None 
            self.bev_feat_file_name_list = bev_feat_file_name_list
            self.bev_feat_file_idx_list = [i for i in range(len(bev_feat_file_name_list))]
        else:
            self.bev_feat_file_idx_list = None
        
        if self.use_occ: # use occ feature
            if self.load_downsampled:
                self.occ_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'occ', '*_downsampled_bitpacked.npy')))
                self.intensity_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'intensity', '*_downsampled.npy')))
                self.elongation_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'elongation', '*_downsampled.npy')))        
                self.occ_file_idx_list = [i for i in range(len(self.occ_list))]
            else:
                occ_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'occ', '*.npy')))
                self.occ_list = [ele for ele in occ_list if not ele.endswith('_downsampled_bitpacked.npy')]
                intensity_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'intensity', '*.npy')))
                self.intensity_list = [ele for ele in intensity_list if not ele.endswith('_downsampled.npy')]
                elongation_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'elongation', '*.npy')))
                self.elongation_list = [ele for ele in elongation_list if not ele.endswith('_downsampled.npy')]
                self.occ_file_idx_list = [i for i in range(len(self.occ_list))]
        else:
            self.occ_file_idx_list = None

        # TODO: handle the final index
        # ipdb.set_trace() # check whether the len of different list is the same, the order of different items in the list are the same
        if self.gd_mae_file_idx_list is not None:
            self.index_list = self.gd_mae_file_idx_list
        elif self.bev_feat_file_idx_list is not None:
            self.index_list = self.bev_feat_file_idx_list
        else:
            self.index_list = self.occ_file_idx_list

        # accuracy annotation (the per-frame per-branch mAP)
        self.load_acc = load_acc
        self.acc_path = acc_path
        if self.load_acc:
            if self.acc_path is None:
                path = os.path.join(data_folder, 'accuracy.npy')
            else:
                path = self.acc_path
            acc = np.load(path) # (#samples, #branches)
            assert acc.shape[0] == len(self.index_list)
            self.acc = torch.from_numpy(acc.astype(np.float32))
        else:
            self.acc = None
            
        # load latency (the per-frame per branch latency)
        self.load_latency = load_latency
        self.latency_path = latency_path
        if self.load_latency:
            if self.latency_path is None:
                path = os.path.join(data_folder, 'latency.npy')
            else:
                path = self.latency_path
            latency = np.load(path) # (#samples, #branches)
            assert latency.shape[0] == len(self.index_list)
            self.latency = torch.from_numpy(latency.astype(np.float32))
            self.avg_latency = torch.mean(self.latency, dim=0)
        else:
            self.latency = None

        # load the loss
        self.load_loss = load_loss
        self.loss_path = loss_path
        if self.load_loss:
            if self.loss_path is None:
                path = os.path.join(data_folder, 'loss.npy')
            else:
                path = self.loss_path            
            loss_val = np.load(path) # (#samples, #branches)
            assert loss_val.shape[0] == len(self.index_list)
            self.loss_val = torch.from_numpy(loss_val.astype(np.float32))
        else:
            self.loss_val = None

        # load sequence range
        seq_content = json.load(open(os.path.join(data_folder, 'sequences_first_occurrence_test.json'))) # (#samples, #branches)
        self.seq_range = {}
        start = 0
        for i, key in enumerate(seq_content):
            if i == 0:
                continue
            elif i == len(seq_content.keys()) -1:
                prev_range = [start, seq_content[key]-1]
                self.seq_range[i-1] = prev_range
                start = seq_content[key]
                curr_range = [start, len(self.index_list)-1]
                self.seq_range[i] = curr_range
            else:
                prev_range = [start, seq_content[key]-1]
                self.seq_range[i-1] = prev_range
                start = seq_content[key]

    def __len__(self):
        #return 200
        return len(self.seq_range.keys())

    def __getitem__(self, idx):
        # the idx here is the sequence index
        # get the range
        curr_seq_range = self.seq_range[idx]
        
        # load 10 sample
        data_list = []
        acc_list = []
        latency_list = []
        average_latency_list = []
        
        for i in range(curr_seq_range[0], curr_seq_range[1]+1):
            # add the new result
            curr_idx = i
            temp_data = self.get_one_data_point(curr_idx)
            data_list.append(temp_data)
            
            if self.load_acc:
                acc_list.append(self.acc[curr_idx].unsqueeze(dim=0))
            if self.load_latency:
                average_latency_list.append(self.avg_latency.unsqueeze(dim=0))
                latency_list.append(self.latency[curr_idx].unsqueeze(dim=0))


        # handle the concatenation of the feature
        if len(data_list[0]) == 1: # if there is only one features
            feature = [torch.cat([ele[0] for ele in data_list], dim=0)]
        else: # if there is multiple features
            # ipdb.set_trace() # check the feature
            num_of_feat = len(data_list[0])
            feature = []
            for i in range(num_of_feat):
                curr_feat_set = [ele[i] for ele in data_list]
                curr_feat_set = torch.cat(curr_feat_set, dim=0)
                feature.append(curr_feat_set)
            # ipdb.set_trace() # check the feature

        # the acc list should be the mAP or reward
        all_infos = {
            'feature': feature,
        }
        if self.load_latency:
            all_infos['latency_value'] = torch.cat(latency_list, dim=0)
            all_infos['avg_latency_value'] = torch.cat(average_latency_list, dim=0)
        
        if self.load_acc:
            all_infos['acc'] = torch.cat(acc_list, dim=0)
        
        return all_infos


class WaymoWindowRLContentionTrain(WaymoWindowRL):
    '''
        This Dataset aims to handle the contention level setting:
        1. It uses a fixed number as latency SLO
        2. It will randomly sample contention level from: (0, 20, 50, 90)
        3. It will handle the interval scheduling setting
        
        Modification
        1. rewrite the __init__
        2. rewrite the __getitem__??
    '''
    def __init__(
        self,
        root,               # dataset root folder
        split='val',        # split name (val | test)
        # occupancy grid config
        size=(300, 300),    # input spatial size (X, Y)
        use_occ=False,       # whether to use occupancy grid (X, Y, C)
        use_intensity=False, # whether to use intensity map (X, Y)
        use_elongation=False,# whether to use elongation map (X, Y)
        load_downsampled=False,
        # feature choice
        use_bev_feat=False,
        use_gd_mae_feat=False,
        gd_feat_path=None,
        # config of the interval scheduling
        max_scheduler_call=5,
        min_scheduler_call=2,
        scheduler_call_interval=10,        
        # for loading the profiling data
        wind_size=10,
        random_sample_len=False, # random sample len for augmentation
        load_acc=True,
        acc_path=None,
        load_latency=False,
        latency_path=None,
        load_loss=False,
        loss_path=None,
        # reward prepare
        return_reward=True,
        reward_type='acc',
        normalize_acc=False,
        soft_adjust=False,
        rescale_reward=True,
        # latency value prepare
        latency_adjusted_reward=False,
        return_latency_value=True,
        use_mean_latency_value=False, # return the mean of the lantency value
        # return latency threshold (randomly generated)
        latency_threshold=200,
        latency_per_wind=False,
        return_latency_thresh=True,
        latency_threshold_type='fixed',
        # for online mAP
        return_idx=False,
        # for some special experiments
        random_feat=False,
        # for contention
        contention_level=None,
        use_contention_level=False,
        # for wind setting
        use_wind_mean_map=False,
        use_wind_mean_latency=False,
        
    ):
        print('in WaymoWindowRLContentionTrain init')
        '''
            This dataset aims to prepare the data for training a scheduler which will be call every
            a few time step.
        '''
        ################################ handle the hyper #############################################
        self.root = root
        self.split = split
        # occupancy grid feature config
        self.size = size
        if not use_bev_feat and not use_gd_mae_feat: # if using occupancy grid, make sure we load at least one data
            assert use_occ or use_intensity or use_elongation
        # for occ feature
        self.use_occ = use_occ                  
        self.use_intensity = use_intensity
        self.use_elongation = use_elongation
        self.load_downsampled = load_downsampled
        # for bev feature
        self.use_bev_feat = use_bev_feat
        # for gd-mae feature
        self.use_gd_mae_feat = use_gd_mae_feat
        self.gd_feat_path = gd_feat_path        
        
        # config of the interval scheduling
        self.max_scheduler_call = max_scheduler_call # include the first idx
        self.min_scheduler_call = min_scheduler_call # include the first idx
        self.scheduler_call_interval = scheduler_call_interval
        
        # handle the profiling data
        self.wind_size = wind_size
        self.use_wind_mean_map = use_wind_mean_map
        self.use_wind_mean_latency = use_wind_mean_latency
        
        self.random_sample_len = random_sample_len
        self.load_acc = load_acc
        self.acc_path = acc_path
        self.load_latency = load_latency
        self.latency_path = latency_path
        self.load_loss = load_loss
        self.loss_path = loss_path
        
        # for reward
        self.return_reward = return_reward
        self.reward_type = reward_type
        self.normalize_acc = normalize_acc
        self.soft_adjust = soft_adjust # if soft_adjust is False then we mask the value to 0, other wise we divide the value by half
        self.rescale_reward = rescale_reward
        
        # for latency input
        self.latency_adjusted_reward = latency_adjusted_reward
        self.return_latency_value = return_latency_value
        self.use_mean_latency_value = use_mean_latency_value        
        
        # for latency threshold
        self.latency_threshold = latency_threshold
        self.latency_per_wind = latency_per_wind
        self.return_latency_thresh = return_latency_thresh
        self.latency_threshold_type = latency_threshold_type # if 'random' then random select a value , if it is 'continuous' then select a value from the interval

        # for online eval mAP
        self.return_idx = return_idx        
        
        # for some special experiments
        self.random_feat = random_feat    
        
        # for the contention level
        self.contention_level = contention_level
        self.use_contention_level = use_contention_level
        
        # some assertion
        if use_contention_level:
            assert isinstance(self.latency_path, list)
            assert len(contention_level) == len(self.latency_path)        
        if self.latency_threshold_type == 'continuous':
            assert len(self.latency_threshold) == 2 # for continuous only give the lower bound and the higher bound
        
        ################################ load the feature #############################################
        # data list
        data_folder = os.path.join(root, split)
        
        # load the feature list or the occ grid datalist
        if self.use_gd_mae_feat:
            gd_mae_file_name_list = sorted(glob.glob(os.path.join(self.gd_feat_path, '*_downsampled.pt'))) # 28351.pt
            intensity_list = None
            elongation_list = None 
            # from_idx_to_gd_mae_file = {ele.split('/')[-1].rstrip('_downsampled.pt'):ele for ele in gd_mae_file_name_list}
            self.gd_mae_file_name_list = gd_mae_file_name_list
            self.gd_mae_file_idx_list = [i for i in range(len(gd_mae_file_name_list))]
        else:
            self.gd_mae_file_idx_list = None
        
        if self.use_bev_feat:
            bev_feat_file_name_list = sorted(glob.glob(os.path.join(data_folder, '*.pt'))) # 28351.pt
            intensity_list = None
            elongation_list = None 
            self.bev_feat_file_name_list = bev_feat_file_name_list
            self.bev_feat_file_idx_list = [i for i in range(len(bev_feat_file_name_list))]
        else:
            self.bev_feat_file_idx_list = None
        
        # ipdb.set_trace()
        if self.use_occ: # use occ feature
            if self.load_downsampled:
                self.occ_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'occ', '*_downsampled_bitpacked.npy')))
                self.intensity_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'intensity', '*_downsampled.npy')))
                self.elongation_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'elongation', '*_downsampled.npy')))        
                self.occ_file_idx_list = [i for i in range(len(self.occ_list))]
            else:
                occ_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'occ', '*.npy')))
                self.occ_list = [ele for ele in occ_list if not ele.endswith('_downsampled_bitpacked.npy')]
                intensity_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'intensity', '*.npy')))
                self.intensity_list = [ele for ele in intensity_list if not ele.endswith('_downsampled.npy')]
                elongation_list = sorted(glob.glob(os.path.join(data_folder, 'occupancy_grid', 'elongation', '*.npy')))
                self.elongation_list = [ele for ele in elongation_list if not ele.endswith('_downsampled.npy')]
                self.occ_file_idx_list = [i for i in range(len(self.occ_list))]
        else:
            self.occ_file_idx_list = None

        # handle the final index
        # ipdb.set_trace() # check whether the len of different list is the same, the order of different items in the list are the same
        if self.gd_mae_file_idx_list is not None:
            self.index_list = self.gd_mae_file_idx_list
        elif self.bev_feat_file_idx_list is not None:
            self.index_list = self.bev_feat_file_idx_list
        else:
            self.index_list = self.occ_file_idx_list

        ################################ load the profiled acc (the per-frame per-branch mAP) #############################################
        if self.load_acc:
            if self.acc_path is None:
                path = os.path.join(data_folder, 'accuracy.npy')
            else:
                path = self.acc_path
            acc = np.load(path) # (#samples, #branches)
            assert acc.shape[0] == len(self.index_list)
            self.acc = torch.from_numpy(acc.astype(np.float32))
        else:
            self.acc = None
        
        ################################ load the profiled latency (the per-frame per branch latency) ################################ 
        # ipdb.set_trace()
        if self.load_latency:
            if self.latency_path is None:
                path = os.path.join(data_folder, 'latency.npy')
            else:
                path = self.latency_path
            if isinstance(path, list):
                lantency_list = []
                for curr_path in path:
                    latency = np.load(curr_path) # (#samples, #branches)
                    assert latency.shape[0] == len(self.index_list)
                    lantency_list.append(torch.from_numpy(latency.astype(np.float32)).unsqueeze(dim=0))
                self.latency = torch.cat(lantency_list, dim=0)
                self.mean_latency = torch.mean(self.latency, dim=1) # mean latency of a branch over all the frames
            else:
                latency = np.load(path) # (#samples, #branches)
                assert latency.shape[0] == len(self.index_list)
                self.latency = torch.from_numpy(latency.astype(np.float32)) # torch.Size([4, 31895, 55])
                self.mean_latency = torch.mean(self.latency, dim=0) # torch.Size([4, 55])
        else:
            self.latency = None
        
        ################################  load the profiled loss ################################ 
        if self.load_loss:
            if self.loss_path is None:
                path = os.path.join(data_folder, 'loss.npy')
            else:
                path = self.loss_path            
            loss_val = np.load(path) # (#samples, #branches)
            assert loss_val.shape[0] == len(self.index_list)
            self.loss_val = torch.from_numpy(loss_val.astype(np.float32))
        else:
            self.loss_val = None
        
        ################################  load sequence range ################################ 
        seq_content = json.load(open(os.path.join(data_folder, 'sequences_first_occurrence.json'))) # (#samples, #branches)
        self.seq_range = {}
        start = 0
        for i, key in enumerate(seq_content):
            if i == 0:
                continue
            elif i == len(seq_content.keys()) -1:
                prev_range = [start, seq_content[key]-1]
                self.seq_range[i-1] = prev_range
                start = seq_content[key]
                curr_range = [start, 31894]
                self.seq_range[i] = curr_range
            else:
                prev_range = [start, seq_content[key]-1]
                self.seq_range[i-1] = prev_range
                start = seq_content[key]
    
    def load_reward(self, idx, latency_thresh=None, interval=None):
        '''
            Load the acc or the loss as rewards of a timestamp, 
            and use latency threshold (requirement) to adjust the rewards.
        '''
        
        if self.reward_type == 'acc': # use acc as reward
            assert self.acc is not None
            # ipdb.set_trace() # check the average map value
            if self.use_wind_mean_map and interval is not None:
                reward = torch.mean(self.acc[idx:idx+interval], dim=0)
            else:
                reward = self.acc[idx]
            # normalize the target (test - test.min(1, keepdim=True)[0]) / (test.max(1, keepdim=True)[0] - test.min(1, keepdim=True)[0])
            # if self.normalize_acc:
            if self.rescale_reward:
                reward = 2 * torch.exp((reward - reward.max()))
        elif self.reward_type == 'loss': # use loss as rewards
            assert self.loss_val is not None
            reward = self.loss_val[idx]
            # define the reward exp(-(ele-minval)) 
            #print('reward before:', reward)
            # the reward will between 0 to 2
            if self.rescale_reward:
                reward = 2 * torch.exp(-(reward - reward.min()))
            #print('reward after:', reward)
        else:
            raise NotImplementedError
    
        if self.latency_adjusted_reward: # ensure the adjustment about -2 to 0?
            assert latency_thresh is not None
            assert self.latency is not None
            curr_all_latency = self.latency[idx]
            reward_mask = curr_all_latency > latency_thresh
            #print('before mask:', reward)
            if self.soft_adjust:
                reward[reward_mask] /= 2
            else:
                reward[reward_mask] = 0
            #print('after mask:', reward)

        return reward.unsqueeze(dim=0)
    
    def load_latency(self, idx):
        '''
            Load the profiling latency value for on time stamps.
        '''        
        
        latency = self.latency[idx]
        return latency.unsqueeze(dim=0)
    
    def __getitem__(self, idx):
        '''
        
            Since in this setting, the scheduler is called every couple timestamps:
            
            1. determine the scheduler called index
            2. get the feature on this index
            3. get the lantency and the acc during this call        
        '''
        
        # find which sequence current index in
        range_idx = math.floor(idx / 200)
        curr_seq_range = self.seq_range[range_idx]
        # ensure the range index
        while idx > curr_seq_range[-1]:
            range_idx += 1
            curr_seq_range = self.seq_range[range_idx]
        assert idx >= curr_seq_range[0] and idx <= curr_seq_range[1]
        
        # make sure index to the end of the sequence is (min_scheduler_call - 1) * scheduler_call_interval + 2 (we can take the last index)
        if idx + (self.min_scheduler_call - 1) * self.scheduler_call_interval + 1 >= curr_seq_range[1]:
            idx = curr_seq_range[1] - ((self.min_scheduler_call - 1) * self.scheduler_call_interval + 1)
            called_idx = [idx, idx + self.scheduler_call_interval]
            #called_sequence_range = [idx, curr_seq_range[1] + 1] # not include the end (we should not select the end)
            train_sequence_end = curr_seq_range[1] + 1 # not include the end (we should not select the end)
            
        else: # if current index to the end of the sequence is more than 2 call, then random sample
            distance_to_end = curr_seq_range[1] - idx
            num_of_call_to_end = math.ceil(distance_to_end / self.scheduler_call_interval)
            max_call = num_of_call_to_end if num_of_call_to_end <= self.max_scheduler_call else self.max_scheduler_call 
            # random sample a call number
            random_call_num = random.randint(self.min_scheduler_call, max_call)
            # get the call index and the sequence range
            called_idx = [self.scheduler_call_interval*i + idx for i in range(random_call_num)]
            # determine the end with the end of the sequence
            # called_sequence_range = [idx, min(curr_seq_range[1], called_idx[-1] + self.scheduler_call_interval - 1) + 1] # not include the end (we should not select the end)
            train_sequence_end = min(curr_seq_range[1], called_idx[-1] + self.scheduler_call_interval - 1) + 1
            
        # load feature, rewards, latency on the called timestamps
        # load 10 sample
        data_list = []
        reward_list = []
        latency_thresh_list = []
        latency_value_list = []
        contention_value_list = []
        curr_lat_thresh = None
        curr_contention_value = None
        
        for curr_idx in called_idx:
            # add the new result
            #curr_idx = idx + i
            temp_data = self.get_one_data_point(curr_idx)
            data_list.append(temp_data)
            curr_end_idx = min(curr_seq_range[1], curr_idx + self.scheduler_call_interval - 1) + 1
            
            # prepare latency threshold
            if self.return_latency_thresh:
                if self.latency_per_wind and curr_lat_thresh is not None:# the threshold number is continuous and but keep for the same for the whole window
                    curr_lat_thresh = curr_lat_thresh
                else: # if the latency threshold for each time stamp, or it is the first time stamp of latency_per_wind
                    if self.latency_threshold_type == 'random':
                        curr_lat_thresh = self.latency_threshold[random.randint(0, len(self.latency_threshold)-1)]
                    elif self.latency_threshold_type == 'continuous':
                        assert len(self.latency_threshold) == 2
                        high = self.latency_threshold[1] - 1
                        low = self.latency_threshold[0]
                        curr_lat_thresh = random.randint(low, high)
                    elif self.latency_threshold_type == 'fixed':
                        curr_lat_thresh = self.latency_threshold
                        # ipdb.set_trace() # chec the latency threshold
                    else:
                        raise NotImplementedError
                latency_thresh_list.append(torch.tensor([curr_lat_thresh]))     
            else:
                curr_lat_thresh = None
                
            # determin the contention level 
            if self.use_contention_level:
                if curr_contention_value is None: # keep the same contention lebel for the whole window
                    # ipdb.set_trace() # check the contention level
                    # determin the contention idx
                    contention_index = random.randint(0, len(self.contention_level) - 1)
                    curr_contention_value = self.contention_level[contention_index]
                contention_value_list.append(curr_contention_value)
                
            # use contention level to choose the latency    
            if self.return_latency_value:
                if self.use_contention_level:
                    if self.use_mean_latency_value:
                        latency_value_list.append(self.mean_latency[contention_index].unsqueeze(dim=0)) 
                    else:
                        # ipdb.set_trace() # check the lantency loading
                        if self.use_wind_mean_latency and self.wind_size is not None:
                            latency_value = torch.mean(self.latency[contention_index][curr_idx:curr_idx + self.wind_size].unsqueeze(dim=0), dim=1)
                            latency_value_list.append(latency_value) 
                        else:
                            latency_value_list.append(self.latency[contention_index][curr_idx].unsqueeze(dim=0)) 
                else:
                    if self.use_mean_latency_value:
                        latency_value_list.append(self.mean_latency.unsqueeze(dim=0)) 
                    else:
                        latency_value_list.append(self.latency[curr_idx].unsqueeze(dim=0))                     
            
            # load the reward
            temp_reward = self.load_reward(curr_idx, curr_lat_thresh, interval=self.wind_size)
            reward_list.append(temp_reward)

        # ipdb.set_trace() # final check of the latency and the contention
        
        # handle the concatenation of the feature
        if len(data_list[0]) == 1: # if there is only one features
            feature = [torch.cat([ele[0] for ele in data_list], dim=0)]
        else: # if there is multiple features
            # ipdb.set_trace() # check the feature
            num_of_feat = len(data_list[0])
            feature = []
            for i in range(num_of_feat):
                curr_feat_set = [ele[i] for ele in data_list]
                curr_feat_set = torch.cat(curr_feat_set, dim=0)
                feature.append(curr_feat_set)
            # ipdb.set_trace() # check the feature

        # the acc list should be the mAP or reward
        all_infos = {
            'feature': feature,
            'reward': torch.cat(reward_list, dim=0) if len(reward_list) > 1 else reward_list[0].squeeze(dim=0),
            'contention_value_list': torch.tensor(contention_value_list),
        }
        if self.return_latency_thresh:
            all_infos['latency_thres'] = torch.cat(latency_thresh_list, dim=0)
        
        if self.return_idx:
            called_idx.append(train_sequence_end)
            all_infos['idx'] = torch.tensor(called_idx)
        
        if self.return_latency_value:
            all_infos['latency_value'] = torch.cat(latency_value_list, dim=0)
        
        return all_infos


def make_dataset(opt, is_training=True):
    opt = deepcopy(opt)
    name = opt.pop('name')
    if name == 'waymoWindow_rl':
        return WaymoWindowRL(**opt)
    elif name == 'waymoWindow_rl_interval':
        return WaymoWindowRLInterval(**opt)
    elif name == 'waymoWindow_rl_eval':
        return WaymoWindowRLEval(**opt)
    elif name == 'waymoWindow_contention_train':
        return WaymoWindowRLContentionTrain(**opt)
    else:
        raise NotImplementedError


def make_dataloader(
    dataset,            # dataset
    generator,          # random number generator that controls worker seed
    batch_size,         # local batch size
    num_workers,        # local number of workers
    is_training=True,   # whether is in training
):

    shuffle = is_training
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
        shuffle=shuffle,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader