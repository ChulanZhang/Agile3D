import os
import random
import shutil
import numpy as np
import ipdb

import torch
import torch.nn.functional as F
import torch.distributions as distributions

class Logger:

    def __init__(self, filepath):

        self.filepath = filepath

    def write(self, log_str):
        print(log_str)
        with open(self.filepath, 'a') as f:
            print(log_str, file=f)


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.mean = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

    def item(self):
        return self.mean


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def fix_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: uncomment for CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ## NOTE: uncomment for pytorch >= 1.8
    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)
    return rng


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker.
    """
    seed = torch.initial_seed() % 2 ** 31
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def save_supervised_checkpoint(epoch_num, opt, model, optimizer):
    '''
        For saving the supervised model
    '''
    
    print("Checkpointing at [epoch " + str(epoch_num))
    model_dir = os.path.join(opt['_root'], 'models')
    state_dir = os.path.join(opt['_root'], 'states')
    model_ckpt = {
        'model': model.state_dict(),
    }
    state_ckpt = {
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
    }
    torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
    torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
    shutil.copyfile(
        os.path.join(model_dir, 'last.pth'),
        os.path.join(model_dir, (str(epoch_num) + ".pth"))
    )


def convert_all_predictions(current_predictions, top_k=50):
    '''
        current_predictions has 6 keys:
        'name': The categories. (381,) 'Pedestrian', 'Vehicle', 'Cyclist', 'Vehicle',
                                                'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Pedestrian'.
       
        'score': The confidence scores. (381,) array([0.94694316, 0.9373796 , 0.93066156, 0.9298923 , 0.90425116,
                                                0.896273  , 0.86850125, 0.86310667, 0.8409763 , 0.83814895,
                                                0.81271386, 0.80882525, 0.79979515, 0.793863  , 0.7911296 ,
                                                0.7910072 , 0.7285689 , 0.72679204,
        'boxes_lidar': current_predictions['boxes_lidar'].shape (381, 7)
        
        'pred_labels': (381,) array([1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1,
                                    1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
         1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'
        'frame_id', 'metadata': are not very important.
        
        return bbox_coordinates, bbox_categories, bbox_confidences in tensor format
    '''
    
    if len(current_predictions['pred_labels']) == 0:
        bbox_coordinates = torch.zeros(1, 7)
        pred_labels = torch.zeros(1,)
        bbox_confidences = torch.zeros(1,)
    
    elif len(current_predictions['pred_labels']) < top_k: # if the detection res is not enough, padded with the last one?
        bbox_coordinates = torch.tensor(current_predictions['boxes_lidar'])
        pred_labels = torch.tensor(current_predictions['pred_labels'])
        bbox_confidences = torch.tensor(current_predictions['score'])
    else:
        topk_idx = torch.topk(torch.tensor(current_predictions['score']), k=top_k)[1]
        bbox_coordinates = torch.tensor(current_predictions['boxes_lidar'])[topk_idx]
        pred_labels = torch.tensor(current_predictions['pred_labels'])[topk_idx]
        bbox_confidences = torch.tensor(current_predictions['score'])[topk_idx]
    
    bbox_coordinates = bbox_coordinates.unsqueeze(dim=0).unsqueeze(dim=0)
    pred_labels = pred_labels.unsqueeze(dim=0).unsqueeze(dim=0)
    bbox_confidences = bbox_confidences.unsqueeze(dim=0).unsqueeze(dim=0)
    
    return bbox_coordinates, pred_labels, bbox_confidences


def get_pred_with_det_res(model, features, latency_thres, all_detection_res, all_idx, sample_action=False, contention_level=None):
    '''
        This function feed the model with the feature, detection result and latency requirement step by step.
        Return the action logits over all the step
        features: list[tensor], tensor [B, T, D] or [B, T, H, W, C]
        
    '''
    # ipdb.set_trace() # check teh contention level is used
    timestamp = len(features[0][0])
    # prepare the empty prediction for the step 0
    prepared_det_info = (None, None, None)
    
    all_logits = []
    all_actions = []
    # import ipdb
    # ipdb.set_trace()
    action_state = [(None, None)] * len(model.layers)
    # loop over all the steps
    # for loop call the model over every time step, get all actions
    for curr_timestamp in range(timestamp):
        curr_feat = [ele[:, curr_timestamp].unsqueeze(dim=1) for ele in features]
        # ipdb.set_trace() # check the contention in the dpo forwards
        if contention_level is not None: # if contetnion level is not None, then use the contention level
            curr_lat_thres = contention_level[:, curr_timestamp].unsqueeze(dim=1)
        else:
            curr_lat_thres = latency_thres[:, curr_timestamp].unsqueeze(dim=1)
        # send in the feat, lantency, prediction
        prob_logit, action_state = \
            model.step(curr_feat, curr_lat_thres, prepared_det_info, action_state) # prob_logit[0].shape
        all_logits.append(prob_logit)
        
        # get the action
        # ipdb.set_trace() # logit and others
        curr_action_prob = F.softmax(prob_logit, dim=-1) # torch.Size([1, 1, 82])
        if sample_action:
            dist = distributions.Categorical(curr_action_prob)
            action = dist.sample() 
        else:
            action = torch.argmax(curr_action_prob, dim=-1) # torch.Size([1, 1])
        
        all_actions.append(action)
        curr_idx = all_idx[curr_timestamp] # all_idx (window_len, )

        # prepare next detection result
        current_predictions = all_detection_res[action[0,0].item()][curr_idx.item()]
        bbox_coordinates, bbox_categories, bbox_confidences = convert_all_predictions(current_predictions)
        prepared_det_info = (bbox_coordinates, bbox_categories, bbox_confidences)
    
    # concat and return the predicted distribution
    all_logits = torch.cat(all_logits, dim=1)
    all_actions = torch.tensor(all_actions).to(all_logits.device)
    
    # ipdb.set_trace() # check the logit
    return all_logits, all_actions


def get_pred_with_det_res_two_branches(model, features, latency_thres, all_detection_res, all_idx, sample_action=False, contention_level=None):
    '''
        This function feed the model with the feature, detection result and latency requirement step by step.
        Return the action logits over all the step
        features: list[tensor], tensor [B, T, D] or [B, T, H, W, C]
        
    '''
    # ipdb.set_trace() # check teh contention level is used
    timestamp = len(features[0][0])
    # prepare the empty prediction for the step 0
    prepared_det_info = (None, None, None)
    
    all_map_x = []
    all_lat_x = []
    all_logits = []
    all_actions = []
    # import ipdb
    # ipdb.set_trace()
    action_state = [(None, None)] * len(model.layers)
    # loop over all the steps
    # for loop call the model over every time step, get all actions
    for curr_timestamp in range(timestamp):
        curr_feat = [ele[:, curr_timestamp].unsqueeze(dim=1) for ele in features]
        # ipdb.set_trace() # check the contention in the dpo forwards
        if contention_level is not None: # if contetnion level is not None, then use the contention level
            curr_lat_thres = contention_level[:, curr_timestamp].unsqueeze(dim=1)
        else:
            curr_lat_thres = latency_thres[:, curr_timestamp].unsqueeze(dim=1)
        # send in the feat, lantency, prediction
        (map_x, lat_x), prob_logit, action_state = \
            model.step(curr_feat, curr_lat_thres, prepared_det_info, action_state) # prob_logit[0].shape
        all_logits.append(prob_logit)
        all_map_x.append(map_x)
        all_lat_x.append(lat_x)
        
        if sample_action:
            dist = distributions.Categorical(prob_logit)
            action = dist.sample() 
        else:
            action = torch.argmax(prob_logit, dim=-1) # torch.Size([1, 1])
        
        all_actions.append(action)
        curr_idx = all_idx[curr_timestamp] # all_idx (window_len, )

        # prepare next detection result
        current_predictions = all_detection_res[action[0,0].item()][curr_idx.item()]
        bbox_coordinates, bbox_categories, bbox_confidences = convert_all_predictions(current_predictions)
        prepared_det_info = (bbox_coordinates, bbox_categories, bbox_confidences)
    
    # concat and return the predicted distribution
    all_logits = torch.cat(all_logits, dim=1)
    all_map_x = torch.cat(all_map_x, dim=1)
    all_lat_x = torch.cat(all_lat_x, dim=1)
    all_actions = torch.tensor(all_actions).to(all_logits.device)
    
    # ipdb.set_trace() # check the logit
    return all_map_x, all_lat_x, all_logits, all_actions


def supervised_train_one_epoch(dataloader, model, optimizer, device, opt,
                               all_detection_res, waymo_gt, class_names,
                               losses_tracker=None,
                               tb_writer=None, 
                               curr_iter=0,
                               print_freq=100,):
            
    # prepare hyper
    use_latency_thresh = opt['train']['use_latency_thresh'] if 'use_latency_thresh' in opt['train'] else False
    use_contention_level = opt['train']['use_contention_level'] if 'use_latency_thresh' in opt['train'] else False
    loss_start_idx = opt['train']['loss_start_idx'] if 'loss_start_idx' in opt['train'] else 0
    use_det_res = opt['train']['use_det_res'] if 'use_det_res' in opt['train'] else False
    interval_schedule = opt['train']['interval_schedule'] if 'interval_schedule' in opt['train'] else False
    accumulate_grad_step = opt['train']['accumulate_grad_step'] if 'accumulate_grad_step' in opt['train'] else None 
    
    two_branches_model = opt['train']['two_branches_model'] if 'two_branches_model' in opt['train'] else None 

    # put total iteration in the opt
    num_epoches = opt['train']['num_epoches'] if 'num_epoches' in opt['train'] else 10 
    opt['total_iters'] = num_epoches * len(dataloader.dataset)
    
    model.train()
    curr_iteration = curr_iter
    # each data tuple here is a episode
    for count_i, data_tuple in enumerate(dataloader):
        # dict_keys(['feature', 'reward', 'contention_value_list', 'latency_thres', 'idx', 'latency_value'])
        # data_tuple['latency_value'] torch.Size([1, 10, 55])
        # data_tuple['contention_value_list'] torch.Size([1, 10])
        # data_tuple['latency_thres'] torch.Size([1, 10])     
         
        # update the curr_iteration in the opt
        opt['curr_iters'] = curr_iteration
        
        # get the training feature, the acc, the latency, latency threshod
        # ipdb.set_trace() # data_tuple
        features = [ele.to(device) for ele in data_tuple['feature']] # (1, #Wind, X, Y, Z)
        raw_rewards = data_tuple['reward'].to(device) # (1, #Wind, action) / # (1, #Wind * scheduler_call_interval, action) this should be per-frame mAP
        # unload the contention level
        if use_contention_level:
            assert 'contention_value_list' in data_tuple # if we need the index then the len should be 4
            contention_level = data_tuple['contention_value_list'].to(device) # (1, #Wind) 
        else:
            contention_level = None
        # unload the latency
        if use_latency_thresh:
            assert 'latency_thres' in data_tuple # if we need the index then the len should be 4
            latency_thres = data_tuple['latency_thres'].to(device) # (1, #Wind) 
        else:
            latency_thres = None
        latency_value = data_tuple['latency_value'].to(device) if 'latency_value' in data_tuple else None # (1, #Wind, action)

        # forward the model
        if use_det_res:
            curr_start_idx = data_tuple['idx'] 
            # ipdb.set_trace() # check teh idx
            if interval_schedule:
                assert len(curr_start_idx[0]) == len(features[0][0]) + 1
                all_idx = curr_start_idx[0, :-1] # Shape: (len(actions[0]), )
            else:
                assert len(curr_start_idx) == 1
                all_idx = torch.tensor([curr_start_idx[0] + i for i in range(opt['train']['data']['wind_size'])])
            if not two_branches_model:
                pred, _ = get_pred_with_det_res(model, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
            else:
                all_map_x, all_lat_x, pred, _ = get_pred_with_det_res_two_branches(model, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
        else:
            # TODO need to fixe the contention setting
            raise NotImplementedError
            pred = model(features, latency_thres)
        
        # ipdb.set_trace() # check the selected action.
        # calculate the loss
        assert latency_value is not None
        assert latency_thres is not None
        assert len(latency_thres[0]) == 1 or latency_thres[0][0] == latency_thres[0][1] # assert the latency threshold is the same for the whole windows
        if not two_branches_model:
            # mask the rewards and find the lables
            curr_latency_threshold = latency_thres[0][0]            
            mask = latency_value > curr_latency_threshold
            # mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)
            map_map = raw_rewards.clone()
            map_map[mask] = float('-inf')
            labels = torch.argmax(map_map, dim=-1) # (1, #Wind)        
            
            loss = F.cross_entropy(pred.view(-1, latency_value.shape[-1]), labels.view(-1))
            
            all_loss = {'final_loss': loss.detach(), }
            print('curr_iteration:', curr_iteration, 'total loss:', all_loss['final_loss'])
            
        else:
            # prepare the label for the map
            # find the branch within the SLO but with the highest performance
            curr_latency_threshold = latency_thres[0][0]            
            over_slo_mask = latency_value > curr_latency_threshold
            # mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)
            map_map = raw_rewards.clone()
            map_map[over_slo_mask] = float('-inf')
            value, _ = torch.topk(map_map, k=3, dim=-1) # (1, #Wind)    
            # ipdb.set_trace() # check the loading     
            
            # get the third best performance and create the mask
            the_third_top = value[:, :, -1].unsqueeze(dim=-1)
            above_top3_mask = raw_rewards.clone() > the_third_top
            
            # generate the map label for each categories (higher than top3 will be postive)
            map_label = above_top3_mask.to(torch.float32)
            
            # prepare the label for the latency 
            below_slo_mask = ~over_slo_mask
            # generate the slo label for each categories (lower then SLO will be positive)
            lat_label = below_slo_mask.to(torch.float32)
            
            # calculate the map loss
            map_loss = F.binary_cross_entropy_with_logits(all_map_x, map_label)
            
            # calculate the slo loss
            lat_loss = F.binary_cross_entropy_with_logits(all_lat_x, lat_label)
            loss = map_loss + lat_loss
            
            all_loss = {'final_loss': loss.detach(), 'map_loss:': map_loss.detach(), 'lat_loss:': lat_loss.detach()}
            print('curr_iteration:', curr_iteration, 'total loss:', all_loss['final_loss'], 
                  'map_loss:', map_loss.detach(), 'lat_loss:', lat_loss.detach())
        
        
        # scale the loss if it accumulates
        if accumulate_grad_step is not None: 
            # ipdb.set_trace() # check the loss scaling
            loss = loss / accumulate_grad_step  # Scale loss to normalize gradients
        
        # backprop and udpate the model
        loss.backward()

        if accumulate_grad_step is not None: 
            # ipdb.set_trace() # check the loss 
            if (count_i + 1) % accumulate_grad_step == 0 or (count_i + 1) == len(dataloader):
                max_norm = 1.0  # Maximum norm of gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            max_norm = 1.0  # Maximum norm of gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        max_norm = 1.0  # Maximum norm of gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        curr_iteration += 1
        # print the loss and update the tensorboard
        if (curr_iteration != 0) and (curr_iteration % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            # batch_time.update((time.time() - start) / print_freq)
            # start = time.time()

            # track all losses
            for key, value in all_loss.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value)

            # log to tensor board
            if tb_writer is not None:
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    curr_iteration
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    curr_iteration
                )        
        
            
    
    return curr_iteration