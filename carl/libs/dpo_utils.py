import math
import ipdb
import random
import torch
import torch.nn.functional as F
import torch.distributions as distributions
from .train_utils import AverageMeter, get_pred_with_det_res, convert_all_predictions


# set up the content-agnostic method lookup table for the 
from_thres_to_branches = {50: 0, 100: 8, 150: 21, 200:19, 250:32, 300: 31, 500: 30}

from_thres_to_pareto_frontier_branches = {50: 0, 100: 7, 150: 13, 200:11, 250:18, 300: 17, 500: 16}

from_thres_to_79branches = {50: 3, 100: 8, 150: 39, 200:34, 250:28, 300: 68, 500: 23}

from_thres_to_82branches = {50: 3, 100: 8, 150: 42, 200:37, 250:31, 300: 71, 500: 26}


def prepare_dpo_pair(acc, 
                     latency, 
                     latency_thresh, 
                     refer_logit=None, 
                     branch_filter_type=None,
                     latency_adjustment=None, 
                     lut_neg_action=False,
                     chosen_action_only=False):
    '''
        This function prepare the positve-negative pair for the DPO training.
        For current implementation, we use the greedy results as the postive pair.
        We use the random result for the negative pair.
    '''
    
    # prepare the chosen actions (at current stage, it would be the greedy search results)
    assert latency is not None
    assert latency_thresh is not None
    assert len(latency_thresh[0]) == 1 or latency_thresh[0][0] == latency_thresh[0][1] # assert the latency threshold is the same for the whole windows
    curr_latency_threshold = latency_thresh[0][0]
    # Do futher adjustment to the latency
    if latency_adjustment is not None:
        # ipdb.set_trace()
        curr_latency_threshold += latency_adjustment
         
    mask = latency > curr_latency_threshold
    # mask = mask.squeeze(dim=0)  # from (B, L, cls_dim) to (L, cls_dim)
    map_map = acc.clone()
    map_map[mask] = float('-inf')
    chosen_actions = torch.argmax(map_map, dim=-1) # (1, #Wind)
    if chosen_action_only:
        return chosen_actions
    
    # prepare the rejected actions (at current stage, it would be the random result)
    # rejected_actions = torch.randint(low=0, high=map_map.shape[-1], size=chosen_actions.shape).to(device=chosen_actions.device)
    # create the negative sample by sampling from the reference logit
    # ipdb.set_trace() # check the distribution
    if not lut_neg_action:
        action_prob = F.softmax(refer_logit, dim=-1)
        dist = distributions.Categorical(action_prob)
        rejected_actions = dist.sample() 
    else:
        assert latency_thresh[0][0] == latency_thresh[0][1] # assert the latency threshold is the same for the whole windows
        curr_latency_threshold = latency_thresh[0][0]
        # get the branch choose by the content agnositc method 
        target_SLO = int(math.ceil(curr_latency_threshold / 50) * 50) # calculate the target_SLO
        if target_SLO > 300: # handle the edge case
            target_SLO = 500    
        
        if branch_filter_type == 'default':
            content_agnostic_branch = from_thres_to_branches[target_SLO]
        elif branch_filter_type == 'pf_only':
            # print('in pf only', len(all_detection_res))
            content_agnostic_branch = from_thres_to_pareto_frontier_branches[target_SLO]
        elif branch_filter_type == 'enlarge79':
            content_agnostic_branch = from_thres_to_79branches[target_SLO]
        elif branch_filter_type == 'enlarge82':
            content_agnostic_branch = from_thres_to_82branches[target_SLO]                
        else:
            raise NotImplementedError
        # ipdb.set_trace() #check the latency, check the branch, check the generated rejected_actions
        rejected_actions = torch.full(chosen_actions.shape, fill_value=content_agnostic_branch).to(device=chosen_actions.device)
    
    return chosen_actions, rejected_actions


def select_and_calculate_logps(logits, labels):
    '''
        select and calculate the policy.
        logits: torch.Size([1, 5, 82])
        labels: torch.Size([1, 5])
    ''' 
    # ipdb.set_trace() # check whether the selection is correct
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    # return (per_token_logps * loss_mask).sum(-1)
    return per_token_logps.sum(-1)


def calculate_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                       reference_chosen_logps, reference_rejected_logps, 
                       beta=0.1, label_smoothing=0.0):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )

    chosen_rewards = (
        beta
        * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()
    )
    rejected_rewards = (
        beta
        * (
            policy_rejected_logps
            - reference_rejected_logps
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def forward_action_with_det_res(model, actions, features, latency_thres, all_detection_res, all_idx, contention_level=None):
    '''
        This function feed the model with the given action, feature, detection result and latency requirement step by step.
        and get the log prob
        Return the action logits over all the step
        features: list[tensor], tensor [B, T, D] or [B, T, H, W, C]
        
    '''
    # ipdb.set_trace() # check teh contention level is used
    timestamp = len(features[0][0])
    # prepare the empty prediction for the step 0
    prepared_det_info = (None, None, None)
    
    all_logits = []
    # ipdb.set_trace() # check the action state
    action_state = [(None, None)] * len(model.layers)
    
    # loop over all the steps
    # for loop call the model over every time step, get all actions
    for curr_timestamp in range(timestamp):
        curr_feat = [ele[:, curr_timestamp].unsqueeze(dim=1) for ele in features]
        curr_action = actions[:, curr_timestamp]
        
        # ipdb.set_trace() # check the contention in the select action
        if contention_level is not None: # if contetnion level is not None, then use the contention level
            curr_lat_thres = contention_level[:, curr_timestamp].unsqueeze(dim=1)
        else:
            curr_lat_thres = latency_thres[:, curr_timestamp].unsqueeze(dim=1)
        # send in the feat, lantency, prediction
        prob_logit, action_state = \
            model.step(curr_feat, curr_lat_thres, prepared_det_info, action_state) # prob_logit[0].shape
        all_logits.append(prob_logit)
        
        # get the time index
        curr_idx = all_idx[curr_timestamp] # all_idx (window_len, )
        
        # prepare next detection result
        current_predictions = all_detection_res[curr_action[0].item()][curr_idx.item()]
        bbox_coordinates, bbox_categories, bbox_confidences = convert_all_predictions(current_predictions)
        prepared_det_info = (bbox_coordinates, bbox_categories, bbox_confidences)
    
    # concat and return the predicted distribution
    all_logits = torch.cat(all_logits, dim=1)
    
    # ipdb.set_trace() # check the logit
    return all_logits


def dpo_train_one_epoch(dataloader, model, ref_model, 
                        optimizer, device, opt,
                        all_detection_res, waymo_gt, class_names,
                        losses_tracker=None,
                        tb_writer=None, 
                        curr_iter=0,
                        print_freq=100,
                        lut_neg_action=False):
    
    # prepare hyper    
    use_latency_thresh = opt['train']['use_latency_thresh'] if 'use_latency_thresh' in opt['train'] else False
    use_contention_level = opt['train']['use_contention_level'] if 'use_latency_thresh' in opt['train'] else False
    use_det_res = opt['train']['use_det_res'] if 'use_det_res' in opt['train'] else False
    interval_schedule = opt['train']['interval_schedule'] if 'interval_schedule' in opt['train'] else False
    latency_adjustment = opt['train']['latency_adjustment'] if 'latency_adjustment' in opt['train'] else None
    branch_filter_type = opt['train']['branch_filter_type'] if 'branch_filter_type' in opt['train'] else 'default'
    accumulate_grad_step = opt['train']['accumulate_grad_step'] if 'accumulate_grad_step' in opt['train'] else None 
    random_positive_action = opt['train']['random_positive_action'] if 'random_positive_action' in opt['train'] else None 
    
    # put total iteration in the opt
    num_epoches = opt['train']['num_epoches'] if 'num_epoches' in opt['train'] else 10 
    opt['total_iters'] = num_epoches * len(dataloader.dataset)
    
    model.train()
    curr_iteration = curr_iter
    # each data tuple here is a episode
    for count_i, data_tuple in enumerate(dataloader):
        
        # update the curr_iteration in the opt
        opt['curr_iters'] = curr_iteration
        
        # get the training data 
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

        # inference the reference model first to get the sequence
        if use_det_res:
            assert 'idx' in data_tuple
            curr_start_idx = data_tuple['idx'] 
            # ipdb.set_trace() # check teh idx
            if interval_schedule:
                assert len(curr_start_idx[0]) == len(features[0][0]) + 1
                all_idx = curr_start_idx[0, :-1] # Shape: (len(actions[0]), )
            else:
                assert len(curr_start_idx) == 1
                all_idx = torch.tensor([curr_start_idx[0] + i for i in range(opt['train']['data']['wind_size'])])
            
            
            # randomly decide whether we use gt as the positve pair 
            random_number = random.random()
            
            if random_positive_action and random_number > 0.5:
                # in this case we will random select positive sample from: 1. greedy action. 2. the reference model result but positive one.
                chosen_actions, rejected_actions, policy_chosen_logit, \
                    policy_reject_logit, reference_chosen_logit, refer_rejected_logit = \
                        use_ref_as_positve(model, ref_model, features, latency_thres, all_detection_res, all_idx, contention_level,
                        raw_rewards, latency_value)
            else:
                # get the reference logit and rejected actions
                chosen_actions, rejected_actions, policy_chosen_logit, \
                    policy_reject_logit, reference_chosen_logit, refer_rejected_logit = \
                        use_gt_as_positve(model, ref_model, features, latency_thres, all_detection_res, all_idx, contention_level,
                        raw_rewards, latency_value)

        else:
            # TODO need to fixe the contention setting
            raise NotImplementedError
            refer_logit = ref_model(features, latency_thres).detach()
            # prepare the positive and negative pair
            chosen_actions, rejected_actions = prepare_dpo_pair(raw_rewards, 
                                                                latency_value, 
                                                                latency_thres, 
                                                                refer_logit, 
                                                                branch_filter_type,
                                                                latency_adjustment=latency_adjustment,
                                                                lut_neg_action=lut_neg_action)
            # forward and get the logit of the policy model
            logit = model(features, latency_thres)
        
        # get the policy and reference logps
        policy_chosen_logps = select_and_calculate_logps(policy_chosen_logit, chosen_actions)
        policy_rejected_logps = select_and_calculate_logps(policy_reject_logit, rejected_actions)
        reference_chosen_logps = select_and_calculate_logps(reference_chosen_logit, chosen_actions)
        reference_rejected_logps = select_and_calculate_logps(refer_rejected_logit, rejected_actions)

        # calculate the dpo loss
        dpo_loss, chosen_rewards, rejected_rewards = calculate_dpo_loss(policy_chosen_logps,
                                      policy_rejected_logps,
                                      reference_chosen_logps,
                                      reference_rejected_logps)
        
        # back propagates the loss and update the params
        # ipdb.set_trace() # check the shape of the loss
        dpo_loss = dpo_loss.mean()
        
        # scale the loss if it accumulates
        if accumulate_grad_step is not None: 
            # ipdb.set_trace() # check the loss scaling
            dpo_loss = dpo_loss / accumulate_grad_step  # Scale loss to normalize gradients
        
        dpo_loss.backward()


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
        
        # store the loss value and the reward values
        all_loss = {'chosen_rewards': chosen_rewards.detach(), 
                    'rejected_rewards': rejected_rewards.detach(),
                    'final_loss': dpo_loss.detach()}

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
        
            print('curr_iteration:', 
                  curr_iteration, 'total loss:', 
                  all_loss['final_loss'],
                  'chosen_rewards', all_loss['chosen_rewards'], 
                  'rejected_rewards', all_loss['rejected_rewards'])
    
    return curr_iteration


def use_gt_as_positve(model, ref_model, features, latency_thres, all_detection_res, all_idx, contention_level,
                      raw_rewards, latency_value):
    # In this version, we use the greedy action as the postive 
    # and use the action from the reference model as the negative
    # we need following things:
    # 1. Get negative action from reference net, and the reference net negative action.
    # 2. Get positive action using greedy.
    # 3. forward the policy net postive action.
    # 4. forward the policy net negative action.
    # 5. forward the reference net postive action.
    
    # get the refer_rejected_logit, and rejected_actions
    refer_rejected_logit, rejected_actions = get_pred_with_det_res(ref_model, features, latency_thres, all_detection_res, all_idx, sample_action=True, contention_level=contention_level)
    refer_rejected_logit = refer_rejected_logit.detach()
    rejected_actions = rejected_actions.unsqueeze(dim=0).detach()
    
    # get the chosen action, which is the greedy action
    chosen_actions = prepare_dpo_pair(raw_rewards, 
                                        latency_value, 
                                        latency_thres,
                                        chosen_action_only=True)
    # forward the policy net with the chosen action, policy_chosen_logit
    # policy_chosen_logit, _ = get_pred_with_det_res(model, features, latency_thres, all_detection_res, all_idx, sample_action=True, contention_level=contention_level)
    policy_chosen_logit = forward_action_with_det_res(model, chosen_actions, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
    
    # forward the policy net with the rejected_actions action, get the policy_reject_logit
    policy_reject_logit = forward_action_with_det_res(model, rejected_actions, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
    
    # forward the reference net with the chosen action, get the reference_chosen_logit
    reference_chosen_logit = forward_action_with_det_res(ref_model, chosen_actions, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
    
    return chosen_actions, rejected_actions, policy_chosen_logit, policy_reject_logit, reference_chosen_logit, refer_rejected_logit


def use_ref_as_positve(model, ref_model, features, latency_thres, all_detection_res, all_idx, contention_level,
                      raw_rewards, latency_value):
    # In this version, we use the reference model as the postive and negative,
    # we sample two action sequence and determine the positive and negative action from them
    # we need following things:
    # 1. Get negative action from reference net, and the reference net negative action.
    # 2. Get positive action using greedy.
    # 3. forward the policy net postive action.
    # 4. forward the policy net negative action.
    # 5. forward the reference net postive action.
    
    # get the action1
    logit_1, actions_1 = get_pred_with_det_res(ref_model, features, latency_thres, all_detection_res, all_idx, sample_action=True, contention_level=contention_level)
    logit_1 = logit_1.detach()
    actions_1 = actions_1.unsqueeze(dim=0).detach()
    
    # get the action2
    logit_2, actions_2 = get_pred_with_det_res(ref_model, features, latency_thres, all_detection_res, all_idx, sample_action=True, contention_level=contention_level)
    logit_2 = logit_2.detach()
    actions_2 = actions_2.unsqueeze(dim=0).detach()
    
    # calculate some stat
    # ipdb.set_trace() # check the forward
    actions_1_acc = torch.gather(raw_rewards, dim=2, index=actions_1.unsqueeze(2))
    actions_1_mean_acc = torch.mean(actions_1_acc)
    actions_1_lat = torch.gather(latency_value, dim=2, index=actions_1.unsqueeze(2)).squeeze(dim=-1)
    actions_1_lat_violate_mask = actions_1_lat > latency_thres
    actions_1_overshot = (True in actions_1_lat_violate_mask)
    actions_1_max_overshot = torch.max(actions_1_lat - latency_thres)
    
    actions_2_acc = torch.gather(raw_rewards, dim=2, index=actions_2.unsqueeze(2))
    actions_2_mean_acc = torch.mean(actions_2_acc)
    actions_2_lat = torch.gather(latency_value, dim=2, index=actions_2.unsqueeze(2)).squeeze(dim=-1)
    actions_2_lat_violate_mask = actions_2_lat > latency_thres
    actions_2_overshot = (True in actions_2_lat_violate_mask)
    actions_2_max_overshot = torch.max(actions_2_lat - latency_thres)
    
    # determine which is the positive action which is the negative actions
    if actions_1_overshot and actions_2_overshot: # both sequence over shot the latency threshold 
        if actions_1_max_overshot < actions_2_max_overshot:
            chosen_actions = actions_1
            rejected_actions = actions_2
            reference_chosen_logit = logit_1
            refer_rejected_logit = logit_2
        else:
            chosen_actions = actions_2
            rejected_actions = actions_1
            reference_chosen_logit = logit_2
            refer_rejected_logit = logit_1
    elif actions_1_overshot and not actions_2_overshot: # if only one overshot the latency threshold
        chosen_actions = actions_2
        rejected_actions = actions_1
        reference_chosen_logit = logit_2
        refer_rejected_logit = logit_1
    elif actions_2_overshot and not actions_1_overshot: # if only one overshot the latency threshold
        chosen_actions = actions_1
        rejected_actions = actions_2
        reference_chosen_logit = logit_1
        refer_rejected_logit = logit_2
    else: # if none of them overshot the ratio, choose the sequence with the highest mean acc
        if actions_1_mean_acc > actions_2_mean_acc:
            chosen_actions = actions_1
            rejected_actions = actions_2
            reference_chosen_logit = logit_1
            refer_rejected_logit = logit_2
        else:
            chosen_actions = actions_2
            rejected_actions = actions_1
            reference_chosen_logit = logit_2
            refer_rejected_logit = logit_1

    # forward the policy net with the chosen action, policy_chosen_logit
    # policy_chosen_logit, _ = get_pred_with_det_res(model, features, latency_thres, all_detection_res, all_idx, sample_action=True, contention_level=contention_level)
    policy_chosen_logit = forward_action_with_det_res(model, chosen_actions, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
    
    # forward the policy net with the rejected_actions action, get the policy_reject_logit
    policy_reject_logit = forward_action_with_det_res(model, rejected_actions, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
    
    # # forward the reference net with the chosen action, get the reference_chosen_logit
    # reference_chosen_logit = forward_action_with_det_res(ref_model, chosen_actions, features, latency_thres, all_detection_res, all_idx, contention_level=contention_level)
    
    return chosen_actions, rejected_actions, policy_chosen_logit, policy_reject_logit, reference_chosen_logit, refer_rejected_logit