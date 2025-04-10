import argparse
import os
import shutil
import ipdb


import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from libs.train_utils import save_supervised_checkpoint
from libs.modeling_utils import ActorCriticWithLatency
from libs.dpo_utils import dpo_train_one_epoch

from libs import SSM, SSM_contention, SSM_contention_two_branches
from libs import load_opt
from libs.dataset import make_dataset, make_dataloader
from libs.train_utils import fix_random_seed

from libs.ql_utils import NEW_BRANCHES, FILTER_HIGH_LATENCY_BRANCHES, PARETO_FRONTIER_BRANCHES, ENLARGE79, ENLARGE82, CONTENTION55, CONTENTION12
from libs.online_eval_utils import init_online_eval


##### This version aims to train the dpo model
def main(opt, args):
    ### dataset and dataloader
    rng = fix_random_seed(opt.get('seed', 2023))
    dataset = make_dataset(opt['train']['data'])
    dataloader = make_dataloader(
        dataset, rng, opt['train']['batch_size'], opt['train']['num_workers']
    )
    # create tensorboard writer
    tb_writer = SummaryWriter(os.path.join(opt['_root'], 'logs'))
    losses_tracker = {}

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### create the model
    if 'model_name' not in opt:
        raise NotImplementedError
    elif 'model_name' in opt and opt['model_name'] == 'ResNet':
        raise NotImplementedError
    elif 'model_name' in opt and opt['model_name'] == 'SSM':
        model_type = SSM
    elif 'model_name' in opt and opt['model_name'] == 'SSM_contention':
        model_type = SSM_contention 
    elif 'model_name' in opt and opt['model_name'] == 'SSM_contention_two_branches':
        model_type = SSM_contention_two_branches         
    else:
        raise NotImplementedError
    
    # create the policy model
    model = model_type(**opt['model']).to(device)

    # load the reference model checkpoint
    ref_model_path = os.path.join(args.ref_model_path, 'models', 'last.pth')
    ref_model_ckpt = torch.load(ref_model_path, map_location='cpu')
    # determine the key
    if 'model' in ref_model_ckpt:
        print('using supervised model')
        model_key = 'model'
    elif 'actor' in ref_model_ckpt:
        print('using actor critic model')
        model_key = 'actor'
    else:
        raise NotImplementedError
    
    # Init the reference model using the reference model
    model.load_state_dict(ref_model_ckpt[model_key])

    # create the reference model
    ref_model = model_type(**opt['model']).to(device)

    # load the reference model
    ref_model.load_state_dict(ref_model_ckpt[model_key])
    # freeze the params in reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # ipdb.set_trace() # check the loading is correct
    # >>> test['model']['fc.layers.1.weight']
    # tensor([[ 0.0976,  1.0161,  0.2251,  ...,  1.2582,  1.9754,  0.2280],
    #         [ 0.0137, -0.3175,  0.0438,  ...,  0.9798,  2.6188,  0.0958],
    #         [ 0.0500,  0.3293,  0.0954,  ...,  0.4881,  1.9992,  0.0922],
    #         ...,
    #         [ 0.1223,  2.5387,  0.1298,  ...,  1.3522,  2.4124,  0.1059],
    #         [ 0.0998,  2.6053,  0.1338,  ...,  1.4534,  2.3913,  0.1541],
    #         [ 0.0803,  2.5943,  0.1150,  ...,  0.9575,  2.1020,  0.1586]],
    #        device='cuda:0')
    ### for resume the training
    starting_epoch = 0
    if opt['_resume']:
        model_path = os.path.join(opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        model.load_state_dict(model_ckpt['model'])

        # optimizer.load_state_dict(state_ckpt['optimizer'])
        #self.scheduler.load_state_dict(state_ckpt['scheduler'])
        epoch = state_ckpt['epoch']
        #e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print("Loaded checkpoint [epoch:" + str(epoch) + ' ]')
        starting_epoch = epoch + 1
    
    # set the hyper
    LEARNING_RATE = opt['train']['optimizer']['lr']
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if opt['_resume']:
        optimizer.load_state_dict(state_ckpt['optimizer'])
    num_epoches = opt['train']['num_epoches'] if 'num_epoches' in opt['train'] else 10

    # prepare the detection results
    branch_filter_type = opt['train']['branch_filter_type'] if 'branch_filter_type' in opt['train'] else 'default'
    use_det_res = opt['train']['use_det_res'] if 'use_det_res' in opt['train'] else False
    if use_det_res:
        # prepare all the profiling detection result
        ## load all the 40 branches
        if branch_filter_type == 'default':
            branches = NEW_BRANCHES
        elif branch_filter_type == 'filter_hl':
            branches = FILTER_HIGH_LATENCY_BRANCHES
        elif branch_filter_type == 'pf_only':
            branches = PARETO_FRONTIER_BRANCHES
        elif branch_filter_type == 'enlarge79':
            branches = ENLARGE79
        elif branch_filter_type == 'enlarge82':
            branches = ENLARGE82
        elif branch_filter_type == 'contention55':
            branches = CONTENTION55
        elif branch_filter_type == 'contention12':
            branches = CONTENTION12                 
        else:
            raise NotImplementedError        
        all_detection_res, waymo_gt, class_names = init_online_eval(branches, opt)
    else:
        all_detection_res, waymo_gt, class_names = None, None, None

    # traing loop
    curr_iter = starting_epoch * len(dataloader.dataset)
    for curr_epoch in range(starting_epoch, num_epoches):
        curr_iter = dpo_train_one_epoch(dataloader, model, ref_model, 
                                        optimizer, device, opt,
                                        all_detection_res, waymo_gt, class_names,
                                        losses_tracker=losses_tracker,
                                        tb_writer=tb_writer,
                                        curr_iter=curr_iter,
                                        lut_neg_action=args.lut_neg_action)
        print('finish training epoch:', curr_epoch)
        save_supervised_checkpoint(curr_epoch, opt, model, optimizer) # save check point for every epoch
        
    tb_writer.close()
    print('Complete')        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help="training options")
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--ref-model-path', dest= 'ref_model_path', type=str, help="path to the reference model")
    parser.add_argument('--lut-neg-action', dest= 'lut_neg_action', action='store_true', help="look up table as the negative actions")
    args = parser.parse_args()

    # create experiment folder
    os.makedirs('experiments', exist_ok=True)
    root = os.path.join('experiments', args.name)
    os.makedirs(root, exist_ok=True)
    
    # load the config 
    try:
        opt = load_opt(args.opt)
        shutil.copyfile(args.opt, os.path.join(root, 'opt.yaml'))
        os.makedirs(os.path.join(root, 'models'), exist_ok=True)
        os.makedirs(os.path.join(root, 'states'), exist_ok=True)
    except:
        raise FileNotFoundError
        
    # update the root of the model
    opt['_root'] = root
    # if the path exist previous models, resume the training
    opt['_resume'] = (
        os.path.exists(os.path.join(root, 'models', 'last.pth'))
        and os.path.exists(os.path.join(root, 'states', 'last.pth'))
    )
    
    main(opt, args)