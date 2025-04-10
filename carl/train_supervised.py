import argparse
import os
import shutil

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from libs.train_utils import supervised_train_one_epoch, save_supervised_checkpoint

from libs import SSM, SSM_contention, SSM_contention_two_branches
from libs import load_opt
from libs.dataset import make_dataset, make_dataloader
from libs.train_utils import fix_random_seed

from libs.ql_utils import NEW_BRANCHES, FILTER_HIGH_LATENCY_BRANCHES, PARETO_FRONTIER_BRANCHES, ENLARGE79, ENLARGE82, CONTENTION55, CONTENTION12
from libs.online_eval_utils import init_online_eval


##### This version aims training the supervised model
def main(opt):
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
    model = model_type(**opt['model']).to(device)

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
        curr_iter = supervised_train_one_epoch(dataloader, model, 
                                               optimizer, device, opt,
                                               all_detection_res, waymo_gt, class_names,
                                               losses_tracker=losses_tracker,
                                               tb_writer=tb_writer,
                                               curr_iter=curr_iter)
        print('finish training epoch:', curr_epoch)
        save_supervised_checkpoint(curr_epoch, opt, model, optimizer) # save check point for every epoch
        
    tb_writer.close()
    print('Complete')        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help="training options")
    parser.add_argument('--name', type=str, help="job name")
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
    
    main(opt)