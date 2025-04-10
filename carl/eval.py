import argparse
import os
import pickle
import torch
import ipdb
from libs import load_opt, Evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--ckpt', type=str, help="checkpoint name")
    parser.add_argument('--lantfilter', dest='lantfilter', action='store_true')
    parser.add_argument('--dumppickle', dest='dumppickle', action='store_true')
    parser.add_argument('--dumpname', type=str, default='prediction.pkl')
    parser.add_argument('--filterthresh', type=int, default=None)
    parser.add_argument('--saveroot', type=str, help="the root checkpoint name")
    parser.add_argument('--windsize', type=int, default=None)
    parser.add_argument('--usepolicy', dest='usepolicy', action='store_true')
    parser.add_argument('--notfilter', dest='notfilter', action='store_true')
    parser.add_argument('--filter-ratio', dest='filter_ratio', type=float, default=1.0)
    parser.add_argument('--interval-schedule', dest='interval_schedule', action='store_true')
    parser.add_argument('--contention-level', dest='contention_level', type=float, default=None)
    parser.add_argument('--lantency-file-path', dest='lantency_file_path', type=str, default=None)
    
    args = parser.parse_args()

    root = os.path.join('experiments', args.name)
    try:
        opt = load_opt(os.path.join(root, 'opt.yaml'))
    except:
        raise ValueError('experiment folder not found')
    assert os.path.exists(os.path.join(root, 'models', f'{args.ckpt}.pth'))
    opt['_root'] = root
    opt['_ckpt'] = args.ckpt
    
    # update the latency file 
    # ipdb.set_trace()
    if args.lantency_file_path is not None:
        opt['eval']['data']['latency_path'] = args.lantency_file_path
    
    if args.saveroot is not None and not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    evaluator = Evaluator(opt, args.usepolicy)
    if 'model_name' in opt and opt['model_name'] == 'QL':
        evaluator.run_rl()
    elif opt['eval']['data']['name'] == 'waymoWindow_rl_eval' and not args.dumppickle:
        latency_filter_thresh = opt['eval']['latency_filter_thresh'] if 'latency_filter_thresh' in opt['eval'] else None
        #filter_pred = opt['eval']['']
        filter_gt = opt['eval']['filter_gt'] if 'filter_gt' in opt['eval'] else False
        filter_pred = False
        
        # for the model which do not use the lantency in pruning:
        # we should either 1. not appling any latency filter 2. not apply filter on both pred and gt
        # for the model which use the latency in pruning:
        # we should add filter on only the GT
        if 'latency_filter_thresh' not in opt['train']: # the first situation the training is not using filtering
            #assert filter_gt is False
            if args.lantfilter:
                assert 'load_latency' in opt['eval']['data'] # assert loading the latency
                assert latency_filter_thresh is not None
                filter_gt = True
                filter_pred = True
            else:
                filter_gt = False
                filter_pred = False
        else: # the second situation
            assert filter_gt is True and filter_pred is False
        evaluator.run_rlv2(latency_filter_thresh, filter_pred=filter_pred, filter_gt=filter_gt)
    
    elif opt['eval']['data']['name'] == 'waymoWindow_rl_eval' and args.dumppickle:
        filter_pred = not args.notfilter
        res, selected_index = evaluator.run_rlv3(args.filterthresh, opt=opt, 
                                                 filter_pred=filter_pred, 
                                                 filter_ratio=args.filter_ratio,
                                                 interval_schedule=args.interval_schedule,
                                                 contention_level=args.contention_level)
        if args.saveroot is not None:
            with open(os.path.join(args.saveroot, args.dumpname), 'wb') as f:
                pickle.dump(res, f)
            torch.save(selected_index, os.path.join(args.saveroot, args.dumpname + 'idx.pt'))
        else:
            with open(os.path.join(root, args.dumpname), 'wb') as f:
                pickle.dump(res, f)
    else:
        evaluator.run()