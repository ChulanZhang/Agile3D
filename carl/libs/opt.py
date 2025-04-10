import yaml


DEFAULTS = {
    'seed': 1234567891,

    'model': {
        'planes': 64,
        'num_layers': 4,
    },

    'train': {
        'data': {
            'split': 'val',
            'size': (320, 280),
            'use_occ': False,
        },

        'batch_size': 16,
        'num_workers': 4,

        'epochs': 25,
        'warmup_epochs': 5,
        'ema_beta': 0.99,

        'optimizer': {
            'name': 'sgd',
            'lr': 1e-3,
            'momentem': 0.9,
            'weight_decay': 1e-4,
        },
        'clip_grad_norm': 1.0,

        'scheduler': {
            'name': 'multistep',
            'steps': (-1, ),
            'gamma': 0.1,
        },
    },

    'log': {
        'log_interval': 100,
        'checkpoint_epochs': (6, 7, 8, 9, 10),
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def _update_opt(opt):
    inplanes = 0
    assert opt['train']['data']['name'] in ('waymoWindow_rl', 'waymoWindow_rl_interval', 'waymoWindow_rl_eval', 'waymoWindow_contention_train')

    if opt['train']['data'].get('use_bev_feat'):
        inplanes += 192
    else:
        if opt['train']['data'].get('use_occ'):
            if opt['train']['data'].get('load_downsampled'):
                inplanes += 30
            else:
                inplanes += 60
        if opt['train']['data'].get('use_intensity'):
            inplanes += 1
        if opt['train']['data'].get('use_elongation'):
            inplanes += 1
    opt['model']['inplanes'] = inplanes
    
    if opt['train']['data'].get('lat_thresh'):
        opt['eval']['data']['lat_thresh'] = opt['train']['data']['lat_thresh']

    opt['train']['scheduler']['epochs'] = opt['train']['epochs']
    opt['train']['scheduler']['warmup_epochs'] = opt['train']['warmup_epochs']


def load_opt(filepath):
    with open(filepath, 'r') as fd:
        opt = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(DEFAULTS, opt)
    _update_opt(opt)
    return opt