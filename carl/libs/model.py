import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from functools import partial
from .blocks import conv3x3, conv1x1, downsample, MLP, BasicBlock, get_sinusoid_encoding
from .ssmblock import Mamba, Block, RLBlock, BidirMambav2

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# backbone (e.g., conv / transformer)
backbones = dict()
def register_backbone(name):
    def decorator(cls):
        backbones[name] = cls
        return cls
    return decorator


# builder functions
def make_backbone(name, **kwargs):
    backbone = backbones[name](**kwargs)
    return backbone


class ResNet(nn.Module):
    '''
        This is a backbone which aim for supervised scheduler, 
        which has a conv backbone and a linear layer to predict the prob of selecting each action
    '''
    def __init__(self, inplanes, planes=64, num_layers=4, num_classes=40):
        super().__init__()

        assert num_layers in (3, 4)
        kernel_size = 7 if num_layers == 4 else 5
        padding = 3 if num_layers == 4 else 2
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size,
            stride=2, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = BasicBlock(
            inplanes=planes, planes=planes,
            stride=2, downsample=downsample(planes, planes, 2)
        )
        self.layer2 = BasicBlock(
            inplanes=planes, planes=planes * 2,
            stride=2, downsample=downsample(planes, planes * 2, 2)
        )
        self.layer3 = BasicBlock(
            inplanes=planes * 2, planes=planes * 4,
            stride=2, downsample=downsample(planes * 2, planes * 4, 2)
        )
        if num_layers == 4:
            self.layer4 = BasicBlock(
                inplanes=planes * 4, planes=planes * 8,
                stride=2, downsample=downsample(planes * 4, planes * 8, 2)
            )
        else:
            self.layer4 = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes * 8 if num_layers == 4 else planes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
               nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x, extra_info=None): # input x: torch.Size([1, 62, 300, 300])
        # handle the shape change 
        if len(x.shape) == 5: # (B, L, X, Y ,Z)
            assert x.shape[0] == 1
            reduced_dim = True
            x = x.squeeze(dim=0)
        else:
            reduced_dim = False
        
        #print('input x:', x.shape)
        x = self.relu(self.bn1(self.conv1(x))) # after first conv: torch.Size([1, 64, 150, 150])
        #print('after first conv:', x.shape)
        x = self.maxpool(x) # after maxpool: torch.Size([1, 64, 75, 75])
        #print('after maxpool:', x.shape)

        x = self.layer1(x) # after layer1: torch.Size([1, 64, 38, 38])
        #print('after layer1:', x.shape)
        x = self.layer2(x) # after layer2: torch.Size([1, 128, 19, 19])
        #print('after layer2:', x.shape)
        x = self.layer3(x) # after layer3: torch.Size([1, 256, 10, 10])
        # print('after layer3:', x.shape)
        x = self.layer4(x) # after layer4: torch.Size([1, 512, 5, 5])
        #print('after layer4:', x.shape)

        x = self.avgpool(x).flatten(1) # after avgpool and flatten: torch.Size([1, 512])
        #print('after avgpool and flatten:', x.shape)
        x = self.fc(x)
        
        if reduced_dim: # maintain the shape of the output (B, L, Number_cls)
            x = x.unsqueeze(dim=0)

        return x
    

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    scale_factor=None,
    use_interpolate=False,
    use_bidir_mamba=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if use_bidir_mamba:
        mixer_name = BidirMambav2
    else:
        mixer_name = Mamba
    
    mixer_cls = partial(mixer_name, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        scale_factor=scale_factor,
        use_interpolate=use_interpolate
    )
    block.layer_idx = layer_idx
    return block


def create_rl_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    use_bidir_mamba=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if use_bidir_mamba:
        mixer_name = BidirMambav2
    else:
        mixer_name = Mamba
    
    mixer_cls = partial(mixer_name, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = RLBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


@register_backbone("conv")
class BaseConvBackbone(nn.Module):
    '''
        This is a convolution based backbone 
        which take occupancy grid feature as input 
        and encoder the feature into the vector 
    '''
    def __init__(self, num_layers, inplanes, planes):
        super().__init__()
        assert num_layers in (3, 4)
        kernel_size = 7 if num_layers == 4 else 5
        padding = 3 if num_layers == 4 else 2
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size,
            stride=2, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = BasicBlock(
            inplanes=planes, planes=planes,
            stride=2, downsample=downsample(planes, planes, 2)
        )
        self.layer2 = BasicBlock(
            inplanes=planes, planes=planes * 2,
            stride=2, downsample=downsample(planes, planes * 2, 2)
        )
        self.layer3 = BasicBlock(
            inplanes=planes * 2, planes=planes * 4,
            stride=2, downsample=downsample(planes * 2, planes * 4, 2)
        )
        if num_layers == 4:
            self.layer4 = BasicBlock(
                inplanes=planes * 4, planes=planes * 8,
                stride=2, downsample=downsample(planes * 4, planes * 8, 2)
            )
        else:
            self.layer4 = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # init the module 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
               nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def forward(self, x):
        #print('input x:', x.shape)
        x = self.relu(self.bn1(self.conv1(x))) # after first conv: torch.Size([B* L, 64, 150, 150])
        #print('after first conv:', x.shape)
        x = self.maxpool(x) # after maxpool: torch.Size([B* L, 64, 75, 75])
        #print('after maxpool:', x.shape)

        x = self.layer1(x) # after layer1: torch.Size([B* L, 64, 38, 38])
        #print('after layer1:', x.shape)
        x = self.layer2(x) # after layer2: torch.Size([B* L, 128, 19, 19])
        #print('after layer2:', x.shape)
        x = self.layer3(x) # after layer3: torch.Size([B* L, 256, 10, 10])
        # print('after layer3:', x.shape)
        x = self.layer4(x) # after layer4: torch.Size([B* L, 512, 5, 5])
        #print('after layer4:', x.shape)

        x = self.avgpool(x).flatten(1) # after avgpool and flatten: torch.Size([B* L, 512])
        #print('after avgpool and flatten:', x.shape)
        # ipdb.set_trace()
        # print('x:', x, 'critic:', self.critic)
        return x

@register_backbone("basevoxel")
class BaseVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        inplanes (int): z dimension of the input
        input_size (tuple(int)): x, y dimension of the input
        out_dim (int): output dimension
    '''

    def __init__(self, inplanes, input_size, planes=2, out_dim=512):
        super().__init__() # [60, 150, 150]
        self.actvn = F.relu
        self.inplanes = inplanes
        self.input_size = input_size
        
        self.conv_in = nn.Conv3d(1, planes, 3, padding=1, stride=2) # torch.Size([B* L, 1, 30, 150, 150]) -> torch.Size([B* L, 2, 15, 75, 75])

        self.conv_0 = nn.Conv3d(planes  , planes*2, 3, padding=1, stride=2) # torch.Size([B* L, 2, 15, 75, 75]) -> torch.Size([B* L, 4, 8, 38, 38])
        self.conv_1 = nn.Conv3d(planes*2, planes*4, 3, padding=1, stride=2) # torch.Size([B* L, 4, 8, 38, 38]) -> torch.Size([B* L, 8, 4, 19, 19])
        self.conv_2 = nn.Conv3d(planes*4, planes*8, 3, padding=1, stride=2) # torch.Size([B* L, 8, 4, 19, 19]) -> torch.Size([B* L, 16,  2, 10, 10])
        self.conv_3 = nn.Conv3d(planes*8, planes*16, 3, padding=1, stride=2) # torch.Size([B* L, 16, 2, 10, 10]) -> torch.Size([B* L, 32, 1, 5, 5])
        # self.conv_4 = nn.Conv3d(32, 64, 3, padding=1, stride=2) # torch.Size([B* L, 32, 1, 5, 5]) -> torch.Size([B* L, 64, 1, 3, 3])
        
        final_planes = planes * 16
        self.fc = nn.Linear(final_planes* 1* 5* 5, out_dim) # 800 -> 512

        # set bias term to 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # ipdb.set_trace()
        assert len(x.shape) == 4
        assert x.shape[1] == self.inplanes
        assert x.shape[2] == self.input_size[0]
        assert x.shape[3] == self.input_size[1]
        
        batch_size = x.size(0)

        x = x.unsqueeze(1) # from torch.Size([B*L, 60, 150, 150]) -> torch.Size([B*L, 1, 60, 150, 150])
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))
        # net = self.conv_4(self.actvn(net))

        hidden = net.view(batch_size, -1)
        c = self.fc(self.actvn(hidden))
        return c


@register_backbone("identity")
class IndentityBackbone(nn.Module):
    ''' This backbone do nothing for the GD-MAE feature
    '''
    def __init__(self):
        super().__init__() # [60, 150, 150]

    def forward(self, x):
        return x


class SSM(nn.Module):
    '''
        the RL model which has three parts:
        1. The backbone which encode the 3D feature into a vector
        2. The State space model (SSM) which encodes the temporal information
        3. A linear layer/MLP to prediction the action probability or state value (actor or critic)
    
    '''
    def __init__(self, 
                # backbone config
                inplanes, # the z dimension of the input
                planes=64, 
                num_layers=4, 
                backbone_name='conv',
                input_size=[150, 150], # the x,y dimension of the infput
                out_dim=512, # the output dimension of the backbone
                # ssm config
                ssm_layers=1,
                ssm_cfg=None,
                norm_epsilon: float=1e-5,
                rms_norm: bool=False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                use_bidir_mamba=False,
                # linear layer config
                num_classes=40, 
                use_sigmoid=False,
                use_mlp=False,
                embd_dim=None, 
                # latency config
                use_lat_thresh=None,
                learnable_latency_token=False,
                latency_token_type='vec',
                latency_thresholds=None,
                latency_embed_dim=None,
                # other model config
                critic=False,
                use_3d_feat_pred=False, # whether we will use feature after 3d backbone for action pred
                feat_use_method='concat', # concat or add
                use_det_info=False, # whether use previouse detection result in the training
                use_det_query=False,
                det_res_dim=7,
                det_res_embed_dim=64,
                det_info_fuse_method='late', # the 'late' means fusion this after SSM, the 'early' means fusion this before the SSM
                num_self_attn_layer=1,
                ):
        super().__init__()
        #### determine whether this is a critic net
        self.critic = critic
        self.use_sigmoid = use_sigmoid
        # prepare the hyper
        self.use_det_info = use_det_info
        self.use_det_query = use_det_query
        self.det_info_fuse_method = det_info_fuse_method
        
        #### init the backbone
        if not isinstance(backbone_name, list):
            self.backbone_name = [backbone_name]
        else:
            self.backbone_name = backbone_name
            
        self.backbone = nn.ModuleList()
        self.conv_out_dim = []
        for b_name in self.backbone_name:
            if b_name == 'conv':
                backbone_config_dict = {'num_layers':num_layers, 'inplanes': inplanes, 'planes': planes}
                self.backbone.append(make_backbone('conv', **backbone_config_dict))
                self.conv_out_dim.append(planes * 8 if num_layers == 4 else planes * 4)
                continue
            elif b_name == 'basevoxel':
                backbone_config_dict = {'inplanes': inplanes, 'input_size': input_size, 'planes': planes, 'out_dim': out_dim}
                self.backbone.append(make_backbone('basevoxel', **backbone_config_dict))
                self.conv_out_dim.append(out_dim)
                continue
            elif b_name == 'identity':
                self.backbone.append(make_backbone('identity', **{}))
                self.conv_out_dim.append(out_dim)
                continue
            else:
                raise NotImplementedError

        #### for the case that we need extra layer to match the dimension of the SSM
        if embd_dim is None:
            self.embd_dim = sum(self.conv_out_dim)
        else:
            self.embd_dim = embd_dim
        
        # add additional dimension for the detection result
        if self.use_det_info and self.det_info_fuse_method == 'early':
            self.embd_dim += det_res_embed_dim

        # additional mapping from the backbone feature dimenstion to the embeding dimension
        # if self.conv_out_dim != self.embd_dim:
        #     self.scaling_layer = nn.Linear(self.conv_out_dim, self.embd_dim)
        # else:
        #     self.scaling_layer = None
    
        #### init the SSM
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.use_bidir_mamba = use_bidir_mamba
        self.layers = nn.ModuleList()
        for idx in range(ssm_layers):
            self.layers.append(
                create_block(
                    self.embd_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=idx,
                    use_bidir_mamba=self.use_bidir_mamba,
                    **factory_kwargs,
                )
            )
        # define the final norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
                self.embd_dim, eps=norm_epsilon, **factory_kwargs
            )
        
        ### handle the learnable latency token
        self.learnable_latency_token = learnable_latency_token
        self.latency_token_type = latency_token_type
        self.latency_thresholds = latency_thresholds
        self.latency_embed_dim = latency_embed_dim if latency_embed_dim is not None else self.embd_dim
        if self.learnable_latency_token:
            if self.latency_token_type == 'vec':
                assert self.latency_thresholds is not None
                num_of_thresh = len(self.latency_thresholds)
                # define the token
                self.latency_tokens = nn.Parameter(torch.zeros((num_of_thresh, self.latency_embed_dim))) 
                self.from_latency_to_idx = {ele: i for i, ele in enumerate(self.latency_thresholds)}
                print('self.from_latency_to_idx:', self.from_latency_to_idx)
            elif self.latency_token_type == 'pe':
                assert self.latency_thresholds is not None
                assert len(self.latency_thresholds) == 2
                num_of_thresh = self.latency_thresholds[1] - self.latency_thresholds[0]
                 # call the func to generate the pe table
                self.latency_tokens = get_sinusoid_encoding(num_of_thresh, self.latency_embed_dim).squeeze(dim=0).permute([1,0]).cuda() # torch.Size([1, 512, 10]) -> torch.Size([dim, token_num]) -> torch.Size([token_num, dim])
            else:
                raise NotImplementedError
        
        # handle the special control for the skip connection
        self.use_lat_thresh = use_lat_thresh
        self.use_3d_feat_pred = use_3d_feat_pred
        self.feat_use_method = feat_use_method
        
        if not self.use_3d_feat_pred:
            feat_dims = self.embd_dim
        else:
            if self.feat_use_method == 'concat':
                feat_dims = self.embd_dim * 2
            elif self.feat_use_method == 'add':
                feat_dims = self.embd_dim
            else:
                raise NotImplementedError
        
        ### calculate the input dimension of the linear layer
        if self.use_lat_thresh and not self.learnable_latency_token:
            self.final_embd_dim = feat_dims + 1
        elif self.use_lat_thresh and self.learnable_latency_token:
            self.final_embd_dim = feat_dims + self.latency_embed_dim
        else:
            self.final_embd_dim = feat_dims
            
        # init the detection embed modules
        if use_det_info:
            self.det_fc = MLP(det_res_dim, det_res_embed_dim // 2, det_res_embed_dim, 2)  
            self.det_norm = nn.LayerNorm(det_res_embed_dim)
            self.num_self_attn_layer = num_self_attn_layer
            if self.use_det_query: # use transformer and query tokens to extract the feature
                self.det_query = nn.Parameter(torch.randn(1, det_res_embed_dim))
                
                self.det_self_attn = nn.ModuleList()
                for i in range(self.num_self_attn_layer):
                    self.det_self_attn.append(
                            torch.nn.MultiheadAttention(det_res_embed_dim, 2, batch_first=True)
                        )
            
            if self.det_info_fuse_method == 'late':
                self.final_embd_dim += det_res_embed_dim 
        
        # define the classification layer the MLP
        self.use_mlp = use_mlp
        output_dim = 1 if self.critic else num_classes
        if self.use_mlp:
            self.fc = MLP(self.final_embd_dim, self.final_embd_dim//2, output_dim, 2)
        else:
            self.fc = nn.Linear(self.final_embd_dim, output_dim)

        # set bias term to 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    if not getattr(m.bias, "_no_reinit", False):
                        nn.init.zeros_(m.bias)

    def get_det_res_stat(self, detect_info):
        # all_bboxes_coordinate: [B(1), T, N, 7] torch.Size([1, 1, 50, 7])
        # all_bbox_confidences: [B(1), T, N] torch.Size([1, 1, 50])
        # all_bbox_categories: [B(1), T, N] torch.Size([1, 1, 50])
        #ipdb.set_trace()
        all_bboxes_coordinate, all_bbox_categories, all_bbox_confidences = detect_info
        
        if all_bboxes_coordinate is None: # which is the first step of the loop
            return [[0.0]*7]
        
        assert all_bboxes_coordinate.shape[0] == 1
        timesteps = all_bboxes_coordinate.shape[1]
        all_stat = []
        
        # calculate for each step
        for i in range(timesteps):
            # each step is [mean_of_conf, std_of_conf, 
            # count_of_car, count_of_person, count_of_bike, 
            # mean_of_size, std_of_size]
            
            curr_corrdinate = all_bboxes_coordinate[0, i]
            curr_categories = all_bbox_categories[0, i]
            curr_confidences = all_bbox_confidences[0, i]
            
            # calculate the mean and std of the confidence
            confidence_mean = torch.mean(curr_confidences)
            confidence_std = torch.std(curr_confidences)
            
            # count the number of each category
            count_of_car = torch.sum(curr_categories==1)
            count_of_person = torch.sum(curr_categories==2)
            count_of_bike = torch.sum(curr_categories==3)
            
            # calulate the size of all bboxes
            all_x_range = torch.abs(curr_corrdinate[:, 0] - curr_corrdinate[:, 3])
            all_y_range = torch.abs(curr_corrdinate[:, 1] - curr_corrdinate[:, 4])
            all_z_range = torch.abs(curr_corrdinate[:, 2] - curr_corrdinate[:, 5])
            all_size = all_x_range * all_y_range * all_z_range
            
            # calculate the mean and std of the bbox size
            size_mean = torch.mean(all_size)
            size_std = torch.std(all_size)
            
            # turn it into a vector
            stat_of_curr_step = [confidence_mean, confidence_std, 
                                 count_of_car, count_of_person, count_of_bike, 
                                 size_mean, size_std]
            all_stat.append(stat_of_curr_step)
        #ipdb.set_trace()
        return all_stat    

    def forward(self, x, extra_info=None, detect_info=None): 
        # input x: torch.Size([B, L, 62, 300, 300])
        # extra_info should be in the shape of (B, L) should be the value of the current latency threshold
        # ipdb.set_trace()
        if detect_info is not None and None not in detect_info:
            if self.use_det_query:
                # ipdb.set_trace()
                all_embedded_res = []
                
                all_bboxes_coordinate, all_bbox_categories, all_bbox_confidences = detect_info
                total_time_step = len(all_bboxes_coordinate)
                # padd one empty at the begining 
                first_detect_res = (torch.zeros(1,1,1,7), torch.zeros(1,1,1), torch.zeros(1,1,1))
                first_detect_res = torch.cat([first_detect_res[0], first_detect_res[1].unsqueeze(dim=-1), first_detect_res[2].unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                first_det_embedding = self.det_fc(first_detect_res) # torch.Size([1, 1, 50, 64])
                B, T, Num_Det, D = first_det_embedding.shape
                first_det_embedding = first_det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])      
                first_det_input = torch.cat([first_det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)          
                for layer in self.det_self_attn:
                    first_det_input, _ = layer(first_det_input, first_det_input, first_det_input) # B*T, N+1, D
                first_det_embedding = first_det_input[:, -1].view(B, T, D)                
                all_embedded_res.append(first_det_embedding)
                
                # pop the last one.
                for i in range(total_time_step):
                    curr_coord, curr_cate, curr_confi = all_bboxes_coordinate[i], all_bbox_categories[i], all_bbox_confidences[i]
                    if i < total_time_step - 1:
                        curr_detect_res = torch.cat([curr_coord, curr_cate.unsqueeze(dim=-1), curr_confi.unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                        curr_det_embedding = self.det_fc(curr_detect_res) # torch.Size([1, 1, 50, 64])
                        B, T, Num_Det, D = curr_det_embedding.shape
                        curr_det_embedding = curr_det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])  
                        curr_det_input = torch.cat([curr_det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)     
                        for layer in self.det_self_attn:
                            curr_det_input, _ = layer(curr_det_input, curr_det_input, curr_det_input) # B*T, N+1, D
                        curr_det_embedding = curr_det_input[:, -1].view(B, T, D)   
                        all_embedded_res.append(curr_det_embedding)
                
                # concat along the temporal axis
                det_embedding = torch.cat(all_embedded_res, dim=1)
                # ipdb.set_trace() # check the det_embedding again
            else:
                # TODO: this branch may have issue since we remove the concatention
                det_stat = self.get_det_res_stat(detect_info)
                # remove the last one
                det_stat.pop(-1)
                # pad zero at the beginning
                det_stat.insert(0, [0.0]*7)
                det_stat = torch.tensor(det_stat).unsqueeze(dim=0).to(x[0].device)
                det_embedding = self.det_fc(det_stat)
                # pass through a norm
                det_embedding = self.det_norm(det_embedding)
        else:
            det_embedding = None
            
        # ipdb.set_trace() # check the forward
        all_processed_feat = []
        for i, curr_feat_x in enumerate(x):
            # TODO: reshape each features
            if len(curr_feat_x.shape) == 5:
                merge_BL = True
                B, L, Z, X, Y = curr_feat_x.shape
                curr_feat_x = curr_feat_x.reshape(-1, Z, X, Y)
            else:
                merge_BL = False
            
            # feature Forward to the model
            curr_feat_x = self.backbone[i](curr_feat_x)
            
            # reshape one by one reshape from torch.Size([B* L, D]) back to torch.Size([B, L, D])
            if merge_BL:
                curr_feat_x = curr_feat_x.reshape(B, L, self.conv_out_dim[i])
            
            all_processed_feat.append(curr_feat_x)
            
        # TODO: handle the concatenation of x, make it also compatible with the old design
        # ipdb.set_trace() # check the concat ans special handle
        if len(all_processed_feat) == 1:
            x = all_processed_feat[0]
        else:
            x = torch.cat(all_processed_feat, dim=-1)
        
        # the linear layer for mapping the dimension
        # if self.scaling_layer is not None:
        #     x = self.scaling_layer(x)
        
        # concat the detection result embedding with feature x
        if det_embedding is not None and self.det_info_fuse_method == 'early':
            x = torch.cat([x, det_embedding], dim=-1)
        
        # the ssm block
        # convert the dim from  (B* L, D) to (B, L, D)
        residual = None
        mask = None
        for layer in self.layers:
            temporal, residual, mask = layer(
                x, residual, 
                inference_params=None,
                mask=mask,
                ctrl_vec=None,
            )
        
        # for the normalization after ssm
        if not self.fused_add_norm:
            residual = (temporal + residual) if residual is not None else temporal
            temporal = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            temporal = fused_add_norm_fn(
                temporal,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # may concat the 3d feature with the temporal feature
        if self.use_3d_feat_pred:
            if self.feat_use_method == 'concat':
                x = torch.cat([x, temporal], dim=-1)
            elif self.feat_use_method == 'add':
                x = x + temporal
            else:
                raise NotImplementedError     
        else:
            x = temporal       
            
        # x: torch.Size([1, 198, 512])
        # extra_info should be in the shape of (B, L)
        # using learnable latency_tokens then it should be torch.Size([1, 198, 512])
        all_feature_list = [x, ]
        
        if self.use_lat_thresh:
            assert extra_info is not None 
            if self.learnable_latency_token: # ensure the table should be (#token, #dim)
                if self.latency_token_type == 'vec': 
                    all_index = (extra_info // 50 - 2).int() ## TODO: this simple solution might cause bug
                    all_tokens = self.latency_tokens[all_index]
                    all_feature_list.append(all_tokens)
                elif self.latency_token_type == 'pe':
                    all_index = (extra_info - self.latency_thresholds[0]).int()
                    #print('all_index:', all_index, torch.max(all_index), torch.min(all_index))
                    all_tokens = self.latency_tokens[all_index]
                    #print('all_tokens:', all_tokens.shape, 'x:', x.shape)
                    all_feature_list.append(all_tokens)
                else:
                    raise NotImplementedError
            else:
                if len(extra_info.shape) == 2 and len(x.shape) == 3: ## a value threshodl
                    extra_info = extra_info.unsqueeze(dim=-1)
                    all_feature_list.append(extra_info)
        
        if det_embedding is not None and self.det_info_fuse_method == 'late':
            all_feature_list.append(det_embedding)
                
        x = torch.cat(all_feature_list, dim=-1)
        x = self.fc(x) # return will be (B, L, D)

        # assuming that our qvalue should be proportional to the mAP value
        # the output should between [0,1]
        if self.use_sigmoid:
            x = F.sigmoid(x)

        return x
    
    def step(self, x, latency, detect_res=None, prev_action_state=None): 
        # input: x: torch.Size([B, 1, 62, 300, 300])
        #        latency: torch.Size([B, 1])
        #        detect_res: ? 
        #        prev_action_state: ?
        # ipdb.set_trace()
        if detect_res is not None:
            if self.use_det_query: # concat the feature and query and send it into the model, take the last one
                # ipdb.set_trace() # check dimension
                # all_bboxes_coordinate: [B(1), T, N, 7] torch.Size([1, 1, 50, 7]) # B = 1, T = 1, detection result = 50, dim = 7
                # all_bbox_confidences: [B(1), T, N] torch.Size([1, 1, 50])
                # all_bbox_categories: [B(1), T, N] torch.Size([1, 1, 50])
                # TODO: encode each detection result
                # concat and embeding all the information
                if None in detect_res: # do the padding if no detection result
                    detect_res = (torch.zeros(1,1,1,7), torch.zeros(1,1,1), torch.zeros(1,1,1))
                
                detect_res = torch.cat([detect_res[0], detect_res[1].unsqueeze(dim=-1), detect_res[2].unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                det_embedding = self.det_fc(detect_res) # torch.Size([1, 1, 50, 64])
                B, T, Num_Det, D = det_embedding.shape
                det_embedding = det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])
                
                all_det_input = torch.cat([det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)
                # ipdb.set_trace() # check dimension
                for layer in self.det_self_attn:
                    all_det_input, _ = layer(all_det_input, all_det_input, all_det_input) # B*T, N+1, D
                det_embedding = all_det_input[:, -1].view(B, T, D)
                # ipdb.set_trace() # check dimension again
            else:
                det_stat = self.get_det_res_stat(detect_res)  #
                det_stat = torch.tensor(det_stat).unsqueeze(dim=0).to(x.device)  # 
                det_embedding = self.det_fc(det_stat)
        else:
            det_embedding = None
        
        # ipdb.set_trace() # check the forward
        all_processed_feat = []
        for i, curr_feat_x in enumerate(x):
            # TODO: reshape each features
            if len(curr_feat_x.shape) == 5:
                merge_BL = True
                B, L, Z, X, Y = curr_feat_x.shape
                curr_feat_x = curr_feat_x.reshape(-1, Z, X, Y)
            else:
                merge_BL = False
            
            # feature Forward to the model
            curr_feat_x = self.backbone[i](curr_feat_x)
            
            # reshape one by one reshape from torch.Size([B* L, D]) back to torch.Size([B, L, D])
            if merge_BL:
                curr_feat_x = curr_feat_x.reshape(B, L, self.conv_out_dim[i])
            
            all_processed_feat.append(curr_feat_x)
            
        # TODO: handle the concatenation of x, make it also compatible with the old design
        # ipdb.set_trace() # check the concat ans special handle
        if len(all_processed_feat) == 1:
            x = all_processed_feat[0]
        else:
            x = torch.cat(all_processed_feat, dim=-1)
        
        
        # the linear layer for mapping the dimension
        # if self.scaling_layer is not None:
        #     x = self.scaling_layer(x)
        
        # concat the detection result embedding with feature x
        if det_embedding is not None and self.det_info_fuse_method == 'early':
            x = torch.cat([x, det_embedding], dim=-1)
            # ipdb.set_trace()
        
        # the ssm block
        # convert the dim from  (B* L, D) to (B, L, D)
        residual = None
        mask = None
        # conv_state, ssm_state = prev_action_state
        new_action_state = []
        temporal = x
        for (old_conv_state, old_ssm_state), layer in zip(prev_action_state, self.layers):
            temporal, residual, new_conv_state, new_ssm_state = layer.step(
                temporal, residual, old_conv_state, old_ssm_state, 
            )
            new_action_state.append((new_conv_state, new_ssm_state))
        
        # for the normalization after ssm
        if not self.fused_add_norm:
            residual = (temporal + residual) if residual is not None else temporal
            temporal = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            temporal = fused_add_norm_fn(
                temporal,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # may concat the 3d feature with the temporal feature
        if self.use_3d_feat_pred:
            if self.feat_use_method == 'concat':
                x = torch.cat([x, temporal], dim=-1)
            elif self.feat_use_method == 'add':
                x = x + temporal
            else:
                raise NotImplementedError     
        else:
            x = temporal       
        
        all_feature_list = [x, ]
        # x: torch.Size([1, 198, 512])
        # latency: should be in the shape of (B, L)
        # using learnable latency_tokens then it should be torch.Size([1, 198, 512])
        if self.use_lat_thresh:
            assert latency is not None 
            if self.learnable_latency_token: # ensure the table should be (#token, #dim)
                if self.latency_token_type == 'vec': 
                    all_index = (latency // 50 - 2).int() ## TODO: this simple solution might cause bug
                    all_tokens = self.latency_tokens[all_index]
                    all_feature_list.append(all_tokens)
                elif self.latency_token_type == 'pe':
                    all_index = (latency - self.latency_thresholds[0]).int()
                    #print('all_index:', all_index, torch.max(all_index), torch.min(all_index))
                    all_tokens = self.latency_tokens[all_index]
                    #print('all_tokens:', all_tokens.shape, 'x:', x.shape)
                    all_feature_list.append(all_tokens)
                else:
                    raise NotImplementedError
            else:
                if len(latency.shape) == 2 and len(x.shape) == 3: ## a value threshodl
                    latency = latency.unsqueeze(dim=-1)
                all_feature_list.append(latency)
        
        # ipdb.set_trace() # control the dimenstion
        if det_embedding is not None and self.det_info_fuse_method == 'late':
            all_feature_list.append(det_embedding)
        
        x = torch.cat(all_feature_list, dim=-1)
        x = self.fc(x) # return will be (B, L, D)

        # assuming that our qvalue should be proportional to the mAP value
        # the output should between [0,1]
        if self.use_sigmoid:
            x = F.sigmoid(x)

        return x, new_action_state



class SSM_contention(nn.Module):
    '''
        This version of the model aims to handle the contention input but ignore the latency slo
        the RL model which has three parts:
        1. The backbone which encode the 3D feature into a vector
        2. The State space model (SSM) which encodes the temporal information
        3. A linear layer/MLP to prediction the action probability or state value (actor or critic)
    
    '''
    def __init__(self, 
                # backbone config
                inplanes, # the z dimension of the input
                planes=64, 
                num_layers=4, 
                backbone_name='conv',
                input_size=[150, 150], # the x,y dimension of the infput
                out_dim=512, # the output dimension of the backbone
                # ssm config
                ssm_layers=1,
                ssm_cfg=None,
                norm_epsilon: float=1e-5,
                rms_norm: bool=False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                use_bidir_mamba=False,
                # linear layer config
                num_classes=40, 
                use_sigmoid=False,
                use_mlp=False,
                embd_dim=None, 
                # latency config
                use_contention=None,
                learnable_contention_token=False,
                contention_token_type='vec',
                contention_embed_dim=None,
                contention_levels=None,
                
                # other model config
                critic=False,
                use_3d_feat_pred=False, # whether we will use feature after 3d backbone for action pred
                feat_use_method='concat', # concat or add
                use_det_info=False, # whether use previouse detection result in the training
                use_det_query=False,
                det_res_dim=7,
                det_res_embed_dim=64,
                det_info_fuse_method='late', # the 'late' means fusion this after SSM, the 'early' means fusion this before the SSM
                num_self_attn_layer=1,
                ):
        super().__init__()
        #### determine whether this is a critic net
        self.critic = critic
        self.use_sigmoid = use_sigmoid
        # prepare the hyper
        self.use_det_info = use_det_info
        self.use_det_query = use_det_query
        self.det_info_fuse_method = det_info_fuse_method
        
        ######################  init the backbone ########################################
        if not isinstance(backbone_name, list):
            self.backbone_name = [backbone_name]
        else:
            self.backbone_name = backbone_name
            
        self.backbone = nn.ModuleList()
        self.conv_out_dim = []
        for b_name in self.backbone_name:
            if b_name == 'conv':
                backbone_config_dict = {'num_layers':num_layers, 'inplanes': inplanes, 'planes': planes}
                self.backbone.append(make_backbone('conv', **backbone_config_dict))
                self.conv_out_dim.append(planes * 8 if num_layers == 4 else planes * 4)
                continue
            elif b_name == 'basevoxel':
                backbone_config_dict = {'inplanes': inplanes, 'input_size': input_size, 'planes': planes, 'out_dim': out_dim}
                self.backbone.append(make_backbone('basevoxel', **backbone_config_dict))
                self.conv_out_dim.append(out_dim)
                continue
            elif b_name == 'identity':
                self.backbone.append(make_backbone('identity', **{}))
                self.conv_out_dim.append(out_dim)
                continue
            else:
                raise NotImplementedError

        #### for the case that we need extra layer to match the dimension of the SSM
        if embd_dim is None:
            self.embd_dim = sum(self.conv_out_dim)
        else:
            self.embd_dim = embd_dim
        
        # add additional dimension for the detection result
        if self.use_det_info and self.det_info_fuse_method == 'early':
            self.embd_dim += det_res_embed_dim

        # additional mapping from the backbone feature dimenstion to the embeding dimension
        # if self.conv_out_dim != self.embd_dim:
        #     self.scaling_layer = nn.Linear(self.conv_out_dim, self.embd_dim)
        # else:
        #     self.scaling_layer = None
    
        #### init the SSM
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.use_bidir_mamba = use_bidir_mamba
        self.layers = nn.ModuleList()
        for idx in range(ssm_layers):
            self.layers.append(
                create_block(
                    self.embd_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=idx,
                    use_bidir_mamba=self.use_bidir_mamba,
                    **factory_kwargs,
                )
            )
        # define the final norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
                self.embd_dim, eps=norm_epsilon, **factory_kwargs
            )
        
        ### handle the learnable contention
        self.learnable_contention_token = learnable_contention_token
        self.contention_token_type = contention_token_type
        self.contention_levels = contention_levels
        self.contention_embed_dim = contention_embed_dim if contention_embed_dim is not None else self.embd_dim
        if self.learnable_contention_token:
            if self.contention_token_type == 'pe':
                assert self.contention_levels is not None
                # assert len(self.latency_thresholds) == 2
                num_of_thresh = len(self.contention_levels)
                 # call the func to generate the pe table
                self.contention_tokens = get_sinusoid_encoding(num_of_thresh, self.contention_embed_dim).squeeze(dim=0).permute([1,0]).cuda() # torch.Size([1, 512, 10]) -> torch.Size([dim, token_num]) -> torch.Size([token_num, dim])
            else:
                raise NotImplementedError
        
        # handle the special control for the skip connection
        self.use_contention = use_contention
        self.use_3d_feat_pred = use_3d_feat_pred
        self.feat_use_method = feat_use_method
        
        if not self.use_3d_feat_pred:
            feat_dims = self.embd_dim
        else:
            if self.feat_use_method == 'concat':
                feat_dims = self.embd_dim * 2
            elif self.feat_use_method == 'add':
                feat_dims = self.embd_dim
            else:
                raise NotImplementedError
        
        ### calculate the input dimension of the linear layer
        if self.use_contention and self.learnable_contention_token:
            self.final_embd_dim = feat_dims + self.contention_embed_dim
        else:
            self.final_embd_dim = feat_dims
            
        # init the detection embed modules
        if use_det_info:
            self.det_fc = MLP(det_res_dim, det_res_embed_dim // 2, det_res_embed_dim, 2)  
            self.det_norm = nn.LayerNorm(det_res_embed_dim)
            self.num_self_attn_layer = num_self_attn_layer
            if self.use_det_query: # use transformer and query tokens to extract the feature
                self.det_query = nn.Parameter(torch.randn(1, det_res_embed_dim))
                
                self.det_self_attn = nn.ModuleList()
                for i in range(self.num_self_attn_layer):
                    self.det_self_attn.append(
                            torch.nn.MultiheadAttention(det_res_embed_dim, 2, batch_first=True)
                        )
            
            if self.det_info_fuse_method == 'late':
                self.final_embd_dim += det_res_embed_dim 
        
        # define the classification layer the MLP
        self.use_mlp = use_mlp
        output_dim = 1 if self.critic else num_classes
        if self.use_mlp:
            self.fc = MLP(self.final_embd_dim, self.final_embd_dim//2, output_dim, 2)
        else:
            self.fc = nn.Linear(self.final_embd_dim, output_dim)

        # set bias term to 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    if not getattr(m.bias, "_no_reinit", False):
                        nn.init.zeros_(m.bias)

    def get_det_res_stat(self, detect_info):
        # all_bboxes_coordinate: [B(1), T, N, 7] torch.Size([1, 1, 50, 7])
        # all_bbox_confidences: [B(1), T, N] torch.Size([1, 1, 50])
        # all_bbox_categories: [B(1), T, N] torch.Size([1, 1, 50])
        #ipdb.set_trace()
        all_bboxes_coordinate, all_bbox_categories, all_bbox_confidences = detect_info
        
        if all_bboxes_coordinate is None: # which is the first step of the loop
            return [[0.0]*7]
        
        assert all_bboxes_coordinate.shape[0] == 1
        timesteps = all_bboxes_coordinate.shape[1]
        all_stat = []
        
        # calculate for each step
        for i in range(timesteps):
            # each step is [mean_of_conf, std_of_conf, 
            # count_of_car, count_of_person, count_of_bike, 
            # mean_of_size, std_of_size]
            
            curr_corrdinate = all_bboxes_coordinate[0, i]
            curr_categories = all_bbox_categories[0, i]
            curr_confidences = all_bbox_confidences[0, i]
            
            # calculate the mean and std of the confidence
            confidence_mean = torch.mean(curr_confidences)
            confidence_std = torch.std(curr_confidences)
            
            # count the number of each category
            count_of_car = torch.sum(curr_categories==1)
            count_of_person = torch.sum(curr_categories==2)
            count_of_bike = torch.sum(curr_categories==3)
            
            # calulate the size of all bboxes
            all_x_range = torch.abs(curr_corrdinate[:, 0] - curr_corrdinate[:, 3])
            all_y_range = torch.abs(curr_corrdinate[:, 1] - curr_corrdinate[:, 4])
            all_z_range = torch.abs(curr_corrdinate[:, 2] - curr_corrdinate[:, 5])
            all_size = all_x_range * all_y_range * all_z_range
            
            # calculate the mean and std of the bbox size
            size_mean = torch.mean(all_size)
            size_std = torch.std(all_size)
            
            # turn it into a vector
            stat_of_curr_step = [confidence_mean, confidence_std, 
                                 count_of_car, count_of_person, count_of_bike, 
                                 size_mean, size_std]
            all_stat.append(stat_of_curr_step)
        #ipdb.set_trace()
        return all_stat    

    def forward(self, x, extra_info=None, detect_info=None): 
        # input x: torch.Size([B, L, 62, 300, 300])
        # extra_info should be in the shape of (B, L) should be the value of the current latency threshold
        # ipdb.set_trace() # check the forward
        if detect_info is not None and None not in detect_info:
            if self.use_det_query:
                # ipdb.set_trace() # check whethet the shape of the self-attention input # torch.Size([1, 1, 50, 64])
                all_embedded_res = []
                
                all_bboxes_coordinate, all_bbox_categories, all_bbox_confidences = detect_info
                total_time_step = len(all_bboxes_coordinate)
                # padd one empty at the begining 
                first_detect_res = (torch.zeros(1,1,1,7), torch.zeros(1,1,1), torch.zeros(1,1,1))
                first_detect_res = torch.cat([first_detect_res[0], first_detect_res[1].unsqueeze(dim=-1), first_detect_res[2].unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                first_det_embedding = self.det_fc(first_detect_res) # torch.Size([1, 1, 50, 64])
                B, T, Num_Det, D = first_det_embedding.shape
                first_det_embedding = first_det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])      
                first_det_input = torch.cat([first_det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)          
                for layer in self.det_self_attn:
                    first_det_input, _ = layer(first_det_input, first_det_input, first_det_input) # B*T, N+1, D
                first_det_embedding = first_det_input[:, -1].view(B, T, D)                
                all_embedded_res.append(first_det_embedding)
                
                # pop the last one.
                for i in range(total_time_step):
                    curr_coord, curr_cate, curr_confi = all_bboxes_coordinate[i], all_bbox_categories[i], all_bbox_confidences[i]
                    if i < total_time_step - 1:
                        curr_detect_res = torch.cat([curr_coord, curr_cate.unsqueeze(dim=-1), curr_confi.unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                        curr_det_embedding = self.det_fc(curr_detect_res) # torch.Size([1, 1, 50, 64])
                        B, T, Num_Det, D = curr_det_embedding.shape
                        curr_det_embedding = curr_det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])  
                        curr_det_input = torch.cat([curr_det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)     
                        for layer in self.det_self_attn:
                            curr_det_input, _ = layer(curr_det_input, curr_det_input, curr_det_input) # B*T, N+1, D
                        curr_det_embedding = curr_det_input[:, -1].view(B, T, D)   
                        all_embedded_res.append(curr_det_embedding)
                
                # concat along the temporal axis
                det_embedding = torch.cat(all_embedded_res, dim=1)
                # ipdb.set_trace() # check the det_embedding again
            else:
                # TODO: this branch may have issue since we remove the concatention
                det_stat = self.get_det_res_stat(detect_info)
                # remove the last one
                det_stat.pop(-1)
                # pad zero at the beginning
                det_stat.insert(0, [0.0]*7)
                det_stat = torch.tensor(det_stat).unsqueeze(dim=0).to(x[0].device)
                det_embedding = self.det_fc(det_stat)
                # pass through a norm
                det_embedding = self.det_norm(det_embedding)
        else:
            det_embedding = None
            
        # ipdb.set_trace() # check the forward
        all_processed_feat = []
        for i, curr_feat_x in enumerate(x):
            # TODO: reshape each features
            if len(curr_feat_x.shape) == 5:
                merge_BL = True
                B, L, Z, X, Y = curr_feat_x.shape
                curr_feat_x = curr_feat_x.reshape(-1, Z, X, Y)
            else:
                merge_BL = False
            
            # feature Forward to the model
            curr_feat_x = self.backbone[i](curr_feat_x)
            
            # reshape one by one reshape from torch.Size([B* L, D]) back to torch.Size([B, L, D])
            if merge_BL:
                curr_feat_x = curr_feat_x.reshape(B, L, self.conv_out_dim[i])
            
            all_processed_feat.append(curr_feat_x)
            
        # handle the concatenation of x, make it also compatible with the old design
        # ipdb.set_trace() # check the concat ans special handle
        if len(all_processed_feat) == 1:
            x = all_processed_feat[0]
        else:
            x = torch.cat(all_processed_feat, dim=-1)
        
        # the linear layer for mapping the dimension
        # if self.scaling_layer is not None:
        #     x = self.scaling_layer(x)
        
        # concat the detection result embedding with feature x
        if det_embedding is not None and self.det_info_fuse_method == 'early':
            x = torch.cat([x, det_embedding], dim=-1)
        
        # the ssm block
        # convert the dim from  (B* L, D) to (B, L, D)
        residual = None
        mask = None
        for layer in self.layers:
            temporal, residual, mask = layer(
                x, residual, 
                inference_params=None,
                mask=mask,
                ctrl_vec=None,
            )
        
        # for the normalization after ssm
        if not self.fused_add_norm:
            residual = (temporal + residual) if residual is not None else temporal
            temporal = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            temporal = fused_add_norm_fn(
                temporal,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # may concat the 3d feature with the temporal feature
        if self.use_3d_feat_pred:
            if self.feat_use_method == 'concat':
                x = torch.cat([x, temporal], dim=-1)
            elif self.feat_use_method == 'add':
                x = x + temporal
            else:
                raise NotImplementedError     
        else:
            x = temporal       
            
        # x: torch.Size([1, 198, 512])
        # extra_info should be in the shape of (B, L)
        # using learnable latency_tokens then it should be torch.Size([1, 198, 512])
        all_feature_list = [x, ]
        
        if self.contention_levels:
            assert extra_info is not None 
            if self.learnable_contention_token: # ensure the table should be (#token, #dim)
                if self.contention_token_type == 'pe':
                    # TODO: find the index
                    # ipdb.set_trace()
                    all_index = torch.zeros(extra_info.shape)
                    # map each value
                    all_index[extra_info==0.0] = 0
                    all_index[extra_info==0.2] = 1
                    all_index[extra_info==0.5] = 2
                    all_index[extra_info==0.9] = 3
                    all_index = all_index.to(dtype=torch.int)
                    # all_index = []
                    # for ele in +
                    # index = my_list.index(3)
                    # all_index = (extra_info - self.latency_thresholds[0]).int()
                    #print('all_index:', all_index, torch.max(all_index), torch.min(all_index))
                    all_tokens = self.contention_tokens[all_index]
                    #print('all_tokens:', all_tokens.shape, 'x:', x.shape)
                    all_feature_list.append(all_tokens)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        if det_embedding is not None and self.det_info_fuse_method == 'late':
            all_feature_list.append(det_embedding)
                
        x = torch.cat(all_feature_list, dim=-1)
        x = self.fc(x) # return will be (B, L, D)

        # assuming that our qvalue should be proportional to the mAP value
        # the output should between [0,1]
        if self.use_sigmoid:
            x = F.sigmoid(x)

        return x
    
    def step(self, x, latency, detect_res=None, prev_action_state=None): 
        # input: x: torch.Size([B, 1, 62, 300, 300])
        #        latency: torch.Size([B, 1])
        #        detect_res: ? 
        #        prev_action_state: ?
        # ipdb.set_trace()
        if detect_res is not None:
            if self.use_det_query: # concat the feature and query and send it into the model, take the last one
                # ipdb.set_trace() # check dimension
                # all_bboxes_coordinate: [B(1), T, N, 7] torch.Size([1, 1, 50, 7]) # B = 1, T = 1, detection result = 50, dim = 7
                # all_bbox_confidences: [B(1), T, N] torch.Size([1, 1, 50])
                # all_bbox_categories: [B(1), T, N] torch.Size([1, 1, 50])
                # TODO: encode each detection result
                # concat and embeding all the information
                if None in detect_res: # do the padding if no detection result
                    detect_res = (torch.zeros(1,1,1,7), torch.zeros(1,1,1), torch.zeros(1,1,1))
                
                detect_res = torch.cat([detect_res[0], detect_res[1].unsqueeze(dim=-1), detect_res[2].unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                det_embedding = self.det_fc(detect_res) # torch.Size([1, 1, 50, 64])
                B, T, Num_Det, D = det_embedding.shape
                det_embedding = det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])
                
                all_det_input = torch.cat([det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)
                # ipdb.set_trace() # check dimension
                for layer in self.det_self_attn:
                    all_det_input, _ = layer(all_det_input, all_det_input, all_det_input) # B*T, N+1, D
                det_embedding = all_det_input[:, -1].view(B, T, D)
                # ipdb.set_trace() # check dimension again
            else:
                det_stat = self.get_det_res_stat(detect_res)  #
                det_stat = torch.tensor(det_stat).unsqueeze(dim=0).to(x.device)  # 
                det_embedding = self.det_fc(det_stat)
        else:
            det_embedding = None
        
        # ipdb.set_trace() # check the forward
        all_processed_feat = []
        for i, curr_feat_x in enumerate(x):
            # TODO: reshape each features
            if len(curr_feat_x.shape) == 5:
                merge_BL = True
                B, L, Z, X, Y = curr_feat_x.shape
                curr_feat_x = curr_feat_x.reshape(-1, Z, X, Y)
            else:
                merge_BL = False
            
            # feature Forward to the model
            curr_feat_x = self.backbone[i](curr_feat_x)
            
            # reshape one by one reshape from torch.Size([B* L, D]) back to torch.Size([B, L, D])
            if merge_BL:
                curr_feat_x = curr_feat_x.reshape(B, L, self.conv_out_dim[i])
            
            all_processed_feat.append(curr_feat_x)
            
        # TODO: handle the concatenation of x, make it also compatible with the old design
        # ipdb.set_trace() # check the concat ans special handle
        if len(all_processed_feat) == 1:
            x = all_processed_feat[0]
        else:
            x = torch.cat(all_processed_feat, dim=-1)
        
        
        # the linear layer for mapping the dimension
        # if self.scaling_layer is not None:
        #     x = self.scaling_layer(x)
        
        # concat the detection result embedding with feature x
        if det_embedding is not None and self.det_info_fuse_method == 'early':
            x = torch.cat([x, det_embedding], dim=-1)
            # ipdb.set_trace()
        
        # the ssm block
        # convert the dim from  (B* L, D) to (B, L, D)
        residual = None
        mask = None
        # conv_state, ssm_state = prev_action_state
        new_action_state = []
        temporal = x
        for (old_conv_state, old_ssm_state), layer in zip(prev_action_state, self.layers):
            temporal, residual, new_conv_state, new_ssm_state = layer.step(
                temporal, residual, old_conv_state, old_ssm_state, 
            )
            new_action_state.append((new_conv_state, new_ssm_state))
        
        # for the normalization after ssm
        if not self.fused_add_norm:
            residual = (temporal + residual) if residual is not None else temporal
            temporal = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            temporal = fused_add_norm_fn(
                temporal,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # may concat the 3d feature with the temporal feature
        if self.use_3d_feat_pred:
            if self.feat_use_method == 'concat':
                x = torch.cat([x, temporal], dim=-1)
            elif self.feat_use_method == 'add':
                x = x + temporal
            else:
                raise NotImplementedError     
        else:
            x = temporal       
        
        all_feature_list = [x, ]
        # x: torch.Size([1, 198, 512])
        # latency: should be in the shape of (B, L)
        # using learnable latency_tokens then it should be torch.Size([1, 198, 512])
        if self.use_contention:
            assert latency is not None 
            if self.learnable_contention_token: # ensure the table should be (#token, #dim)
                if self.contention_token_type == 'pe':
                    # TODO: figure out the index
                    # ipdb.set_trace()
                    all_index = torch.zeros(latency.shape)
                    # map each value
                    all_index[latency==0.0] = 0
                    all_index[latency==0.2] = 1
                    all_index[latency==0.5] = 2
                    all_index[latency==0.9] = 3                    
                    
                    # all_index = (latency - self.latency_thresholds[0]).int()
                    #print('all_index:', all_index, torch.max(all_index), torch.min(all_index))
                    all_index = all_index.to(dtype=torch.int)
                    all_tokens = self.contention_tokens[all_index]
                    #print('all_tokens:', all_tokens.shape, 'x:', x.shape)
                    all_feature_list.append(all_tokens)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        # ipdb.set_trace() # control the dimenstion
        if det_embedding is not None and self.det_info_fuse_method == 'late':
            all_feature_list.append(det_embedding)
        
        x = torch.cat(all_feature_list, dim=-1)
        x = self.fc(x) # return will be (B, L, D)

        # assuming that our qvalue should be proportional to the mAP value
        # the output should between [0,1]
        if self.use_sigmoid:
            x = F.sigmoid(x)

        return x, new_action_state



class SSM_contention_two_branches(nn.Module):
    '''
        This version of the model aims to handle the contention input but ignore the latency slo
        the RL model which has three parts:
        1. The backbone which encode the 3D feature into a vector
        2. The State space model (SSM) which encodes the temporal information
        3. A linear layer/MLP to prediction latency binary cls
        4. A linear layer/MLP to prediction mAP binary cls
    '''
    def __init__(self, 
                # backbone config
                inplanes, # the z dimension of the input
                planes=64, 
                num_layers=4, 
                backbone_name='conv',
                input_size=[150, 150], # the x,y dimension of the infput
                out_dim=512, # the output dimension of the backbone
                # ssm config
                ssm_layers=1,
                ssm_cfg=None,
                norm_epsilon: float=1e-5,
                rms_norm: bool=False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                use_bidir_mamba=False,
                # linear layer config
                num_classes=40, 
                use_mlp=False,
                embd_dim=None, 
                # latency config
                use_contention=None,
                learnable_contention_token=False,
                contention_token_type='vec',
                contention_embed_dim=None,
                contention_levels=None,
                
                # other model config
                critic=False,
                use_3d_feat_pred=False, # whether we will use feature after 3d backbone for action pred
                feat_use_method='concat', # concat or add
                use_det_info=False, # whether use previouse detection result in the training
                use_det_query=False,
                det_res_dim=7,
                det_res_embed_dim=64,
                det_info_fuse_method='late', # the 'late' means fusion this after SSM, the 'early' means fusion this before the SSM
                num_self_attn_layer=1,
                ):
        super().__init__()
        #### determine whether this is a critic net
        self.critic = critic
        # prepare the hyper
        self.use_det_info = use_det_info
        self.use_det_query = use_det_query
        self.det_info_fuse_method = det_info_fuse_method
        
        ######################  init the backbone ########################################
        if not isinstance(backbone_name, list):
            self.backbone_name = [backbone_name]
        else:
            self.backbone_name = backbone_name
            
        self.backbone = nn.ModuleList()
        self.conv_out_dim = []
        for b_name in self.backbone_name:
            if b_name == 'conv':
                backbone_config_dict = {'num_layers':num_layers, 'inplanes': inplanes, 'planes': planes}
                self.backbone.append(make_backbone('conv', **backbone_config_dict))
                self.conv_out_dim.append(planes * 8 if num_layers == 4 else planes * 4)
                continue
            elif b_name == 'basevoxel':
                backbone_config_dict = {'inplanes': inplanes, 'input_size': input_size, 'planes': planes, 'out_dim': out_dim}
                self.backbone.append(make_backbone('basevoxel', **backbone_config_dict))
                self.conv_out_dim.append(out_dim)
                continue
            elif b_name == 'identity':
                self.backbone.append(make_backbone('identity', **{}))
                self.conv_out_dim.append(out_dim)
                continue
            else:
                raise NotImplementedError

        #### for the case that we need extra layer to match the dimension of the SSM
        if embd_dim is None:
            self.embd_dim = sum(self.conv_out_dim)
        else:
            self.embd_dim = embd_dim
        
        # add additional dimension for the detection result
        if self.use_det_info and self.det_info_fuse_method == 'early':
            self.embd_dim += det_res_embed_dim

        # additional mapping from the backbone feature dimenstion to the embeding dimension
        # if self.conv_out_dim != self.embd_dim:
        #     self.scaling_layer = nn.Linear(self.conv_out_dim, self.embd_dim)
        # else:
        #     self.scaling_layer = None
    
        #### init the SSM
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.use_bidir_mamba = use_bidir_mamba
        self.layers = nn.ModuleList()
        for idx in range(ssm_layers):
            self.layers.append(
                create_block(
                    self.embd_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=idx,
                    use_bidir_mamba=self.use_bidir_mamba,
                    **factory_kwargs,
                )
            )
        # define the final norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
                self.embd_dim, eps=norm_epsilon, **factory_kwargs
            )
        
        ### handle the learnable contention
        self.learnable_contention_token = learnable_contention_token
        self.contention_token_type = contention_token_type
        self.contention_levels = contention_levels
        self.contention_embed_dim = contention_embed_dim if contention_embed_dim is not None else self.embd_dim
        if self.learnable_contention_token:
            if self.contention_token_type == 'pe':
                assert self.contention_levels is not None
                # assert len(self.latency_thresholds) == 2
                num_of_thresh = len(self.contention_levels)
                 # call the func to generate the pe table
                self.contention_tokens = get_sinusoid_encoding(num_of_thresh, self.contention_embed_dim).squeeze(dim=0).permute([1,0]).cuda() # torch.Size([1, 512, 10]) -> torch.Size([dim, token_num]) -> torch.Size([token_num, dim])
            else:
                raise NotImplementedError
        
        # handle the special control for the skip connection
        self.use_contention = use_contention
        self.use_3d_feat_pred = use_3d_feat_pred
        self.feat_use_method = feat_use_method
        
        if not self.use_3d_feat_pred:
            feat_dims = self.embd_dim
        else:
            if self.feat_use_method == 'concat':
                feat_dims = self.embd_dim * 2
            elif self.feat_use_method == 'add':
                feat_dims = self.embd_dim
            else:
                raise NotImplementedError
        
        ### calculate the input dimension of the linear layer
        if self.use_contention and self.learnable_contention_token:
            self.final_embd_dim = feat_dims + self.contention_embed_dim
        else:
            self.final_embd_dim = feat_dims
            
        # init the detection embed modules
        if use_det_info:
            self.det_fc = MLP(det_res_dim, det_res_embed_dim // 2, det_res_embed_dim, 2)  
            self.det_norm = nn.LayerNorm(det_res_embed_dim)
            self.num_self_attn_layer = num_self_attn_layer
            if self.use_det_query: # use transformer and query tokens to extract the feature
                self.det_query = nn.Parameter(torch.randn(1, det_res_embed_dim))
                
                self.det_self_attn = nn.ModuleList()
                for i in range(self.num_self_attn_layer):
                    self.det_self_attn.append(
                            torch.nn.MultiheadAttention(det_res_embed_dim, 2, batch_first=True)
                        )
            
            if self.det_info_fuse_method == 'late':
                self.final_embd_dim += det_res_embed_dim 
        
        # define the classification layer the MLP
        self.use_mlp = use_mlp
        output_dim = 1 if self.critic else num_classes
        if self.use_mlp:
            self.map_fc = MLP(self.final_embd_dim, self.final_embd_dim//2, output_dim, 2)
            self.lat_fc = MLP(self.final_embd_dim, self.final_embd_dim//2, output_dim, 2)
        else:
            self.map_fc = nn.Linear(self.final_embd_dim, output_dim)
            self.lat_fc = nn.Linear(self.final_embd_dim, output_dim)

        # set bias term to 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    if not getattr(m.bias, "_no_reinit", False):
                        nn.init.zeros_(m.bias)

    def get_det_res_stat(self, detect_info):
        # all_bboxes_coordinate: [B(1), T, N, 7] torch.Size([1, 1, 50, 7])
        # all_bbox_confidences: [B(1), T, N] torch.Size([1, 1, 50])
        # all_bbox_categories: [B(1), T, N] torch.Size([1, 1, 50])
        #ipdb.set_trace()
        all_bboxes_coordinate, all_bbox_categories, all_bbox_confidences = detect_info
        
        if all_bboxes_coordinate is None: # which is the first step of the loop
            return [[0.0]*7]
        
        assert all_bboxes_coordinate.shape[0] == 1
        timesteps = all_bboxes_coordinate.shape[1]
        all_stat = []
        
        # calculate for each step
        for i in range(timesteps):
            # each step is [mean_of_conf, std_of_conf, 
            # count_of_car, count_of_person, count_of_bike, 
            # mean_of_size, std_of_size]
            
            curr_corrdinate = all_bboxes_coordinate[0, i]
            curr_categories = all_bbox_categories[0, i]
            curr_confidences = all_bbox_confidences[0, i]
            
            # calculate the mean and std of the confidence
            confidence_mean = torch.mean(curr_confidences)
            confidence_std = torch.std(curr_confidences)
            
            # count the number of each category
            count_of_car = torch.sum(curr_categories==1)
            count_of_person = torch.sum(curr_categories==2)
            count_of_bike = torch.sum(curr_categories==3)
            
            # calulate the size of all bboxes
            all_x_range = torch.abs(curr_corrdinate[:, 0] - curr_corrdinate[:, 3])
            all_y_range = torch.abs(curr_corrdinate[:, 1] - curr_corrdinate[:, 4])
            all_z_range = torch.abs(curr_corrdinate[:, 2] - curr_corrdinate[:, 5])
            all_size = all_x_range * all_y_range * all_z_range
            
            # calculate the mean and std of the bbox size
            size_mean = torch.mean(all_size)
            size_std = torch.std(all_size)
            
            # turn it into a vector
            stat_of_curr_step = [confidence_mean, confidence_std, 
                                 count_of_car, count_of_person, count_of_bike, 
                                 size_mean, size_std]
            all_stat.append(stat_of_curr_step)
        #ipdb.set_trace()
        return all_stat    

    def forward(self, x, extra_info=None, detect_info=None): 
        # input x: torch.Size([B, L, 62, 300, 300])
        # extra_info should be in the shape of (B, L) should be the value of the current latency threshold
        # ipdb.set_trace() # check the forward
        if detect_info is not None and None not in detect_info:
            if self.use_det_query:
                # ipdb.set_trace() # check whethet the shape of the self-attention input # torch.Size([1, 1, 50, 64])
                all_embedded_res = []
                
                all_bboxes_coordinate, all_bbox_categories, all_bbox_confidences = detect_info
                total_time_step = len(all_bboxes_coordinate)
                # padd one empty at the begining 
                first_detect_res = (torch.zeros(1,1,1,7), torch.zeros(1,1,1), torch.zeros(1,1,1))
                first_detect_res = torch.cat([first_detect_res[0], first_detect_res[1].unsqueeze(dim=-1), first_detect_res[2].unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                first_det_embedding = self.det_fc(first_detect_res) # torch.Size([1, 1, 50, 64])
                B, T, Num_Det, D = first_det_embedding.shape
                first_det_embedding = first_det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])      
                first_det_input = torch.cat([first_det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)          
                for layer in self.det_self_attn:
                    first_det_input, _ = layer(first_det_input, first_det_input, first_det_input) # B*T, N+1, D
                first_det_embedding = first_det_input[:, -1].view(B, T, D)                
                all_embedded_res.append(first_det_embedding)
                
                # pop the last one.
                for i in range(total_time_step):
                    curr_coord, curr_cate, curr_confi = all_bboxes_coordinate[i], all_bbox_categories[i], all_bbox_confidences[i]
                    if i < total_time_step - 1:
                        curr_detect_res = torch.cat([curr_coord, curr_cate.unsqueeze(dim=-1), curr_confi.unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                        curr_det_embedding = self.det_fc(curr_detect_res) # torch.Size([1, 1, 50, 64])
                        B, T, Num_Det, D = curr_det_embedding.shape
                        curr_det_embedding = curr_det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])  
                        curr_det_input = torch.cat([curr_det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)     
                        for layer in self.det_self_attn:
                            curr_det_input, _ = layer(curr_det_input, curr_det_input, curr_det_input) # B*T, N+1, D
                        curr_det_embedding = curr_det_input[:, -1].view(B, T, D)   
                        all_embedded_res.append(curr_det_embedding)
                
                # concat along the temporal axis
                det_embedding = torch.cat(all_embedded_res, dim=1)
                # ipdb.set_trace() # check the det_embedding again
            else:
                # TODO: this branch may have issue since we remove the concatention
                det_stat = self.get_det_res_stat(detect_info)
                # remove the last one
                det_stat.pop(-1)
                # pad zero at the beginning
                det_stat.insert(0, [0.0]*7)
                det_stat = torch.tensor(det_stat).unsqueeze(dim=0).to(x[0].device)
                det_embedding = self.det_fc(det_stat)
                # pass through a norm
                det_embedding = self.det_norm(det_embedding)
        else:
            det_embedding = None
            
        # ipdb.set_trace() # check the forward
        all_processed_feat = []
        for i, curr_feat_x in enumerate(x):
            # TODO: reshape each features
            if len(curr_feat_x.shape) == 5:
                merge_BL = True
                B, L, Z, X, Y = curr_feat_x.shape
                curr_feat_x = curr_feat_x.reshape(-1, Z, X, Y)
            else:
                merge_BL = False
            
            # feature Forward to the model
            curr_feat_x = self.backbone[i](curr_feat_x)
            
            # reshape one by one reshape from torch.Size([B* L, D]) back to torch.Size([B, L, D])
            if merge_BL:
                curr_feat_x = curr_feat_x.reshape(B, L, self.conv_out_dim[i])
            
            all_processed_feat.append(curr_feat_x)
            
        # handle the concatenation of x, make it also compatible with the old design
        # ipdb.set_trace() # check the concat ans special handle
        if len(all_processed_feat) == 1:
            x = all_processed_feat[0]
        else:
            x = torch.cat(all_processed_feat, dim=-1)
        
        # the linear layer for mapping the dimension
        # if self.scaling_layer is not None:
        #     x = self.scaling_layer(x)
        
        # concat the detection result embedding with feature x
        if det_embedding is not None and self.det_info_fuse_method == 'early':
            x = torch.cat([x, det_embedding], dim=-1)
        
        # the ssm block
        # convert the dim from  (B* L, D) to (B, L, D)
        residual = None
        mask = None
        for layer in self.layers:
            temporal, residual, mask = layer(
                x, residual, 
                inference_params=None,
                mask=mask,
                ctrl_vec=None,
            )
        
        # for the normalization after ssm
        if not self.fused_add_norm:
            residual = (temporal + residual) if residual is not None else temporal
            temporal = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            temporal = fused_add_norm_fn(
                temporal,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # may concat the 3d feature with the temporal feature
        if self.use_3d_feat_pred:
            if self.feat_use_method == 'concat':
                x = torch.cat([x, temporal], dim=-1)
            elif self.feat_use_method == 'add':
                x = x + temporal
            else:
                raise NotImplementedError     
        else:
            x = temporal       
            
        # x: torch.Size([1, 198, 512])
        # extra_info should be in the shape of (B, L)
        # using learnable latency_tokens then it should be torch.Size([1, 198, 512])
        all_feature_list = [x, ]
        
        if self.contention_levels:
            assert extra_info is not None 
            if self.learnable_contention_token: # ensure the table should be (#token, #dim)
                if self.contention_token_type == 'pe':
                    # TODO: find the index
                    # ipdb.set_trace()
                    all_index = torch.zeros(extra_info.shape)
                    # map each value
                    all_index[extra_info==0.0] = 0
                    all_index[extra_info==0.2] = 1
                    all_index[extra_info==0.5] = 2
                    all_index[extra_info==0.9] = 3
                    all_index = all_index.to(dtype=torch.int)
                    # all_index = []
                    # for ele in +
                    # index = my_list.index(3)
                    # all_index = (extra_info - self.latency_thresholds[0]).int()
                    #print('all_index:', all_index, torch.max(all_index), torch.min(all_index))
                    all_tokens = self.contention_tokens[all_index]
                    #print('all_tokens:', all_tokens.shape, 'x:', x.shape)
                    all_feature_list.append(all_tokens)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        if det_embedding is not None and self.det_info_fuse_method == 'late':
            all_feature_list.append(det_embedding)
                
        x = torch.cat(all_feature_list, dim=-1)
        # prepare the raw output
        map_x = self.map_fc(x) # return will be (B, L, D)
        lat_x = self.lat_fc(x)
        # prepare the prob
        map_prob = F.sigmoid(map_x)
        lat_prob = F.sigmoid(lat_x)
        final_prob = map_prob * lat_prob

        return (map_x, lat_x), final_prob
    
    def step(self, x, latency, detect_res=None, prev_action_state=None): 
        # input: x: torch.Size([B, 1, 62, 300, 300])
        #        latency: torch.Size([B, 1])
        #        detect_res: ? 
        #        prev_action_state: ?
        # ipdb.set_trace()
        if detect_res is not None:
            if self.use_det_query: # concat the feature and query and send it into the model, take the last one
                # ipdb.set_trace() # check dimension
                # all_bboxes_coordinate: [B(1), T, N, 7] torch.Size([1, 1, 50, 7]) # B = 1, T = 1, detection result = 50, dim = 7
                # all_bbox_confidences: [B(1), T, N] torch.Size([1, 1, 50])
                # all_bbox_categories: [B(1), T, N] torch.Size([1, 1, 50])
                # TODO: encode each detection result
                # concat and embeding all the information
                if None in detect_res: # do the padding if no detection result
                    detect_res = (torch.zeros(1,1,1,7), torch.zeros(1,1,1), torch.zeros(1,1,1))
                
                detect_res = torch.cat([detect_res[0], detect_res[1].unsqueeze(dim=-1), detect_res[2].unsqueeze(dim=-1)], dim=-1).to(x[0].device) # torch.Size([1, 1, 50, 9])
                det_embedding = self.det_fc(detect_res) # torch.Size([1, 1, 50, 64])
                B, T, Num_Det, D = det_embedding.shape
                det_embedding = det_embedding.view(-1, Num_Det, D) # torch.Size([1, 50, 64])
                
                all_det_input = torch.cat([det_embedding, self.det_query.unsqueeze(dim=0)], dim=1)
                # ipdb.set_trace() # check dimension
                for layer in self.det_self_attn:
                    all_det_input, _ = layer(all_det_input, all_det_input, all_det_input) # B*T, N+1, D
                det_embedding = all_det_input[:, -1].view(B, T, D)
                # ipdb.set_trace() # check dimension again
            else:
                det_stat = self.get_det_res_stat(detect_res)  #
                det_stat = torch.tensor(det_stat).unsqueeze(dim=0).to(x.device)  # 
                det_embedding = self.det_fc(det_stat)
        else:
            det_embedding = None
        
        # ipdb.set_trace() # check the forward
        all_processed_feat = []
        for i, curr_feat_x in enumerate(x):
            # TODO: reshape each features
            if len(curr_feat_x.shape) == 5:
                merge_BL = True
                B, L, Z, X, Y = curr_feat_x.shape
                curr_feat_x = curr_feat_x.reshape(-1, Z, X, Y)
            else:
                merge_BL = False
            
            # feature Forward to the model
            curr_feat_x = self.backbone[i](curr_feat_x)
            
            # reshape one by one reshape from torch.Size([B* L, D]) back to torch.Size([B, L, D])
            if merge_BL:
                curr_feat_x = curr_feat_x.reshape(B, L, self.conv_out_dim[i])
            
            all_processed_feat.append(curr_feat_x)
            
        # TODO: handle the concatenation of x, make it also compatible with the old design
        # ipdb.set_trace() # check the concat ans special handle
        if len(all_processed_feat) == 1:
            x = all_processed_feat[0]
        else:
            x = torch.cat(all_processed_feat, dim=-1)
        
        
        # the linear layer for mapping the dimension
        # if self.scaling_layer is not None:
        #     x = self.scaling_layer(x)
        
        # concat the detection result embedding with feature x
        if det_embedding is not None and self.det_info_fuse_method == 'early':
            x = torch.cat([x, det_embedding], dim=-1)
            # ipdb.set_trace()
        
        # the ssm block
        # convert the dim from  (B* L, D) to (B, L, D)
        residual = None
        mask = None
        # conv_state, ssm_state = prev_action_state
        new_action_state = []
        temporal = x
        for (old_conv_state, old_ssm_state), layer in zip(prev_action_state, self.layers):
            temporal, residual, new_conv_state, new_ssm_state = layer.step(
                temporal, residual, old_conv_state, old_ssm_state, 
            )
            new_action_state.append((new_conv_state, new_ssm_state))
        
        # for the normalization after ssm
        if not self.fused_add_norm:
            residual = (temporal + residual) if residual is not None else temporal
            temporal = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            temporal = fused_add_norm_fn(
                temporal,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # may concat the 3d feature with the temporal feature
        if self.use_3d_feat_pred:
            if self.feat_use_method == 'concat':
                x = torch.cat([x, temporal], dim=-1)
            elif self.feat_use_method == 'add':
                x = x + temporal
            else:
                raise NotImplementedError     
        else:
            x = temporal       
        
        all_feature_list = [x, ]
        # x: torch.Size([1, 198, 512])
        # latency: should be in the shape of (B, L)
        # using learnable latency_tokens then it should be torch.Size([1, 198, 512])
        if self.use_contention:
            assert latency is not None 
            if self.learnable_contention_token: # ensure the table should be (#token, #dim)
                if self.contention_token_type == 'pe':
                    # TODO: figure out the index
                    # ipdb.set_trace()
                    all_index = torch.zeros(latency.shape)
                    # map each value
                    all_index[latency==0.0] = 0
                    all_index[latency==0.2] = 1
                    all_index[latency==0.5] = 2
                    all_index[latency==0.9] = 3                    
                    
                    # all_index = (latency - self.latency_thresholds[0]).int()
                    #print('all_index:', all_index, torch.max(all_index), torch.min(all_index))
                    all_index = all_index.to(dtype=torch.int)
                    all_tokens = self.contention_tokens[all_index]
                    #print('all_tokens:', all_tokens.shape, 'x:', x.shape)
                    all_feature_list.append(all_tokens)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        # ipdb.set_trace() # control the dimenstion
        if det_embedding is not None and self.det_info_fuse_method == 'late':
            all_feature_list.append(det_embedding)
        
        x = torch.cat(all_feature_list, dim=-1)
        # prepare the raw output
        map_x = self.map_fc(x) # return will be (B, L, D)
        lat_x = self.lat_fc(x)
        # prepare the prob
        map_prob = F.sigmoid(map_x)
        lat_prob = F.sigmoid(lat_x)
        final_prob = map_prob * lat_prob

        return (map_x, lat_x), final_prob, new_action_state