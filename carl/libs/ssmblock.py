import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from .blocks import ConvBlock, get_sinusoid_encoding
from .ops import selective_scan_fn, mamba_inner_fn, selective_scan_ref


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        use_pytorch_implementation=False,
        use_recursive=False,
        layer_idx=None,
        device=None,
        dtype=None,
        bidir=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.use_pytorch_implementation = use_pytorch_implementation
        self.use_recursive = use_recursive
        self.bidir = bidir

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # temp for empty forward 
        self.empty_conv_state = torch.zeros((1, self.d_inner, self.d_conv))
        self.empty_ssm_state = torch.zeros((1, self.d_inner, self.d_state))

    def forward(self, hidden_states, inference_params=None, ctrl_vec=None, mask=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            if not self.use_pytorch_implementation:
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
            elif self.use_pytorch_implementation and self.use_recursive:
                y = selective_scan_ref(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )         
            else:
                raise NotImplementedError
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state=None, ssm_state=None):
        if conv_state is None:
            B, _, dim = hidden_states.shape
            None_true = True
            # d_conv = self.d_conv
            # d_state = self.d_state
            # create zero for the conv_state and ssm_state
            #print('just before the creation:')
            conv_state = self.empty_conv_state.repeat([B, 1, 1]).to(hidden_states.device)
            ssm_state = self.empty_ssm_state.repeat([B, 1, 1]).to(hidden_states.device)
            #print('after creation conv_state:', conv_state.shape, conv_state, 'ssm_state:', ssm_state.shape, ssm_state)        
        else:
            None_true = False
        
        # ssm_state should be (B, D, N)
        # conv_state should be (B, D, W)
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # from (B, 1, D) to (B, D) to (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dim=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        #if None_true:
        #    print('after the layer conv_state:', conv_state.shape, conv_state, 'ssm_state:', ssm_state.shape, ssm_state)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class BidirMambav2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        dropout=0.0, # for dropout layer
        use_ctrl_info=False, # for using the ctrl information
        trainable_ctrl=False,
        text_embedding_path=None,
        #control_dim=1024,
        #text_embed_dim=512,
        ctrl_merging_version='v2',
        use_abs_pe_for_ctrl=False,
        max_seq_len=None,
        map_ctrl_val=False, # for control with different dimension
        ctrl_dim=None,
        num_head=1, # how many B and C should be use for all hidden dimension
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # init the model hyper-params
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.num_head = num_head
        assert num_head < d_model
        assert d_model % num_head == 0 # assert the input dimension can be divided by the num_head

        # define input mapping layers
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # define activation layers
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # define just one conv for both forward and backward
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )  
    
        # the layer for producing forward delta, B, C 
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * self.num_head * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * self.num_head * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        # init the layer for the creating final delta
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        # S4D real initialization
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True) # (K=4, D, N)      
        # D "skip" parameter
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True) # (K=4, D, N)     
        
        # the layer project inner dim back to the input dimension
        self.out_norm = RMSNorm(self.d_inner)        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None        
        
        # for the control information
        self.use_ctrl_info = use_ctrl_info
        self.trainable_ctrl = trainable_ctrl
        self.text_embedding_path = text_embedding_path
        #self.control_dim = control_dim
        #self.text_embed_dim = text_embed_dim
        self.ctrl_merging_version = ctrl_merging_version
        self.use_abs_pe_for_ctrl = use_abs_pe_for_ctrl     # for pos embedding
        self.max_seq_len = max_seq_len                     # used in pos embedding
        self.map_ctrl_val = map_ctrl_val
        self.ctrl_dim = ctrl_dim
        
        if self.use_ctrl_info:
            if self.map_ctrl_val: # in the case that similar to transformer, generate the key val from the input
                assert self.ctrl_dim is not None
                # prepare the ctrl key and value mapping
                self.ctrl_val_mapping = nn.Linear(self.ctrl_dim, self.d_inner, bias=bias, **factory_kwargs)   
                self.ctrl_key_mapping = nn.Linear(self.ctrl_dim, self.d_state * self.num_head * 2, bias=bias, **factory_kwargs) # 2 for the bidrection
            else: # for the case that we only generate the key not the value
                self.ctrl_key_mapping = nn.Linear(self.d_inner, self.d_state * self.num_head * 2, bias=bias, **factory_kwargs) # 2 for the bidrection

            # prepare the control info posit embedding
            if self.use_abs_pe_for_ctrl:
                assert ctrl_merging_version == 'v2' # assert the contrl information is using v2 merging
                pos_embd = get_sinusoid_encoding(self.max_seq_len, self.d_inner) / (self.d_inner**0.5)
                self.register_buffer("pos_embd", pos_embd, persistent=False)       
        
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj


    def forward(self, hidden_states, inference_params=None, ctrl_vec=None, mask=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        ################ input mapping and the split the x, z We do matmul and transpose BLH -> HBL at the same time
        # the input project and bias
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x, z = xz.chunk(2, dim=1) # (B D L)

        # do the convolution, x is (B D L)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )            
            
        ################# the scanning #######################
        # repeat the x (b, 2, d, l)
        xs = torch.cat([x.unsqueeze(dim=1), x.flip(-1).unsqueeze(dim=1)], dim=1) 

        # preparing the delta, B, C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # split the result
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state*self.num_head, self.d_state*self.num_head], dim=2)
        # convert the delta from r dim to d dimension
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(batch, -1, seqlen) # (b, 2 * input_d, l)
        dts = dts.contiguous().float().view(batch, -1, seqlen) # (b, 2 * input_d, l)
        Bs = Bs.float().reshape(batch, 2 * self.num_head, -1, seqlen) # (b, 2, d_state * self.num_head, l) -> (b, 2 * self.num_head, d_state, l)
        Cs = Cs.float().reshape(batch, 2 * self.num_head, -1, seqlen) # (b, 2, d_state * self.num_head, l) -> (b, 2 * self.num_head, d_state, l)
        Ds = self.Ds.float().view(-1) # (2 * input_d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (2 * input_d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (2 * d)

        # do the bidirection scanning
        y_all = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, 2, -1, seqlen)

        # split the resilt 
        y_fw = y_all[:, 0] # (B, D, L)
        y_bw = y_all[:, 1].flip(-1) # (B, D, L) 
        y = y_fw + y_bw # (B, D, L) 
        
        # use the control info (v2) add in the encoding state:
        # y_t = C_t * B_t * u_t + C_t * A * B_(t-1) * u_(t-1) + C_t * (A**2) * B_(t-2) * u_(t-2) + ... +  C_t * (A**t) * B_0 * u_0 + D*u_t
        #       + C_t * B' * ctrl + C_t * A * B' * ctrl + C_t * (A**2) * * B' * ctrl + ... +  C_t * (A**t) *  B' * ctrl
        # use addditional scanning for: C_t * B' * ctrl + C_t * A * B' * ctrl + C_t * (A**2) * * B' * ctrl + ... +  C_t * (A**t) *  B' * ctrl
        # the problem of this version is: Bc is multiplied with delta_t which is related to input x,
        # but in fact we need this when we do the discretization
        
        if self.use_ctrl_info and self.ctrl_merging_version == 'v2':            
            ### xs will be control information, repeat the control vector
            if ctrl_vec.dim() == 2:  # if the control information is per sample
                assert ctrl_vec.shape[0] == batch
                #print('before expand: ctrl_vec', ctrl_vec.shape, ctrl_vec)
                cxs = ctrl_vec.unsqueeze(dim=1)
                cxs = cxs.repeat(1, seqlen, 1) # repeat to (b, l, input_d)
                #print('after repeat and expand:', cxs.shape, cxs)
            elif ctrl_vec.dim() == 1:  # if the control information is the same for whole batch
                cxs = ctrl_vec.repeat(batch, seqlen, 1) # repeat to (b, l, input_d)
            else:
                raise NotImplementedError
            
            # training: using fixed length position embeddings
            if self.use_abs_pe_for_ctrl and self.training:
                # permute from (b, l, input_d) to (b, input_d, l)
                cxs = cxs.permute([0, 2, 1])
                assert seqlen <= self.max_seq_len, "Reached max length."
                pe = self.pos_embd
                # add pe to cxs
                cxs = cxs + pe[:, :, :seqlen] * mask.to(x.dtype)
                # permute from (b, input_d, l) to (b, l, input_d) 
                cxs = cxs.permute([0, 2, 1])                

            # inference: re-interpolate position embeddings for over-length sequences
            if self.use_abs_pe_for_ctrl and (not self.training):
                # permute from (b, l, input_d) to (b, input_d, l)
                cxs = cxs.permute([0, 2, 1])                
                
                if seqlen >= self.max_seq_len:
                    pe = F.interpolate(
                        self.pos_embd, seqlen, mode='linear', align_corners=False)
                else:
                    pe = self.pos_embd
                # add pe to cxs
                cxs = cxs + pe[:, :, :seqlen] * mask.to(x.dtype)
                # permute from (b, input_d, l) to (b, l, input_d) 
                cxs = cxs.permute([0, 2, 1])              

            if self.map_ctrl_val:
                # generate the key and value
                ctrl_key = self.ctrl_key_mapping(cxs) # the ctrl_key should be (b, l, 2 * input_d * self.num_head)
                cxs = self.ctrl_val_mapping(cxs) # the cxs should be (b, l, input_d)
            else: # generate the key only, the value is the control itself
                assert ctrl_vec.shape[-1] == self.d_inner
                # only create ctrl key in each layer
                ctrl_key = self.ctrl_key_mapping(cxs) # the ctrl_key should be (b, l, 2 * input_d * self.num_head)

            # repeat and permute for the bidirection
            cxs = cxs.repeat(1, 1, 2) # repeat to (b, l, 2 * input_d)
            cxs = cxs.permute([0, 2, 1]) # permute (b, l, 2 * input_d) to (b, 2 * input_d, l) 
            
            #cBs = ctrl_key.repeat(1, 1, 2) # repeat to (b, l, 2 * d_state)
            cBs = ctrl_key.permute([0, 2, 1]) # from (b, l, 2 * input_d * self.num_head) to (b, 2 * input_d * self.num_head, l)
            cBs = cBs.float().view(batch, 2 * self.num_head, -1, seqlen)
           
            # dts will remain the same, generated base one the x
            # C will be the same
            # A will be the same
            # dt_projs_bias will be the same
            # D do not need
            # do the bidirection scanning for control
            ctrl_all = selective_scan_fn(
                cxs,
                dts,
                As,
                cBs,
                Cs,
                D=None,
                z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(batch, 2, -1, seqlen)
        
            # split the resilt, reverse the second one
            ctrl_fw = ctrl_all[:, 0] # (B, D, L)
            ctrl_bw = ctrl_all[:, 1].flip(-1) # (B, D, L) 
            y = y + ctrl_fw + ctrl_bw # (B, D, L)         
        
        y = rearrange(y, "b d l -> b l d")
        
        # use the control info (v1) add in the decoding state:
        # y_t (scalor) = C_t * h_t + D * u_t + C_t * B_c * Ctrl (scalor)
        if self.use_ctrl_info and self.ctrl_merging_version == 'v1':   
            # only create ctrl key in each layer
            ctrl_key = self.ctrl_key_mapping(ctrl_vec)
            
            # create the kv
            qkvs = torch.einsum('b k s l, s, d -> b k l d', Cs, ctrl_key, ctrl_vec)
            
            # sum the value of bidirection
            cross_attn_fw = qkvs[:, 0]
            cross_attn_bw = qkvs[:, 1]
            y = y + cross_attn_fw + cross_attn_bw
        
        # output norm use rms norm
        y = self.out_norm(y)
        # gating
        y = y * F.silu(z.permute(0,2,1)) # z from (B, D, L) to  (B, L, D)
        # output project
        out = self.out_proj(y)
        # drop out
        if self.dropout is not None:
            out = self.dropout(out)        
        return out


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, 
        residual_in_fp32=False, scale_factor=False, use_interpolate=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            
        # for the downsample
        self.scale_factor = scale_factor
        self.use_interpolate = use_interpolate
        # create the convnet for downsample
        if self.scale_factor is not None:
            self.down_sample_hidden_state = ConvBlock(dim, 3, self.scale_factor)
            if not self.use_interpolate:
                self.down_sample_residual = nn.AvgPool1d(3, stride=self.scale_factor, padding=3//2)
            else:
                self.down_sample_residual = F.interpolate
        else:
            self.down_sample_hidden_state = None
            self.down_sample_residual = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, 
        inference_params=None, mask=None, ctrl_vec=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, ctrl_vec=ctrl_vec, mask=mask)
        
        return hidden_states, residual, mask

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def step(self, hidden_states, residual, conv_state, ssm_state):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        #print('before conv_state ssm_state:', conv_state, ssm_state)
        hidden_states, conv_state, ssm_state = self.mixer.step(hidden_states, conv_state, ssm_state)
        #print('After conv_state ssm_state:', conv_state, ssm_state)
        return hidden_states, residual, conv_state, ssm_state


class RLBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, 
        residual_in_fp32=False, 
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states, residual, conv_state, ssm_state, device
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        #print('before conv_state ssm_state:', conv_state, ssm_state)
        hidden_states, conv_state, ssm_state = self.mixer.step(hidden_states, conv_state, ssm_state, device)
        #print('After conv_state ssm_state:', conv_state, ssm_state)
        return hidden_states, residual, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
