import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(
            x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

class SublayerConnection(nn.Module):

    def __init__(self, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)

class fa_Block(nn.Module):  # Frequency Adaptive block
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)   # [dim,2]
        self.complex_weight_low = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)   # [dim,2]
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)   # [dim,2]
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight_low, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):  #[B,L_fft,D]

        B, L_fft, _ = x_fft.shape
        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions
        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)
        # 生成自适应掩码
        high_adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        low_adaptive_mask = 1-high_adaptive_mask
        # 拓展维度
        high_adaptive_mask = high_adaptive_mask.unsqueeze(-1)
        low_adaptive_mask = low_adaptive_mask.unsqueeze(-1)

        return high_adaptive_mask,low_adaptive_mask

    def forward(self, x_in):
        x_in = x_in.transpose(1,2)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        # Adaptive High Frequency Mask (no need for dimensional adjustments)
        high_freq_mask, low_freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        high_masked = x_fft * high_freq_mask.to(x.device)
        low_masked = x_fft * low_freq_mask.to(x.device)

        # 获取高低频权重
        weight_high = torch.view_as_complex(self.complex_weight_high)
        weight_low = torch.view_as_complex(self.complex_weight_low)

        # 加权
        x_weighted_high = high_masked * weight_high
        x_weighted_low = low_masked * weight_low

        x_weighted2 = x_weighted_high + x_weighted_low

        x_weighted = x_weighted2 * weight

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        x = x.transpose(1,2)

        return x

class Block(nn.Module):                       #FreConvNet
    def __init__(self, dmodel, dff, nvars, drop=0.1):

        super(Block, self).__init__()
        self.fab = fa_Block(dmodel)
        self.fab_norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)
        self.ffn_norm = nn.BatchNorm1d(dmodel)

        self.ffn_ratio = dff//dmodel
    def forward(self,x):
        input = x
        B, M, D, N = x.shape
        
        x = x.reshape(B*M,D,N)
        print('input_asb', x.shape)
        x = self.fab(x)
        print('after_asb', x.shape)
        x = self.fab_norm(x)
        #
        x = x.reshape(B,M,D,N)
        x = x.reshape(B,M*D,N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = x+input  # residual
        x = x.reshape(B*M,D, N)
        x = self.ffn_norm(x)
        x = x.reshape(B, M, D, N)
        return x

class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, dmodel, nvars,
                 small_kernel_merged=False, drop=0.1):
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block( dmodel=dmodel, dff=d_ffn, nvars=nvars, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCN(nn.Module):
    def __init__(self,patch, task_name,patch_size,patch_stride, downsample_ratio, ffn_ratio, num_blocks,num_stage, dims,
                 nvars, backbone_dropout=0.01, seq_len=512,  class_drop=0.01,class_num = 10):

        super(ModernTCN, self).__init__()

        self.patch = patch
        self.task_name = task_name
        self.class_drop = class_drop
        self.class_num = class_num
        self.seq_len = seq_len
        self.in_feat = nvars
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.num_stage = num_stage

        # Embedding
        self.downsample_layers = nn.ModuleList()
        # patch
        if self.patch == 'CF':
            stem = nn.Sequential(
            nn.Conv1d(self.in_feat, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
            )
            self.downsample_layers.append(stem)

        #stem layer & down sampling layers
        elif self.patch == 'CI':
            stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
            )
            self.downsample_layers.append(stem)

        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        n_stride = (seq_len - patch_size) // self.patch_stride + 1
        self.n_padding = n_stride * self.patch_stride + self.patch_size - seq_len
        if self.patch_size!=self.patch_stride:
            self.num_patches = n_stride + 1
        else:
            self.num_patches = n_stride

        # backbone
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], dmodel=dims[stage_idx], nvars=nvars, drop=backbone_dropout)
            self.stages.append(layer)


        # head
        d_model = dims[self.num_stage-1]


        if self.task_name == 'classification':
            self.act_class = F.gelu
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)    # [batch,channel,1,1]
            self.class_dropout = nn.Dropout(self.class_drop)
            self.flatten = nn.Flatten()
            self.class_fc = nn.Linear(d_model,self.class_num) # [bs,classes]



    def forward_feature(self, x, te=None):

        B,M,L=x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            print('stage',i)
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i==0:
                if self.patch_size != self.patch_stride:
                    pad = x[:, :, -1:].repeat(1, 1, self.n_padding)
                    x = torch.cat([x, pad], dim=-1)
                _, _D, _N = x.shape
                x = self.downsample_layers[i](x)
                _, D_, N_ = x.shape
                x = x.reshape(B, M, D_, N_)
                x = self.stages[i](x)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
                x= self.downsample_layers[i](x)
                _, D_, N_ = x.shape
                x = x.reshape(B, M, D_, N_)
                x = self.stages[i](x)

        return x

    def classification(self,x):
        batch = x.shape[0]
        x = self.forward_feature(x,te=None)      # [bs,channel,d_encoder,patch_num]
        x = self.act_class(x)
        x = x.permute(0,2,1,3)
        x = self.global_avgpool(x)
        x = self.class_fc(x.reshape(batch, -1))
        x = self.class_dropout(x)
        return x


    def forward(self, x, te=None):

        if self.task_name == 'classification':
            x = self.classification(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param
        self.patch = configs.patch_method
        self.task_name = configs.task_name
        self.downsample_ratio = configs.downsample_ratio   #
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks   # num_block in each stage
        self.num_stage = configs.num_stage
        self.dims = configs.dims
        self.nvars = configs.data_shape[1]
        self.drop_backbone = configs.dropout
        self.seq_len = configs.data_shape[0]

        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        #classification
        self.class_dropout = configs.class_dropout
        self.class_num = configs.num_class
        self.act_class = F.gelu
        self.model = ModernTCN(patch=self.patch,task_name=self.task_name,patch_size=self.patch_size, patch_stride=self.patch_stride,
                            downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks,num_stage=self.num_stage,
                            dims=self.dims,nvars=self.nvars,backbone_dropout=self.drop_backbone,seq_len=self.seq_len,
                            class_drop = self.class_dropout, class_num = self.class_num)
    def forward(self, x):   # [batch,length,channel]
        x = x.permute(0, 2, 1)
        te = None
        x = self.model(x, te)
        return x