"""
Efficient and Explicit Modelling of Image Hierarchies for Image Restoration
Image restoration transformers with global, regional, and local modelling
A clean version of the.
Shared buffers are used for relative_coords_table, relative_position_index, and attn_mask.
"""

"""


Returns:
    _type_: _description_
"""


#Conv和Encoder相+，然后拼接
#每层融合模块不相同

import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper
from omegaconf import OmegaConf
from componets.grl import Upsample, UpsampleOneStep
from componets import *
from componets.grl.mixed_attn_block_efficient import (
    _get_stripe_info,
    EfficientMixAttnTransformerBlock,
    
)

from componets.grl.swin_v1_block import Mlp

from componets.grl.ops import (
    bchw_to_blc,
    blc_to_bchw,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
)
from componets.grl.swin_v1_block import (
    build_last_conv,
)
from timm.models.layers import to_2tuple, trunc_normal_

# from componets.grl.rcb import ResidualGroup
from componets.grl.rcb3 import ResidualGroup2, ResidualGroup3



from componets.grl.DCA import DCAT

from componets.swim.swim_generator import BilinearUpsample

from componets.swim.swim import PatchMerging

#Encoder中融合,Conv提取后Encoder后残差
class MRF(nn.Module):
    def __init__(self, nf, n_blks=[1, 1, 1, 1, 1, 1]):
        super(MRF, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_a = nn.Conv2d(nf, nf, 3, 1, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_b = nn.Conv2d(nf, nf, 3, 1, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)


    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out
    
class Encoder(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Encoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))

        return [fea_L1, fea_L2, fea_L3]


class Middle(nn.Module):
    def __init__(self, nf, n_blks=[1, 1, 1, 1, 1, 1]):
        super(Middle, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_L2 = make_layer(block, n_blks[1])

        self.conv_L1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_blks[2])

        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        fea_L3 = self.act(self.conv_L3(x))
        fea_L3 = self.blk_L3(fea_L3)

        fea_L2 = self.act(self.conv_L2(fea_L3))
        fea_L2 = self.blk_L2(fea_L2)
        fea_L2_up = F.interpolate(fea_L2, scale_factor=2, mode='bilinear', align_corners=False)

        fea_L1 = self.act(self.conv_L1(torch.cat([fea_L2_up, x], dim=1)))
        fea_L1 = self.blk_L1(fea_L1)

        return fea_L1
    

class SAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out

    
    
class BilinearUpsample1(nn.Module):
    """ BilinearUpsample Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, dim, out_dim=None):
        # print('inpur: ', input_resolution)
        # print('dim: ', dim)
        # print('out dim: ', out_dim)
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0, init_size=out_dim // 2)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        if x.dim() == 3:
            H, W = x_size
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            assert C == self.dim, "wrong in PatchMerging"
            # out_size
            x = x.view(B, H, W, -1)
            x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        else:
            B, C, H, W = x.shape
            L = H*W
            
            
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # x = self.upsample(x)
        # x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W*4, C)   # B,H,W,C
        # x = self.norm(x)
        # x = self.reduction(x)

        # Add SPE    
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha#B,C,H/2,W/2
        # x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x
    
class DCATMRSR(nn.Module):

    def __init__(
        self,
        lr_mlp=0.01,
        img_size=64,
        in_channels=3,
        out_channels=None,
        depth = 4,
        num_head = 8,
        # num_heads=[4, 4, 4, 4, 4, 4, 4, 4, 4],
        unet_narrow = 0.5,
        channel_multiplier=1,
        
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        conv_type="1conv",
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        euclidean_dist=False,
        style_dim=512,
        patch_norm=True,
        patch_size=1,
        ape=True,
        mlp_ratio=4.0,
        sft_half=True,
        embed_dim=None,
        **kwargs
        
    ):
        super(DCATMRSR, self).__init__()
        # Process the input arguments
        out_channels = out_channels or in_channels
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.depth = depth
        self.sft_half = sft_half
        
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
    
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }
        for key in channels.keys():
            channels[key] = embed_dim
            print('dim: ', embed_dim)
            
        n_blks = [2,2,2]
        
        in_channels = channels[f'{img_size}']

        #####################################################################################################
        ################################### 1, Tar/Ref LR feature extraction ###################################
        self.conv2d = Conv2D(in_chl=self.in_channels, nf=in_channels,  act='leakyrelu')
        
        
        #####################################################################################################
        ################################### 2, Encoder ######################################
        self.end = int(math.log(img_size, 2))
        self.start = self.end-depth
        self.depths=[2] * 12
        
        self.split_size_0 = [1,1,1,2,2,4,4,8,8,16]
        self.split_size_1 = [1,2,4,4,8,8,16,16,32,32]
        
        self.patch_norm = patch_norm
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=in_channels,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, in_channels))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        patches_resolution = self.patch_embed.patches_resolution
        self.input_resolution = to_2tuple(img_size)
        self.num_layers = len(self.depths)

        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # Body of the network
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        
        self.fusion_layers = nn.ModuleList()
        
        # output_resolutions = []
        self.input_blocks = nn.ModuleList()
        self.input_resolution = img_size
        downsample = None
        output_resolution = self.input_resolution
        for i_layer in range(self.end, self.start, -1):
            in_channels = channels[f'{2**i_layer}']
            out_channels = channels[f'{2**(i_layer)}']
            if i_layer != self.end:
                downsample = PatchMerging
                output_resolution = output_resolution // 2

            layer = ResidualGroup2(
                dim=in_channels,
                reso=output_resolution,
                num_heads=num_head,
                drop_paths=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                split_size_0=self.split_size_0[i_layer],
                downsample=downsample,
                split_size_1=self.split_size_1[i_layer])
            
            self.input_blocks.append(layer)
            
            fusion = DCAT(input_size=output_resolution*output_resolution, hidden_size=out_channels, proj_size=out_channels, num_heads=num_head)
            self.fusion_layers.append(fusion)
            self.input_resolution = output_resolution
            # output_resolutions.append(input_resolution)
            
        self.norm = norm_layer(out_channels)
        self.middle_layer = MRF(out_channels)
        # self.middle_after = Middle(out_channels, n_blks)
        # nn.Conv2d(out_channels*2, out_channels, 3, 1, 1)

        
        # EqualLinear(out_channels*2, out_channels, bias=True, lr_mul=lr_mlp, activation='fused_lrelu')

        self.conditions = nn.ModuleList()
        # self.condition_shift = nn.ModuleList()
        for i in range(self.end, self.start, -1):
            out_channels = channels[f'{2**i}']
            sft_out_channels = out_channels
            self.conditions.append(
                (
                    nn.Sequential(
                        EqualConv2d(out_channels, out_channel=out_channels, kernel_size=3, stride=1, padding=1, bias=True, bias_init_val=0),
                        ScaledLeakyReLU(0.2),
                        EqualConv2d(out_channels, sft_out_channels, kernel_size=3, stride=1, padding=1, bias=True, bias_init_val=0),               
                    )
                )
            )
            
        
        #####################################################################################################
        ################################### 3, Decoder ######################################
        # self.merge = nn.ModuleList()
        self.upsample = BilinearUpsample1(embed_dim, out_dim=embed_dim)
        self.pdA = SAM(embed_dim, use_residual=True, learnable=True)
        self.merge = MRF(embed_dim, embed_dim)
        # nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
        
        self.act = nn.ReLU(inplace=True)
        self.output_blocks = nn.ModuleList()
        self.pdAs = nn.ModuleList()
        
        self.merges = nn.ModuleList()
        # self.fusions = nn.Sequential(
        #     self.pdA = SAM(embed_dim, use_residual=True, learnable=True),
        #     self.merge = nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1),
            
        self.act = nn.ReLU(inplace=True)
        output_resolution = self.input_resolution * 2
        self.pdAs.append(SAM(embed_dim, use_residual=True, learnable=True))
        self.merges.append(nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1))
        for i_layer in range(self.start+2, self.end+1):
            in_channels = channels[f'{2**i_layer}']
            out_channels = channels[f'{2**(i_layer)}']
            
            print('split_size 0: ', self.split_size_0[i_layer])
            print('split_size 1: ', self.split_size_1[i_layer])
            layer = ResidualGroup3(
                dim=in_channels,
                reso=output_resolution,
                num_heads=num_head,
                drop_paths=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                split_size_0=self.split_size_0[i_layer],
                split_size_1=self.split_size_1[i_layer])
            
            self.output_blocks.append(layer)
            in_channels = out_channels
            
            output_resolution *= 2

            self.pdAs.append(SAM(embed_dim, use_residual=True, learnable=True))
            self.merges.append(nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1))
        ################################### 4, deep feature ######################################
        self.conv_deep =  Conv2D(in_channels, in_channels, n_blks=n_blks, act='leakyrelu')
        
        self.conv_last = nn.Conv2d(in_channels, 3, 1,1)    
                   
        self.apply(self._init_weights)

        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x


    def forward(self, tar, ref):
        H, W = tar.shape[2:]
        #shallow
        tar_lr = self.conv2d(tar)
        ref_0 = self.conv2d(ref)
        
        
        shallow = tar_lr
        
        #deep
        x_size = (tar_lr.shape[2], tar_lr.shape[3])
        tar_lr = self.patch_embed(tar_lr)
        ref_hr = self.patch_embed(ref_0) 
        
        if self.ape:
            tar_lr = tar_lr + self.absolute_pos_embed
            ref_hr = ref_hr + self.absolute_pos_embed
            
        tar_lr = self.pos_drop(tar_lr)
        ref_hr = self.pos_drop(ref_hr)
        
        tar_lr_o = tar_lr
        ref_hr_o = ref_hr
        conditions = []
        # input_resolution = self.input_resolution
        out_size = (self.img_size, self.img_size)
        for i_layer in range(self.depth):
            # block_size = self.lr_block_sizes[i_layer+1]
            layer = self.input_blocks[i_layer]
            fusion_layer = self.fusion_layers[i_layer]
            tar_lr, _ = layer(tar_lr, Ref=None, x_size=out_size)
            ref_hr, out_size = layer(ref_hr, Ref=None, x_size=out_size)
            
            b = tar_lr.size()[0]
            # if i_layer != self.depth-1:
            warp_ref_l = fusion_layer(tar_lr, ref_hr)
            
            warp_ref_l = blc_to_bchw(warp_ref_l, (out_size[0], out_size[1])).contiguous()
            # warp_ref_l = self.conditions[i_layer](warp_ref_l)#B,C,H,W
            conditions.insert(0,warp_ref_l)
                 
        x = blc_to_bchw(tar_lr, (out_size[0], out_size[1])).contiguous()
        # ref_hr = blc_to_bchw(ref_hr, (out_size[0], out_size[1])).contiguous()
        
        for i_layer in range(self.depth):
            warp_ref_l = conditions[i_layer]
            # pdA = self.
            warp_ref_l = self.pdAs[i_layer](warp_ref_l, x)
            x = self.act(self.merges[i_layer](torch.cat([warp_ref_l, x], dim=1)))
            x = x.flatten(2).transpose(1, 2)
            if i_layer != 0:
                layer = self.output_blocks[i_layer-1]
                x, _ = layer(x, scale=warp_ref_l, sft_half=self.sft_half, Ref=None, x_size=out_size)
                
            if i_layer != self.depth-1:
                x = self.upsample(x, out_size)
                out_size = (out_size[0]*2, out_size[1]*2)
            
        # for i_layer in range(self.depth-1):
        #     scale = conditions[i_layer]
        #     x = self.upsample(x, out_size)
        #     scale = self.pdA(scale, x)
        #     x = self.merge(tar_lr, ref_hr)
        #     x = x.flatten(2).transpose(1, 2)
        #     layer = self.output_blocks[i_layer]
        #     x, _ = layer(x, scale=scale, sft_half=self.sft_half, Ref=None, x_size=out_size)
        #     out_size = (out_size[0]*2, out_size[1]*2)
            
        x = blc_to_bchw(x, (out_size[0], out_size[1])).contiguous()
        deep = self.conv_deep(x)
        feat = deep + shallow
        res = self.conv_last(feat)
        image = tar + res
        return image


    def flops(self):
        pass

    def convert_checkpoint(self, state_dict):
        for k in list(state_dict.keys()):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
                # or k.find(".upsample.") >= 0
            ):
                state_dict.pop(k)
                print(k)
        return state_dict


if __name__ == "__main__":
    window_size = 8
    img_size = 256
    # Large, 20.13 
    model = DcamcmrsrNew3(
        img_size=256,
        embed_dim=180,
        num_head=6,
        depth=4,
    )

    print(model)
    # print(height, width, model.flops() / 1e9)

    lr = torch.randn((1, 3, img_size, img_size)).to('cuda')
    hr = torch.randn((1,3,img_size, img_size)).to('cuda')
    
    model = model.to('cuda')
    x = model(lr, hr)
    
    
    print(x.shape)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")
