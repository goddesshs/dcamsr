import torch

import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from einops.layers.torch import Rearrange
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

#Rectangle-Window Cross-Attention Block.
def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos



class Attention_regular(nn.Module):
    """ Regular Rectangle Cross-Window (regular-Rwin) self-attention with dynamic relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        resolution (int): Input resolution.
        idx (int): The identix of V-Rwin and H-Rwin, 0 is H-Rwin, 1 is Vs-Rwin.
        split_size (tuple(int)): Height and Width of the regular rectangle window (regular-Rwin).
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(self, dim, resolution, idx, split_size=[2, 4], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0.,
                 qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class RCA(nn.Module):
    """  Rectangle-Window Cross-Attention Block.
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of the regular rectangle window (regular-Rwin).
        shift_size (tuple(int)): Shift size for regular-Rwin.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, reso, num_heads,
                 split_size=[2, 4], shift_size=[1, 2], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            Attention_regular(
                dim // 2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1],
                                     self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1],
                                                                            1)  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0],
                                     self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1],
                                                                            1)  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, Ref=None, x_size=None):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """

        H, W = self.patches_resolution, self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        q = qkv[0].transpose(-2, -1).contiguous().view(B, C, H, W)  # Tar_Q
        
        if Ref is not None:
            img_Ref = self.norm1(Ref)
            
            qkv_Ref = self.qkv(img_Ref).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
            # v without partition
            # v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)
            # Ref_Q without partition
            qkv[1, :, :, :] = qkv_Ref[1]  # ref_K
            qkv[2, :, :, :] = qkv_Ref[2]  # ref_V
        

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            qkv = qkv.view(3, B, H, W, C)
            # H-Shift
            qkv_0 = torch.roll(qkv[:, :, :, :, :C // 2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, L, C // 2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, L, C // 2)

            if self.patches_resolution != H or self.patches_resolution != W:
                mask_tmp = self.calculate_mask(H, W)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=mask_tmp[0].to(x.device))
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=mask_tmp[1].to(x.device))

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1.view(B, L, C // 2).contiguous()
            x2 = x2.view(B, L, C // 2).contiguous()
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:, :, :, :C // 2], H, W).view(B, L, C // 2).contiguous()
            # H-Rwin
            x2 = self.attns[1](qkv[:, :, :, C // 2:], H, W).view(B, L, C // 2).contiguous()
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)

        # Locality Complementary Module
        lcm = self.get_v(q)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        attened_x = attened_x + lcm

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
    def dual_forward():
        pass


class DRCA(nn.Module):
    """  Dual Rectangle-Window Cross-Attention Block.
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of the regular rectangle window (regular-Rwin).
        shift_size (tuple(int)): Shift size for regular-Rwin.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
    """
    
    
    def __init__(self, dim, reso, num_heads,
                 split_size=[2, 4],  mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.kvv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.E = self.F = nn.Linear(reso*reso, dim, bias=qkv_bias)
        
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        
        self.attn_drop = nn.Dropout(drop)
        
        self.out_proj = nn.Linear(dim, int(dim//2))
        # self.out_proj2 = nn.Linear(dim, int(dim//2))
        
        self.attns = nn.ModuleList([
            Attention_regular(
                dim // 2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv



    def forward(self, x, Ref=None, x_size=None):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """

        H, W = self.patches_resolution, self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        x = self.norm1(x)
        Ref = self.norm1(Ref)
        query = self.q(x).reshape(B, -1, 2, C//2).permute(2,0,1,3)  # 2, B, HW, C//2
        
        kv = self.kvv(Ref).reshape(B, -1, 2, 2, C//2).permute(2,3,0,1,4)  # B, HW, 2, 2, C//2
        
        k = kv[0,...]
        v = kv[0,...]
        
        
        #### 通道注意力
        q_c = query[0,...] # B, HW, C//2
        q_c = q_c.reshape(B, -1, self.num_heads//2, C//self.num_heads).permute(0,2,3,1)
        k_c = k[0,...]
        k_c = k_c.reshape(B, -1, self.num_heads//2, C//self.num_heads).permute(0,2,3,1)
        v_c = v[0, ...]
        v_c = v_c.reshape(B, -1, self.num_heads//2, C//self.num_heads).permute(0,2,3,1)
        
        attn_CA = (q_c @ k_c.transpose(-2, -1))

        attn_CA = attn_CA.softmax(dim=-1) #B,Head,C,C
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_c).permute(0, 3, 1, 2).reshape(B, L, C//2) # (B,Head,C,C) * (B,Head,C,N) -> (B,Head,C,N) -> (B,N,C)
        


        #### 位置注意力
        q = query[1,...]
        k = k[1,...]
        v = v[1,...]
        qkv = torch.stack((q,k,v),dim=0)
        #V-Rwin
        x1 = self.attns[0](qkv, H, W).view(B, L, C // 2).contiguous()
        # H-Rwin
        x2 = self.attns[1](qkv, H, W).view(B, L, C // 2).contiguous()
        # Concat
        x_SA = torch.cat([x1, x2], dim=2)
        x_SA = self.out_proj(x_SA)

        return x_CA, x_SA

# class DRCA(nn.Module):
#     """  Dual Rectangle-Window Cross-Attention Block.
#     Args:
#         dim (int): Number of input channels.
#         reso (int): Input resolution.
#         num_heads (int): Number of attention heads.
#         split_size (tuple(int)): Height and Width of the regular rectangle window (regular-Rwin).
#         shift_size (tuple(int)): Shift size for regular-Rwin.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float): Dropout rate. Default: 0.0
#         attn_drop (float): Attention dropout rate. Default: 0.0
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
#     """
    
    
#     def __init__(self, dim, reso, num_heads,
#                  split_size=[2, 4],  mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.patches_resolution = reso
#         self.split_size = split_size
#         self.mlp_ratio = mlp_ratio
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
#         self.kvv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
#         self.E = self.F = nn.Linear(reso*reso, dim, bias=qkv_bias)
        
#         self.norm1 = norm_layer(dim)
#         self.branch_num = 2

#         self.proj1 = nn.Linear(dim, dim//2)
#         self.proj2 = nn.Linear(dim, dim//2)
        
#         self.proj_drop1 = nn.Dropout(drop)
#         self.proj_drop2 = nn.Dropout(drop)
        

#         self.attns = nn.ModuleList([
#             Attention_regular(
#                 dim // 2, resolution=self.patches_resolution, idx=i,
#                 split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
#                 qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
#             for i in range(self.branch_num)])

#         mlp_hidden_dim = int(dim * mlp_ratio)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = norm_layer(dim)

#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

#         if self.shift_size[0] > 0 or self.shift_size[1] > 0:
#             attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

#             self.register_buffer("attn_mask_0", attn_mask[0])
#             self.register_buffer("attn_mask_1", attn_mask[1])
#         else:
#             attn_mask = None

#             self.register_buffer("attn_mask_0", None)
#             self.register_buffer("attn_mask_1", None)

#     def calculate_mask(self, H, W):
#         # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#         # calculate attention mask for Rwin
#         img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
#         img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
#         h_slices_0 = (slice(0, -self.split_size[0]),
#                       slice(-self.split_size[0], -self.shift_size[0]),
#                       slice(-self.shift_size[0], None))
#         w_slices_0 = (slice(0, -self.split_size[1]),
#                       slice(-self.split_size[1], -self.shift_size[1]),
#                       slice(-self.shift_size[1], None))

#         h_slices_1 = (slice(0, -self.split_size[1]),
#                       slice(-self.split_size[1], -self.shift_size[1]),
#                       slice(-self.shift_size[1], None))
#         w_slices_1 = (slice(0, -self.split_size[0]),
#                       slice(-self.split_size[0], -self.shift_size[0]),
#                       slice(-self.shift_size[0], None))
#         cnt = 0
#         for h in h_slices_0:
#             for w in w_slices_0:
#                 img_mask_0[:, h, w, :] = cnt
#                 cnt += 1
#         cnt = 0
#         for h in h_slices_1:
#             for w in w_slices_1:
#                 img_mask_1[:, h, w, :] = cnt
#                 cnt += 1

#         # calculate mask for H-Shift
#         img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1],
#                                      self.split_size[1], 1)
#         img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1],
#                                                                             1)  # nW, sw[0], sw[1], 1
#         mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
#         attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
#         attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

#         # calculate mask for V-Shift
#         img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0],
#                                      self.split_size[0], 1)
#         img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1],
#                                                                             1)  # nW, sw[1], sw[0], 1
#         mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
#         attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
#         attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

#         return attn_mask_0, attn_mask_1

#     def forward(self, x, Ref=None, x_size=None):
#         """
#         Input: x: (B, H*W, C), x_size: (H, W)
#         Output: x: (B, H*W, C)
#         """

#         H, W = self.patches_resolution, self.patches_resolution
#         B, L, C = x.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#         img = self.norm1(x)
        
#         q_shared = self.q(x).reshape(B, -1, 2, C).permute(2,0,1,3)  # 2, B, HW, C
        
#         kvv = self.kvv(Ref).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        
#         k_shared, v_CA, v_SA = kvv[0], kvv[1], kvv[2]
        
        
#         # qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
#         #### 通道注意力
#         qkv_c = q_shared.repeat(3, 1)
#         qkv_c[1, :, :, :] = k_shared
#         qkv_c[2, :, :, :] = v_CA
#         # V-Rwin
#         x1 = self.attns[0](qkv_c[:, :, :, :C // 2], H, W).view(B, L, C // 2).contiguous()
#         # H-Rwin
#         x2 = self.attns[1](qkv_c[:, :, :, C // 2:], H, W).view(B, L, C // 2).contiguous()
#         # Concat
#         attened_xca = torch.cat([x1, x2], dim=2)


#        #### 位置注意力
#         k_shared_projected = self.E(k_shared)
#         v_SA_projected = self.F(v_SA)
        
#         qkv_s = q_shared.repeat(3, 1)
#         qkv_s[1, :, :, :] = k_shared_projected
#         qkv_c[2, :, :, :] = v_SA_projected

#         x1 = self.attns[0](qkv_c[:, :, :, :C // 2], H, W).view(B, L, C // 2).contiguous()
#         # H-Rwin
#         x2 = self.attns[1](qkv_c[:, :, :, C // 2:], H, W).view(B, L, C // 2).contiguous()
#         # Concat
#         attened_xsa = torch.cat([x1, x2], dim=2)
        
        
#         x_CA = self.proj1(attened_xca)
#         x_SA = self.proj2(attened_xsa)
        
        
#         x = torch.cat((x_SA, x_CA), dim=-1)

        
        
       

#         attened_x = attened_x + lcm

#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x
    
    # def dual_forward():
    

class RCB(nn.Module):
    def __init__(self, dim, reso, num_heads,
                 split_size=[2, 4], shift_size=[1, 2], mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., branch_num=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, down=True):
        super().__init__()
        self.down = down
        self.attn1 = RCA(
            dim=dim,
            num_heads=num_heads,
            reso=reso,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            split_size=split_size,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            shift_size=shift_size)
                
        self.attn2 = RCA(
            dim=dim,
            num_heads=num_heads,
            reso=reso,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            split_size=split_size,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            shift_size=shift_size)    
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm = norm_layer(dim)
        # self.norm = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    #self-attention, cross-attention
    def forward(self, x, ref=None):
        #self attention
        # if ref is None:
        #     ref = x
        if self.down:
            x = self.attn1(x) + x
            ref = self.attn1(ref) + ref
            
            x = self.attn2(x, ref) + x
            ref = self.attn2(ref) + ref
            
            x = self.mlp(self.norm(x)) + x
            ref = self.mlp(self.norm(x)) + ref
            return x, ref
        
        else:
            x = self.attn1(x) + x
            x = self.attn2(x) + x
            x = self.mlp(self.norm(x)) + x
        
            return x, None

class ResidualGroup(nn.Module):
    """ ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size_0 (int): The one side of rectangle window.
        split_size_1 (int): The other side of rectangle window. For axial-Rwin, split_size_w is Zero.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of Cross Aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 dim,
                 out_dim,
                 reso,
                 num_heads,
                 split_size_0=7,
                 split_size_1=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_paths=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 depth=2,
                 use_chk=False,
                 resi_connection='1conv',
                 downsample=None,
                 upsample=None):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            RCB(
                dim=dim,
                num_heads=num_heads,
                reso=reso,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=[split_size_0, split_size_1],
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                shift_size=[0, 0] if (i % 2 == 0) else [split_size_0 // 2, split_size_1 // 2],
                down=(upsample is None))
            for i in range(depth)])

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.downsample = None
        self.upsample = None
        if downsample is not None:
            self.downsample = downsample(reso, dim=dim, out_dim = out_dim, norm_layer=norm_layer)
        if upsample is not None:
            self.upsample = upsample(reso, dim=dim, out_dim = out_dim)
            
    def forward(self, x, Ref=None, x_size=None):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = self.reso, self.reso
        res = x
        ref_ori = Ref
        for blk in self.blocks:
            if self.use_chk:
                x, Ref = checkpoint.checkpoint(blk, x, Ref)
            else:
                x, Ref = blk(x, Ref)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x
        if Ref is not None:
            Ref = rearrange(Ref, "b (h w) c -> b c h w", h=H, w=W)
            Ref = self.conv(Ref)
            Ref = rearrange(Ref, "b c h w -> b (h w) c")
            Ref = ref_ori + Ref

        out_size = self.reso
        if self.downsample is not None and Ref is not None:
            x, out_size = self.downsample(x)
            Ref, _ = self.downsample(Ref)
            return x, Ref, out_size
        # elif Ref is None:
            
        elif self.upsample is not None:
            x  = self.upsample(x)
        return x
        # return x