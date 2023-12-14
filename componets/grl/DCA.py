import torch
import torch.nn as nn
from componets.grl.swin_v1_block import Mlp

class DCAT(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int = 64,
            num_heads: int = 8,
            dropout_rate: float = 0.1,
            pos_embed=True,
            mlp_ratio=4.0,
            drop=0.0,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        # self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = DCA(input_size=input_size, input_size1=input_size,hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)

        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        else:
            self.pos_embed = None

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x,ref):
        
        # B, C, H, W = ref.shape
        # ref = ref.reshape(B, C, H * W).permute(0, 2, 1)
        
        # B, C, H, W = x.shape
        # x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
            ref = ref + self.pos_embed
            
        attn = x + self.epa_block(self.norm1(x),self.norm1(ref))

        # attn_skip = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        attn_skip = self.norm2(self.mlp(attn)) + attn
        return attn_skip

class DCA(nn.Module):

    def __init__(self, input_size,input_size1, hidden_size, proj_size, num_heads=4, qkv_bias=True,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(hidden_size,hidden_size)
        
        self.kvv = nn.Linear(hidden_size,hidden_size*3)
        
        self.E = self.F = nn.Linear(input_size1, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size//2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size//2))

    def forward(self, x,ref):
        B, N, C = x.shape
        B1,N1,C1 = ref.shape
        
        x = self.q(x)
        q_shared = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        kvv = self.kvv(ref).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        kvv = kvv.permute(2, 0, 3, 1, 4)
        k_shared, v_CA, v_SA = kvv[0], kvv[1], kvv[2]

        #### 通道注意力
        q_shared = q_shared.transpose(-2, -1) #B,Head,C,N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature # (B,Head,C,N) * (#B,Head,N,C) -> (B,Head,C,C)

        attn_CA = attn_CA.softmax(dim=-1) #B,Head,C,C
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C) # (B,Head,C,C) * (B,Head,C,N) -> (B,Head,C,N) -> (B,N,C)
        
        
        #### 位置注意力
        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2 # (B,Head,N,C) * (B,Head,C,64) -> (B,Head,N,64)

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C) # (B,Head,N,64) * (B,Head,64,C) -> (B,Head,N,C) -> (B,N,C)

        # Concat fusion
        x_CA = self.out_proj(x_CA)
        x_SA = self.out_proj2(x_SA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x