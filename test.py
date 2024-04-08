import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import functools
from functools import partial
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from semseg.models.modules.ffm import FeatureRectifyModule as FRM
from semseg.models.modules.ffm import ChannelEmbed
from semseg.models.modules.mspa import MSPABlock
from semseg.utils.utils import nchw_to_nlc, nlc_to_nchw
from semseg.models.backbones.resnet_encoder import ResNet
from semseg.models.backbones.SwinT import SwinTransformer
from semseg.models.backbones.DCR_fuse import DCR
#from semseg.models.backbones.DCR import DCR

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

########}??¦Ì¡ê####
class DFEM(nn.Module):
    def __init__(self, infeature):
        super(DFEM, self).__init__()
        self.depth_spatial_attention = SpatialAttention()
        self.depth_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()

    def forward(self, r, d):
        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        d_f = d * sa
        d_f = d + d_f
        d_ca = self.depth_channel_attention(d_f)
        d_out = d * d_ca
        return d_out

########
class LFEM(nn.Module):
    def __init__(self, infeature):
        super(LFEM, self).__init__()
        self.lidar_spatial_attention = SpatialAttention()
        self.lidar_channel_attention = ChannelAttention(infeature)
        self.del_spatial_attention = SpatialAttention()

    def forward(self, d, e, l):
        mul_fuse = d * e * l 
        sa = self.del_spatial_attention(mul_fuse)
        l_f = l * sa
        l_f = l + l_f
        l_ca = self.lidar_channel_attention(l_f)
        l_out = l * l_ca
        return l_out

class EFEM(nn.Module):
    def __init__(self, infeature):
        super(EFEM, self).__init__()
        self.event_spatial_attention = SpatialAttention()
        self.event_channel_attention = ChannelAttention(infeature)
        self.del_spatial_attention = SpatialAttention()

    def forward(self,  d, e, l):
        mul_fuse = d * e * l
        sa = self.del_spatial_attention(mul_fuse)
        e_f = e * sa
        e_f = e + e_f
        e_ca = self.event_channel_attention(e_f)
        e_out = e * e_ca
        return e_out


class DFEM(nn.Module):
    def __init__(self, infeature):
        super(DFEM, self).__init__()
        self.depth_spatial_attention = SpatialAttention()
        self.depth_channel_attention = ChannelAttention(infeature)
        self.del_spatial_attention = SpatialAttention()

    def forward(self,  d, e, l):
        mul_fuse = d * e * l
        sa = self.del_spatial_attention(mul_fuse)
        d_f = d * sa
        d_f = d + d_f
        d_ca = self.depth_channel_attention(d_f)
        d_out = d * d_ca
        return d_out

class RFEM(nn.Module):
    def __init__(self, infeature, w=12, h=12, heads=4):
        super(RFEM, self).__init__()
        self.rgb_channel_attention = ChannelAttention(infeature)
        self.rgb_spatial_attention = SpatialAttention()
        self.rd_spatial_attention = SpatialAttention()

    def forward(self, r, d):

        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        r_f = r + r_f
        r_ca = self.rgb_channel_attention(r_f)
        r_out = r * r_ca
        return r_out


class CMFM(nn.Module):
    def __init__(self, infeature, w=12, h=12, heads=4):
        super(CMFM, self).__init__()

        self.dfem = DFEM(infeature)
        self.efem = EFEM(infeature)
        self.lfem  = LFEM(infeature)
        #self.rfem = RFEM(infeature, w, h, heads)
        self.ca = ChannelAttention(infeature * 2)
        self.conv  = nn.Conv2d(infeature * 2, infeature, kernel_size=3, stride=1, padding=1)
    def forward(self, d, e, l):
        #fr = self.rfem(d, e, l)
        fd = self.dfem(d, e, l)
        fe = self.efem(d, e, l)
        fl = self.lfem(d, e, l)
        mul_fea = fd * fe * fl
        add_fea = fd + fe + fl
        fuse_fea = self.conv(torch.cat([mul_fea, add_fea], dim=1))
        # att = torch.sigmoid(fuse_fea)
        # fuse_fea = fr*att+(1-att)*fd
        # fuse_fea_ca = self.ca(fuse_fea)
        # fuse_fea = fuse_fea_ca * fuse_fea
        # att = torch.sigmoid(fuse_fea)
        # fuse_fea = fuse_fea * att
        
        return fuse_fea



class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, in_dim3, k_dim, v_dim, num_heads=8):

        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim3, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, x3, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        seq_len3 = x3.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x3).view(batch_size, seq_len3, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output




class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))    # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))
        
        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x 


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


cmnext_settings = {
    # 'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    # 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    # 'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class CMNeXt(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        extra_depths = depths 
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)

        self.encode_r  = ResNet() 
        self.encode_d = ResNet() 
        self.encode_e = ResNet() 
        self.encode_l  = ResNet() 


        ############stage1,2 fuse##########

        self.cmfm1 = CMFM(64, 256, 256, 4)
        self.cmfm2 = CMFM(128, 128, 128, 4)

        self.cmfm3 = CMFM(320, 64, 64, 4)
        self.cmfm4 = CMFM(512, 32, 32, 4)




        ############capsule##########
        #self.DCR0 = DCR()
        #self.DCR1 = DCR()
        #self.DCR2 = DCR()
        self.DCR3 = DCR()
        self.DCR4 = DCR()

        self.conv2048_256r =  conv3x3_bn_relu(2048, 256)
        self.conv2048_256d =  conv3x3_bn_relu(2048, 256)
        self.conv2048_256e =  conv3x3_bn_relu(2048, 256)
        self.conv2048_256l =  conv3x3_bn_relu(2048, 256)

        self.conv1024_256r =  conv3x3_bn_relu(1024, 256)
        self.conv1024_256d =  conv3x3_bn_relu(1024, 256)
        self.conv1024_256e =  conv3x3_bn_relu(1024, 256)
        self.conv1024_256l =  conv3x3_bn_relu(1024, 256)

        self.conv512_256r =  conv3x3_bn_relu(512, 256)
        self.conv512_256d =  conv3x3_bn_relu(512, 256)
        self.conv512_256e =  conv3x3_bn_relu(512, 256)
        self.conv512_256l =  conv3x3_bn_relu(512, 256)

        self.conv320_64d1 =  conv3x3_bn_relu(320, 64) 
        self.conv320_64e1 =  conv3x3_bn_relu(320, 64) 
        self.conv320_64l1 =  conv3x3_bn_relu(320, 64) 

        self.conv640_128d2 =  conv3x3_bn_relu(640, 128)
        self.conv640_128e2 =  conv3x3_bn_relu(640, 128)
        self.conv640_128l2 =  conv3x3_bn_relu(640, 128)

        self.conv832_320f3 = conv3x3_bn_relu(832, 320)
        self.conv256_320_shd = conv3x3_bn_relu(256, 320)
        self.conv256_320_spc = conv3x3_bn_relu(256, 320)

        self.conv1024_512f4 =  conv3x3_bn_relu(1024, 512)
        self.conv256_512_shd = conv3x3_bn_relu(256, 512)
        self.conv256_512_spc = conv3x3_bn_relu(256, 512)




        ############cross attention##########

        #self.cat1 = CrossAttention(64, 64, 46, 256, 256)
        #self.cat2 = CrossAttention(128, 128, 128, 128, 128)
        self.cat3 = CrossAttention(320,320,320,64,64)
        self.cat4 = CrossAttention(512,512,512,32,32)






        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
   
        if self.num_modals > 0:
            self.extra_downsample_layers = nn.ModuleList([
                PatchEmbedParallel(3, embed_dims[0], 7, 4, 7//2, self.num_modals),
                *[PatchEmbedParallel(embed_dims[i], embed_dims[i+1], 3, 2, 3//2, self.num_modals) for i in range(3)]
            ])
        if self.num_modals > 1:
            self.extra_score_predictor = nn.ModuleList([PredictorConv(embed_dims[i], self.num_modals) for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        if self.num_modals > 0:
            self.extra_block1 = nn.ModuleList([MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[0])]) # --- MSPABlock
            self.extra_norm1 = ConvLayerNorm(embed_dims[0])
            
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        if self.num_modals > 0:
            self.extra_block2 = nn.ModuleList([MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[1])])
            self.extra_norm2 = ConvLayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        if self.num_modals > 0:
            self.extra_block3 = nn.ModuleList([MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[2])])
            self.extra_norm3 = ConvLayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        if self.num_modals > 0:
            self.extra_block4 = nn.ModuleList([MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[3])])
            self.extra_norm4 = ConvLayerNorm(embed_dims[3])

        if self.num_modals > 0:
            num_heads = [1,2,5,8]
            self.FRMs = nn.ModuleList([
                FRM(dim=embed_dims[0], reduction=1),
                FRM(dim=embed_dims[1], reduction=1),
                FRM(dim=embed_dims[2], reduction=1),
                FRM(dim=embed_dims[3], reduction=1)])
            self.FFMs = nn.ModuleList([
                FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=nn.BatchNorm2d)])

    def tokenselect(self, x_ext, module):    
        x_scores = module(x_ext) 
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        x_f = functools.reduce(torch.max, x_ext)
        return x_f
     
    def forward(self, x: list) -> list:

        outs = []

        #rgb_list = self.encode_r(x[0])
        depth_list = self.encode_d(x[1])
        event_list =self.encode_e(x[2])
        lidar_list =self.encode_l(x[3])
        
        #r0 = rgb_list[0]  # 2048, 32
        #r1 = rgb_list[1]  # 1024, 64
        #r2 = rgb_list[2]  # 512, 128
        #r3 = rgb_list[3]  # 256, 256
        #r4 = rgb_list[4]  # 64, 256

        d0 = depth_list[0]
        d1 = depth_list[1]
        d2 = depth_list[2]
        d3 = depth_list[3]
        d4 = depth_list[4]

        e0 = event_list[0]
        e1 = event_list[1]
        e2 = event_list[2]
        e3 = event_list[3]	
        e4 = event_list[4]

        l0 = lidar_list[0]
        l1 = lidar_list[1]
        l2 = lidar_list[2]
        l3 = lidar_list[3]
        l4 = lidar_list[4]


        x_cam = x[0]        
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]


        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)  #[1, 65536, 64]

        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)    #[1, 64, 256, 256]

        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[0](x_ext)

            x_ext[0] = self.conv320_64d1(torch.cat((x_ext[0],d3), dim =1))
            x_ext[1] = self.conv320_64e1(torch.cat((x_ext[1],e3), dim =1))
            x_ext[2] = self.conv320_64l1(torch.cat((x_ext[2],l3), dim =1))
            x_ext1 = [x_ext[0], x_ext[1], x_ext[2]]

            x_f1 = self.cmfm1(x_ext[0], x_ext[1], x_ext[2])
            x_f2 = self.tokenselect(x_ext1, self.extra_score_predictor[0]) if self.num_modals > 1 else x_ext[0] 
            x_f = x_f1 + x_f2
            for blk in self.extra_block1:
                x_f = blk(x_f)
            x1_f = self.extra_norm1(x_f)
            x1_cam, x1_f = self.FRMs[0](x1_cam, x1_f)
            x_fused = self.FFMs[0](x1_cam, x1_f)
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x1_f for x_ in x_ext] if self.num_modals > 1 else [x1_f]
        else:
            outs.append(x1_cam)
        #print(111111111111111111111111111111)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)

        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[1](x_ext)

            x_ext[0] = self.conv640_128d2(torch.cat((x_ext[0],d2), dim =1))
            x_ext[1] = self.conv640_128e2(torch.cat((x_ext[0],e2), dim =1))
            x_ext[2] = self.conv640_128l2(torch.cat((x_ext[0],l2), dim =1))
            x_ext2 = [x_ext[0], x_ext[1], x_ext[2]]
            #print(222222)

            #b2, c2, h2, w2 = x_ext[0].shape
            #x_ext_flatten0 =  x_ext[0].view(b2, c2, -1).permute(0, 2, 1)
            #x_ext_flatten1 =  x_ext[1].view(b2, c2, -1).permute(0, 2, 1)
            #x_ext_flatten2 =  x_ext[2].view(b2, c2, -1).permute(0, 2, 1)

            #x_ext_f =self.cat2(x_ext_flatten0,x_ext_flatten1,x_ext_flatten2).view(b2, h2, w2,-1).permute(0, 3,1,2)

            x_f1 = self.cmfm2(x_ext[0], x_ext[1], x_ext[2])
            #x_f = x_ext_f + x_f

            x_f2 = self.tokenselect(x_ext2, self.extra_score_predictor[1]) if self.num_modals > 1 else x_ext[0] 
            x_f = x_f1 + x_f2

      
            #print(x_f.shape)
            for blk in self.extra_block2:
                x_f = blk(x_f)
            #print(x_f.shape)
            #print(33333333)
            x2_f = self.extra_norm2(x_f)

            x2_cam, x2_f = self.FRMs[1](x2_cam, x2_f)
            x_fused = self.FFMs[1](x2_cam, x2_f)
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x2_f for x_ in x_ext] if self.num_modals > 1 else [x2_f]
        else:
            outs.append(x2_cam)
        #print(111111111111111111111111111111)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)

        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[2](x_ext)

            x3_dcr = self.DCR3( self.conv1024_256d(d1), self.conv1024_256e(e1), self.conv1024_256l(l1))  #[1, 256, 64 ,64]
            

            b3, c3, h3, w3 = x_ext[0].shape
            x_ext_flatten0 =  x_ext[0].view(b3, c3, -1).permute(0, 2, 1)
            x_ext_flatten1 =  x_ext[1].view(b3, c3, -1).permute(0, 2, 1)
            x_ext_flatten2 =  x_ext[2].view(b3, c3, -1).permute(0, 2, 1)
            x_ext_f =self.cat3(x_ext_flatten0,x_ext_flatten1,x_ext_flatten2).view(b3,h3,w3,-1).permute(0, 3,1,2)
            #print(x_ext_f.shape)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[2]) if self.num_modals > 1 else x_ext[0]
            print(x_f.shape)

            #x_f =  x_f + x_ext_f

            x3_dcr_shd = self.conv256_320_shd(x3_dcr[0])
            x3_dcr_spc = self.conv256_320_spc(x3_dcr[1])
            #print(x3_dcr_shd.shape)
            x3_capf = self.cmfm3(x3_dcr_spc, x3_dcr_shd)
            x_f =  x_f + x_ext_f + x3_capf



            #x_f = self.tokenselect(x_ext, self.extra_score_predictor[2]) if self.num_modals > 1 else x_ext[0] 
            for blk in self.extra_block3:
                x_f = blk(x_f)

            #x3_cam = self.conv832_320f3(torch.cat((x3_dcr[0],x3_dcr[1],x3_cam), dim =1))

            #x_f = self.conv256_320_shd(x3_dcr[0])
            x3_f = self.extra_norm3(x_f)
            x3_cam, x3_f = self.FRMs[2](x3_cam, x3_f)
            x_fused = self.FFMs[2](x3_cam, x3_f)
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x3_f for x_ in x_ext] if self.num_modals > 1 else [x3_f]
        else:
            outs.append(x3_cam)


        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[3](x_ext)

            x4_dcr = self.DCR4( self.conv2048_256d(d0), self.conv2048_256e(e0), self.conv2048_256l(l0))  #[1, 256, 128 ,128]

            b4, c4, h4, w4 = x_ext[0].shape
            x_ext_flatten0 =  x_ext[0].view(b4, c4, -1).permute(0, 2, 1)
            x_ext_flatten1 =  x_ext[1].view(b4, c4, -1).permute(0, 2, 1)
            x_ext_flatten2 =  x_ext[2].view(b4, c4, -1).permute(0, 2, 1)

            x_ext_f =self.cat4(x_ext_flatten0,x_ext_flatten1,x_ext_flatten2).view(b4,h4,w4,-1).permute(0, 3,1,2)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[3]) if self.num_modals > 1 else x_ext[0] 


            #x_f = 0.5* x_f + 0.5*x_ext_f
            x4_dcr_shd = self.conv256_512_shd(x4_dcr[0])
            x4_dcr_spc = self.conv256_512_spc(x4_dcr[1])
            x4_capf = self.cmfm4(x4_dcr_spc, x4_dcr_shd)
            x_f =  x_f + x_ext_f + x4_capf


            print(6666)
            #x_f = self.tokenselect(x_ext, self.extra_score_predictor[3]) if self.num_modals > 1 else x_ext[0] 
            for blk in self.extra_block4:
                x_f = blk(x_f)

            #x4_cam = self.conv1024_512f4(torch.cat((x4_dcr[0],x4_dcr[1],x4_cam), dim =1))
            #x_f = self.conv256_512_shd(x4_dcr[0])
            x4_f = self.extra_norm4(x_f)
            x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)
            x_fused = self.FFMs[3](x4_cam, x4_f)
            outs.append(x_fused)
        else:
            outs.append(x4_cam)
        #exit(0)
        return outs


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = CMNeXt('B2', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)

