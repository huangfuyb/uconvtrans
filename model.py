from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple


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


class DWConv(nn.Module):

    def __init__(self, inplines):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(inplines, inplines, 3, 1, 1, bias=True, groups=inplines)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp_DW(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpLight(nn.Module):

    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0.0, ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports  non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block. Without shift size!
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dw=False):
        super().__init__()
        self.dw = dw
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads  # 3, 6, 12, 24
        self.window_size = window_size
        self.shift_size = shift_size  # 0 or window_size//2
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if dw:
            self.mlp = Mlp_DW(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if self.dw:
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class ResConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ResConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        return x


class Conv_Block_3with3(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv_Block_3with3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=1, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride,
                                                  padding=(kernel_size - stride) // 2, bias=False),
                                        nn.BatchNorm2d(outplanes), nn.ReLU())

    def forward(self, x):
        return self.conv_block(x)


class Up_ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(Up_ConvBlock, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(outplanes), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class FuseTrans2Conv(nn.Module):

    def __init__(self, inplanes, outplanes, act_layer=nn.ReLU, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 method='add'):
        super().__init__()
        self.method = method
        if method == 'add':
            self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_project = nn.Conv2d(inplanes + outplanes, outplanes, kernel_size=1, stride=1, padding=0)

        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, t, H, W, c):
        '''

        :param t:the output of transformer
        :param H:
        :param W:
        :param xc:the output of convolution
        :return: fuse output
        '''
        B, _, C = t.shape
        t_r = t.transpose(1, 2).reshape(B, C, H, W)
        if self.method == 'add':
            x = self.act(self.bn(self.conv_project(t_r)))
            return x + c
        else:
            temp = torch.cat((t_r, c), dim=1)
            return self.act(self.bn(self.conv_project(temp)))


class FuseConv2Trans(nn.Module):
    def __init__(self, inplanes, outplanes, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 method='add'):
        super().__init__()
        self.method = method
        if method == 'add':
            self.project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        else:
            self.project = nn.Linear(inplanes + outplanes, outplanes)
        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, c, t):
        if self.method == 'add':
            x = self.project(c)
            x = x.flatten(2).transpose(1, 2)
            x = self.ln(x)
            x = self.act(x)
            return x + t
        else:
            c = c.flatten(2).transpose(1, 2)
            x = torch.cat((c, t), dim=-1)
            x = self.project(x)
            x = self.ln(x)
            x = self.act(x)
            return x


class ReversePatch(nn.Module):

    def __init__(self, input_resolution, embed_dim, outplanes, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.linear = nn.Linear(embed_dim, outplanes, bias=False)
        self.norm = norm_layer(outplanes)
        self.act = act_layer()
        self.outplanes = outplanes

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, _, C = x.shape
        H, W = self.input_resolution
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2).reshape(B, self.outplanes, H, W)

        return x


class LastFuseBlock(nn.Module):

    def __init__(self, inplanes, outplanes, embed_dim, input_resolution, method='add'):
        super().__init__()
        self.method = method
        self.convblock = ConvBlock(inplanes=inplanes, outplanes=outplanes, kernel_size=1)
        self.transblock = ReversePatch(input_resolution, embed_dim, outplanes)
        if self.method == 'add':
            self.fuse = None
        else:
            self.fuse = ConvBlock(inplanes=outplanes + outplanes, outplanes=outplanes, kernel_size=1)

    def forward(self, x, t):
        if self.method == 'add':
            c = self.convblock(x)
            t = self.transblock(t)
            return c + t
        else:
            c = self.convblock(x)
            t = self.transblock(t)
            return self.fuse(torch.cat((c, t), dim=1))


class TransResConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride, res_conv, embed_dim, input_resolution, window_size,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 groups=1, conv_depth=2, trans_depth=2, conv_type='res', dw=False, method='add'):
        super().__init__()
        self.method = method
        # first convblock
        self.convblock = nn.ModuleList()
        for i in range(conv_depth // 2):
            if conv_type == 'res':
                conv_layers1 = ResConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                            groups=groups)
            else:
                conv_layers1 = Conv_Block_3with3(inplanes=inplanes, outplanes=outplanes)
            self.convblock.append(conv_layers1)
        # transformer to conv
        self.conv_project = FuseTrans2Conv(embed_dim, outplanes, method=method)
        # second convblock
        self.fuseconv = nn.ModuleList()
        for i in range(conv_depth // 2):
            if conv_type == 'res':
                conv_layers2 = ResConvBlock(inplanes=outplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                            groups=groups)
            else:
                conv_layers2 = Conv_Block_3with3(inplanes=inplanes, outplanes=outplanes)
            self.fuseconv.append(conv_layers2)
        # first transformer block
        self.transblock = nn.ModuleList()
        for i in range(trans_depth // 2):
            trans_layers1 = SwinTransformerBlock(embed_dim, input_resolution, num_heads, window_size=window_size,
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                                attn_drop=attn_drop, drop_path=drop_path, dw=dw)
            self.transblock.append(trans_layers1)
        # conv to transformer
        self.trans_project = FuseConv2Trans(outplanes, embed_dim, method=method)
        # second transformer block
        self.fusetrans = nn.ModuleList()
        for i in range(trans_depth // 2):
            trans_layers2 = SwinTransformerBlock(embed_dim, input_resolution, num_heads, window_size=window_size,
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                                attn_drop=attn_drop, drop_path=drop_path, dw=dw)
            self.fusetrans.append(trans_layers2)

    def forward(self, x, t):
        _, _, H, W = x.shape
        for layer in self.convblock:
            x = layer(x)
        for layer in self.transblock:
            t = layer(t)
        # conv fuse trans
        x = self.conv_project(t, H, W, x)
        for layer in self.fuseconv:
            x = layer(x)
        # trans fuse conv
        t = self.trans_project(x, t)
        for layer in self.fusetrans:
            t = layer(t)
        return x, t


class UConFormer(nn.Module):

    def __init__(self, num_class=4, inplanes=1, basic_channel=32, img_size=256, embed_dim=32, window_size=8,
                 patch_size=1, num_heads=[2, 4, 8, 16], mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, conv_depths=[2, 2, 2, 2], trans_depths=[2, 2, 2, 2],
                 conv_type='res', dw=False, method='cat'):
        super(UConFormer, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Stem stage:
        self.stem = nn.Sequential(nn.Conv2d(inplanes, basic_channel, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(basic_channel), nn.ReLU(inplace=True))  # [B, C, 128, 128]

        # stage 1:
        self.conv1 = ConvBlock(basic_channel, basic_channel, kernel_size=3, stride=1)
        self.patch_embed = PatchEmbed(img_size=img_size // 2, patch_size=patch_size, in_chans=basic_channel,
                                      embed_dim=embed_dim, norm_layer=norm_layer)  # [B, 128*128, D]
        self.TransConv_1 = TransResConvBlock(inplanes=basic_channel, outplanes=basic_channel, stride=1,
                                             res_conv=False,
                                             embed_dim=embed_dim, input_resolution=(img_size // 2, img_size // 2),
                                             window_size=window_size, num_heads=num_heads[0], mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                             conv_depth=conv_depths[0], trans_depth=trans_depths[0],
                                             conv_type=conv_type, dw=dw, method=method)

        # stage 2:
        self.conv2 = ConvBlock(inplanes=basic_channel, outplanes=basic_channel * 2, kernel_size=3)
        self.TransDownsample1 = PatchMerging(to_2tuple(img_size // 2), embed_dim)
        self.TransConv_2 = TransResConvBlock(inplanes=basic_channel * 2, outplanes=basic_channel * 2, stride=1,
                                             res_conv=False,
                                             embed_dim=embed_dim * 2, input_resolution=(img_size // 4, img_size // 4),
                                             window_size=window_size, num_heads=num_heads[1], mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                             conv_depth=conv_depths[1], trans_depth=trans_depths[1],
                                             conv_type=conv_type, dw=dw, method=method)

        # stage 3:
        self.conv3 = ConvBlock(inplanes=basic_channel * 2, outplanes=basic_channel * 4, kernel_size=3)
        self.TransDownsample2 = PatchMerging(to_2tuple(img_size // 4), embed_dim * 2)
        self.TransConv_3 = TransResConvBlock(inplanes=basic_channel * 4, outplanes=basic_channel * 4, stride=1,
                                             res_conv=False,
                                             embed_dim=embed_dim * 4, input_resolution=(img_size // 8, img_size // 8),
                                             window_size=window_size, num_heads=num_heads[2], mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                             conv_depth=conv_depths[2], trans_depth=trans_depths[2],
                                             conv_type=conv_type, dw=dw, method=method)
        # stage 4:
        self.conv4 = ConvBlock(inplanes=basic_channel * 4, outplanes=basic_channel * 8, kernel_size=3)
        self.TransDownsample3 = PatchMerging(to_2tuple(img_size // 8), embed_dim * 4)
        self.TransConv_4 = TransResConvBlock(inplanes=basic_channel * 8, outplanes=basic_channel * 8, stride=1,
                                             res_conv=False,
                                             embed_dim=embed_dim * 8, input_resolution=(img_size // 16, img_size // 16),
                                             window_size=window_size, num_heads=num_heads[3], mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                             conv_depth=conv_depths[3], trans_depth=trans_depths[3],
                                             conv_type=conv_type, dw=dw, method=method)
        # decoder:
        # stage 3:
        self.DeUpConv3 = Up_ConvBlock(inplanes=basic_channel * 8, outplanes=basic_channel * 4)
        self.DeCatConv3 = ConvBlock(inplanes=basic_channel * 8, outplanes=basic_channel * 4, kernel_size=3)
        self.DeCatTrans3 = MlpLight(in_features=embed_dim * 8, out_features=embed_dim * 4, drop=drop)
        self.TransUpsample3 = PatchExpand(to_2tuple(img_size // 16), dim=embed_dim * 8, dim_scale=2,
                                          norm_layer=norm_layer)
        self.DeTransConv_3 = TransResConvBlock(inplanes=basic_channel * 4, outplanes=basic_channel * 4, stride=1,
                                               res_conv=False,
                                               embed_dim=embed_dim * 4, input_resolution=(img_size // 8, img_size // 8),
                                               window_size=window_size, num_heads=num_heads[2], mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                               conv_depth=conv_depths[2], trans_depth=trans_depths[2],
                                               conv_type=conv_type, dw=dw, method=method)
        # stage 2:
        self.DeUpConv2 = Up_ConvBlock(inplanes=basic_channel * 4, outplanes=basic_channel * 2)
        self.DeCatConv2 = ConvBlock(inplanes=basic_channel * 4, outplanes=basic_channel * 2, kernel_size=3)
        self.DeCatTrans2 = MlpLight(in_features=embed_dim * 4, out_features=embed_dim * 2, drop=drop)
        self.TransUpsample2 = PatchExpand(to_2tuple(img_size // 8), dim=embed_dim * 4, dim_scale=2,
                                          norm_layer=norm_layer)
        self.DeTransConv_2 = TransResConvBlock(inplanes=basic_channel * 2, outplanes=basic_channel * 2, stride=1,
                                               res_conv=False,
                                               embed_dim=embed_dim * 2, input_resolution=(img_size // 4, img_size // 4),
                                               window_size=window_size, num_heads=num_heads[1], mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                               conv_depth=conv_depths[1], trans_depth=trans_depths[1],
                                               conv_type=conv_type, dw=dw, method=method)

        # stage 1:
        self.DeUpConv1 = Up_ConvBlock(inplanes=basic_channel * 2, outplanes=basic_channel)
        self.DeCatConv1 = ConvBlock(inplanes=basic_channel * 2, outplanes=basic_channel, kernel_size=3)
        self.DeCatTrans1 = MlpLight(in_features=embed_dim * 2, out_features=embed_dim, drop=drop)
        self.TransUpsample1 = PatchExpand(to_2tuple(img_size // 4), dim=embed_dim * 2, dim_scale=2,
                                          norm_layer=norm_layer)
        self.DeTransConv_1 = TransResConvBlock(inplanes=basic_channel, outplanes=basic_channel, stride=1,
                                               res_conv=False,
                                               embed_dim=embed_dim, input_resolution=(img_size // 2, img_size // 2),
                                               window_size=window_size, num_heads=num_heads[0], mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                               conv_depth=conv_depths[0], trans_depth=trans_depths[0],
                                               conv_type=conv_type, dw=dw, method=method)
        # last fuse:
        self.LastFuse = LastFuseBlock(inplanes=basic_channel, outplanes=basic_channel, embed_dim=embed_dim,
                                      input_resolution=to_2tuple(img_size // 2), method=method)

        self.classify = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                      nn.Conv2d(basic_channel, num_class, kernel_size=1))

    def forward(self, x):
        x1 = self.stem(x)  # [B, C, 128, 128]
        # encoder:
        # stage 1:
        x1_conv_input = self.conv1(x1)  # [B, C, 128, 128] todo
        x1_trans_input = self.patch_embed(x1)  # [B, 128*128, D] todo
        x1_conv1, x1_trans1 = self.TransConv_1(x1_conv_input, x1_trans_input)  # [B, C, 128, 128], [B, 128*128, D]
        # stage 2:
        x2 = self.MaxPool(x1_conv1)  # [B, C, 64, 64]
        x2_conv_input = self.conv2(x2)  # [B, C*2, 64, 64] todo
        x2_trans_input = self.TransDownsample1(x1_trans1)  # [B, 64*64, D*2] todo
        x2_conv1, x2_trans1 = self.TransConv_2(x2_conv_input, x2_trans_input)  # [B, C*2, 64, 64], [B, 64*64, D*2]

        # stage 3:
        x3 = self.MaxPool(x2_conv1)  # [B, C*2, 32, 32]
        x3_conv_input = self.conv3(x3)  # [B, C*4, 32, 32] todo
        x3_trans_input = self.TransDownsample2(x2_trans1)  # [B, 32*32, D*4] todo
        x3_conv1, x3_trans1 = self.TransConv_3(x3_conv_input, x3_trans_input)  # [B, C*4, 32, 32], [B, 32*32, D*4]

        # stage 4:
        x4 = self.MaxPool(x3_conv1)
        x4_conv_input = self.conv4(x4)  # [B, C*8, 16, 16] todo
        x4_trans_input = self.TransDownsample3(x3_trans1)  # [B, 16*16, D*8] todo
        x4_conv1, x4_trans1 = self.TransConv_4(x4_conv_input, x4_trans_input)  # [B, C*8, 16, 16], [B, 16*16, D*8]

        # decoder:
        # stage 3:
        d3 = self.DeUpConv3(x4_conv1)  # [B, C*4, 32, 32]
        d3_conv_input = self.DeCatConv3(torch.cat((d3, x3_conv1), dim=1))  # [B, C*4, 32, 32]
        d3_trans_input = self.TransUpsample3(x4_trans1)  # [B, 32*32, D*4]
        d3_trans_input = self.DeCatTrans3(torch.cat((d3_trans_input, x3_trans1), dim=-1))  # [B, 64*64, D*4]
        d3_conv1, d3_trans1 = self.DeTransConv_3(d3_conv_input, d3_trans_input)  # [B, C*4, 32, 32], [B, 32*32, D*4]

        # stage 2:
        d2 = self.DeUpConv2(d3_conv1)  # [B, C*2, 64, 64]
        d2_conv_input = self.DeCatConv2(torch.cat((d2, x2_conv1), dim=1))  # [B, C*2, 64, 64]
        d2_trans_input = self.TransUpsample2(d3_trans1)  # [B, 64*64, D*2]
        d2_trans_input = self.DeCatTrans2(torch.cat((d2_trans_input, x2_trans1), dim=-1))  # [B, 64*64, D*2]
        d2_conv1, d2_trans1 = self.DeTransConv_2(d2_conv_input, d2_trans_input)  # [B, C*2, 64, 64], [B, 64*64, D*2]

        # stage 1:
        d1 = self.DeUpConv1(d2_conv1)  # [B, C, 128, 128]
        d1_conv_input = self.DeCatConv1(torch.cat((d1, x1_conv1), dim=1))  # [B, C, 128, 128]
        d1_trans_input = self.TransUpsample1(d2_trans1)  # [B, 128*128, D]
        d1_trans_input = self.DeCatTrans1(torch.cat((d1_trans_input, x1_trans1), dim=-1))  # [B, 128*128, D]
        d1_conv1, d1_trans1 = self.DeTransConv_1(d1_conv_input, d1_trans_input)  # [B, C, 128, 128], [B, 128*128, D]

        # last fuse:
        f1 = self.LastFuse(d1_conv1, d1_trans1)
        out = self.classify(f1)
        return [out]


if __name__ == '__main__':
    model = UConFormer(basic_channel=32, embed_dim=32, window_size=8, conv_type='res', dw=False, method='cat')
    dummy_input = torch.randn(1, 1, 256, 256, dtype=torch.float)
    y = model(dummy_input)
    print(y[0].shape)
