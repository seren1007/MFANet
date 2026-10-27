import torch
import torch.nn as nn
import einops


class ScaleDotProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3)
        return x123


class _DepthwiseProj(nn.Module):
    def __init__(self, channels, kernel_size=(1, 1), padding=(0, 0)):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size,
                              padding=padding, groups=channels, bias=True)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C -> B C H W', H=P)
        x = self.proj(x)
        return einops.rearrange(x, 'B C H W -> B (H W) C')


class PoolEmbedding(nn.Module):
    def __init__(self, patch):
        super().__init__()
        self.proj = nn.AdaptiveAvgPool2d((patch, patch))

    def forward(self, x):
        return einops.rearrange(self.proj(x), 'B C H W -> B (H W) C')


class UpsampleConv(nn.Module):
    def __init__(self, in_features, out_features, scale=(2, 2)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        return self.conv(self.up(x))

class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.q_map = _DepthwiseProj(out_features)
        self.k_map = _DepthwiseProj(in_features)
        self.v_map = _DepthwiseProj(in_features)
        self.projection = _DepthwiseProj(out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q = self.q_map(x[0])
        k = self.k_map(x[1])
        v = self.v_map(x[2])
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        return self.projection(att)


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.q_map = _DepthwiseProj(in_features)
        self.k_map = _DepthwiseProj(in_features)
        self.v_map = _DepthwiseProj(out_features)
        self.projection = _DepthwiseProj(out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q = self.q_map(x[0])
        k = self.k_map(x[1])
        v = self.v_map(x[2])
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        return self.projection(att)


class TCSABlock(nn.Module):
    def __init__(self, features, channel_head, spatial_head,
                 channel_att=True, spatial_att=True):
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:  #TCA
            self.channel_norm = nn.ModuleList([nn.LayerNorm(f, eps=1e-6) for f in features])
            self.c_attention = nn.ModuleList([
                ChannelAttention(in_features=sum(features), out_features=f, n_heads=h)
                for f, h in zip(features, channel_head)])
        if self.spatial_att:  #TSA
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(f, eps=1e-6) for f in features])
            self.s_attention = nn.ModuleList([
                SpatialAttention(in_features=sum(features), out_features=f, n_heads=h)
                for f, h in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x = self.m_sum(x, self.channel_attention(x))
        if self.spatial_att:
            x = self.m_sum(x, self.spatial_attention(x))
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = torch.cat(x_c, dim=2)
        return self.m_apply([[q, x_cin, x_cin] for q in x_c], self.c_attention)

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = torch.cat(x_c, dim=2)
        return self.m_apply([[x_cin, x_cin, v] for v in x_c], self.s_attention)

    @staticmethod
    def m_apply(x, module):
        return [module[i](j) for i, j in enumerate(x)]

    @staticmethod
    def m_sum(x, y):
        return [xi + xj for xi, xj in zip(x, y)]

class TDA(nn.Module):

    def __init__(self, features, strides, patch=28,
                 channel_att=True, spatial_att=True, n=1,
                 channel_head=[1, 1, 1, 1], spatial_head=[4, 4, 4, 4]):
        super().__init__()
        self.n = n
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(patch) for _ in features])
        self.avg_map = nn.ModuleList([_DepthwiseProj(f) for f in features])
        self.attention = nn.ModuleList([
            TCSABlock(features=features, channel_head=channel_head, spatial_head=spatial_head,
                      channel_att=channel_att, spatial_att=spatial_att)
            for _ in range(n)])
        self.upconvs = nn.ModuleList([
            UpsampleConv(in_features=f, out_features=f, scale=s)
            for f, s in zip(features, strides)])
        self.bn_relu = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(f), nn.ReLU()) for f in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [einops.rearrange(i, 'B (H W) C -> B C H W', H=self.patch) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out,)

    @staticmethod
    def m_apply(x, module):
        return [module[i](j) for i, j in enumerate(x)]

    @staticmethod
    def m_sum(x, y):
        return [xi + xj for xi, xj in zip(x, y)]
