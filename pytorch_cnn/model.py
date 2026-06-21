import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth (DropPath) for regularizing deep networks.
    Randomly drops entire residual branches during training.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, 1, 1, 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize to 0 or 1
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D image inputs (channels-first, channel-wise).
    Computes mean/variance along the channel dimension only, i.e., at each pixel.
    """
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    Squeezes spatial dimensions and excites channels using a two-layer MLP.
    Highly optimized for MBConv structures with customizable reduction ratio.
    """
    def __init__(self, in_planes, ratio=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=True),
            nn.Mish(),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ChannelAttention(nn.Module):
    """
    Upgraded Channel Attention (CBAM) using Mish activation for better gradient flow.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.Mish(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Standard Spatial Attention (CBAM).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Upgraded Convolutional Block Attention Module (CBAM).
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FusedMBConvBlock(nn.Module):
    """
    Fused Inverted Bottleneck Block (fuses 1x1 expansion and 3x3 depthwise into a single 3x3 conv).
    Commonly used in early stages of EfficientNetV2 for higher GPU arithmetic intensity.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, 
                 kernel_size=3, attention_type="se", drop_prob=0.0,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.Mish):
        super(FusedMBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio
        padding = kernel_size // 2
        
        # Fused convolution combining expansion and spatial learning
        if expand_ratio != 1:
            self.fused_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, 
                          stride=stride, padding=padding, bias=False),
                norm_layer(mid_channels),
                act_layer()
            )
        else:
            # If no expansion, do a regular standard 3x3 conv
            self.fused_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False),
                norm_layer(out_channels),
                act_layer()
            )
            mid_channels = out_channels
            
        # Attention Layer
        if attention_type.lower() == "se":
            self.attention = SqueezeExcitation(mid_channels, ratio=16)
        elif attention_type.lower() == "cbam":
            self.attention = CBAM(mid_channels, reduction=16)
        else:
            self.attention = nn.Identity()
            
        # Projection convolution (only if expand_ratio != 1)
        if expand_ratio != 1:
            self.project_conv = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels)
            )
        else:
            self.project_conv = nn.Identity()
            
        # Stochastic Depth / DropPath (applied only on residual connections)
        self.drop_path = DropPath(drop_prob) if self.use_residual and drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.fused_conv(x)
        out = self.attention(out)
        out = self.project_conv(out)
        
        if self.use_residual:
            return x + self.drop_path(out)
        return out


class MBConvBlock(nn.Module):
    """
    Upgraded Inverted Bottleneck Block with optional Squeeze-and-Excitation (SE)
    or CBAM Attention, Stochastic Depth (DropPath), variable kernel size,
    and customizable normalization/activation functions (supporting LayerNorm2d and Mish).
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, 
                 kernel_size=3, attention_type="se", drop_prob=0.0,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.Mish):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio
        padding = kernel_size // 2
        
        # 1. Expansion: 1x1 Conv (skip if expand_ratio is 1)
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                norm_layer(mid_channels),
                act_layer()
            )
        else:
            self.expand_conv = nn.Identity()
        
        # 2. Depthwise Convolution: kernel_size x kernel_size grouped by channels
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, groups=mid_channels, bias=False),
            norm_layer(mid_channels),
            act_layer()
        )
        
        # 3. Attention Layer
        if attention_type.lower() == "se":
            self.attention = SqueezeExcitation(mid_channels, ratio=16)
        elif attention_type.lower() == "cbam":
            self.attention = CBAM(mid_channels, reduction=16)
        else:
            self.attention = nn.Identity()
        
        # 4. Pointwise Linear Projection: 1x1 Conv
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )
        
        # 5. Stochastic Depth / DropPath (applied only on residual connections)
        self.drop_path = DropPath(drop_prob) if self.use_residual and drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.attention(out)
        out = self.project_conv(out)
        
        if self.use_residual:
            return x + self.drop_path(out)
        return out


class MHSA2D(nn.Module):
    """
    2D Multi-Head Self-Attention (MHSA) with learnable absolute positional embeddings.
    Designed for small spatial dimensions (e.g. 7x7 grid) at the end of Stage 5.
    """
    def __init__(self, channels, width=7, height=7, num_heads=4, dropout=0.1):
        super(MHSA2D, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Learnable 2D absolute position embeddings: shape (1, channels, height, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, height, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Interpolate positional embeddings if incoming shape does not match (self.height, self.width)
        if H != self.height or W != self.width:
            pos_embed = F.interpolate(
                self.pos_embed, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed
        
        # Compute Queries, Keys, Values: shape (B, 3*C, H, W)
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1) # each is (B, C, H, W)
        
        # Reshape to (B, num_heads, head_dim, H*W) and transpose to (B, num_heads, H*W, head_dim)
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2) # (B, num_heads, N, head_dim)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2) # (B, num_heads, N, head_dim)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2) # (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        # Query: (B, h, N, d), Key: (B, h, N, d) -> Attn matrix: (B, h, N, N)
        attn = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Attn: (B, h, N, N), Value: (B, h, N, d) -> Output: (B, h, N, d)
        out = attn @ v
        
        # Transpose back to (B, num_heads, head_dim, N) and reshape to (B, C, H, W)
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        
        # Projection
        out = self.proj_conv(out)
        out = self.proj_dropout(out)
        return out


class CBAM_EfficientNet(nn.Module):
    """
    Upgraded CBAM-EfficientNet (v3): A deep, from-scratch vision hybrid CNN-Transformer featuring
    fused MBConv stages (Stages 1-3), standard MBConv stages with LayerNorm2d and Mish (Stages 4-5),
    2D Multi-Head Self-Attention, and a robust pre-classifier LayerNorm2d head.
    """
    def __init__(self, attention_type="se", max_drop_path=0.2):
        super(CBAM_EfficientNet, self).__init__()
        
        # Stem: Input (3 x 224 x 224) -> Output (40 x 112 x 112)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.Mish()
        )
        
        # Define MBConv / Fused configurations
        # Format: (in_channels, out_channels, num_repeats, stride, expand_ratio, kernel_size, block_type, norm_layer)
        self.stage_configs = [
            (40, 40, 1, 1, 1, 3, "fused", nn.BatchNorm2d),      # Stage 1
            (40, 80, 2, 2, 4, 3, "fused", nn.BatchNorm2d),      # Stage 2
            (80, 160, 3, 2, 4, 3, "fused", nn.BatchNorm2d),     # Stage 3
            (160, 320, 4, 2, 4, 5, "standard", LayerNorm2d),    # Stage 4
            (320, 480, 2, 2, 6, 5, "standard", LayerNorm2d)     # Stage 5
        ]
        
        # Calculate total number of blocks for Stochastic Depth scaling
        total_blocks = sum(cfg[2] for cfg in self.stage_configs)
        block_idx = 0
        
        # Build stages dynamically
        stages = []
        for in_c, out_c, repeats, stride, expand, kernel, b_type, norm_layer in self.stage_configs:
            stage_blocks = []
            for i in range(repeats):
                # Only downsample/change channels in the first block of the stage
                b_stride = stride if i == 0 else 1
                b_in_c = in_c if i == 0 else out_c
                
                # Linearly decaying drop path probability
                drop_prob = max_drop_path * (block_idx / (total_blocks - 1)) if total_blocks > 1 else 0.0
                
                if b_type == "fused":
                    stage_blocks.append(
                        FusedMBConvBlock(
                            in_channels=b_in_c,
                            out_channels=out_c,
                            stride=b_stride,
                            expand_ratio=expand,
                            kernel_size=kernel,
                            attention_type=attention_type,
                            drop_prob=drop_prob,
                            norm_layer=norm_layer,
                            act_layer=nn.Mish
                        )
                    )
                else:
                    stage_blocks.append(
                        MBConvBlock(
                            in_channels=b_in_c,
                            out_channels=out_c,
                            stride=b_stride,
                            expand_ratio=expand,
                            kernel_size=kernel,
                            attention_type=attention_type,
                            drop_prob=drop_prob,
                            norm_layer=norm_layer,
                            act_layer=nn.Mish
                        )
                    )
                block_idx += 1
            stages.append(nn.Sequential(*stage_blocks))
            
        self.stages = nn.Sequential(*stages)
        
        # 2D Multi-Head Self-Attention at the end of Stage 5 (480 channels, 7x7 spatial layout)
        self.mhsa = MHSA2D(channels=480, width=7, height=7, num_heads=4, dropout=0.1)
        
        # Pre-classifier Head Projection: 480 x 7 x 7 -> 1280 x 7 x 7
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=480, out_channels=1280, kernel_size=1, bias=False),
            LayerNorm2d(1280),
            nn.Mish()
        )
        
        # Global Avg Pooling: collapses spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Single-linear classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1280, out_features=1)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.mhsa(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


# Backwards compatibility aliases
CustomCNN = CBAM_EfficientNet
ResidualCNN = CBAM_EfficientNet
MBConvCBAM = MBConvBlock


if __name__ == "__main__":
    # Dimension verification round-trip test
    print("Testing Upgraded CBAM_EfficientNet v3 dimensions with a dummy batch...")
    
    # Test with SE attention (default)
    model_se = CBAM_EfficientNet(attention_type="se")
    print(f"Total model parameters (SE Attention): {sum(p.numel() for p in model_se.parameters()):,}")
    
    # Test with CBAM attention
    model_cbam = CBAM_EfficientNet(attention_type="cbam")
    print(f"Total model parameters (CBAM Attention): {sum(p.numel() for p in model_cbam.parameters()):,}")
    
    # Test multiple input shapes to verify progressive resizing / interpolation in MHSA
    for sz in [128, 192, 224]:
        dummy_input = torch.randn(4, 3, sz, sz) # Batch size of 4
        output = model_se(dummy_input)
        print(f"Input size: {sz}x{sz} | Output shape: {output.shape}")
        assert output.shape == (4, 1), f"Unexpected output shape: {output.shape}"
        
    print("✓ Model v3 dimension round-trip validation successful!")
