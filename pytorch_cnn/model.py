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
    Highly optimized for MBConv structures with customizable reduction ratio and activation.
    """
    def __init__(self, in_planes, ratio=4, act_layer=nn.SiLU):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=True),
            act_layer(),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ChannelAttention(nn.Module):
    """
    Upgraded Channel Attention (CBAM) using SiLU/Mish activation for better gradient flow.
    """
    def __init__(self, in_planes, ratio=16, act_layer=nn.SiLU):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            act_layer(),
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
    Convolutional Block Attention Module (CBAM).
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7, act_layer=nn.SiLU):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio=reduction, act_layer=act_layer)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FusedMBConvBlock(nn.Module):
    """
    Fused Inverted Bottleneck Block (fuses 1x1 expansion and 3x3 depthwise into a single 3x3 conv).
    Optimized to use standard BatchNorm2d and SiLU (Swish) for high stability and fast GPU convergence.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, 
                 kernel_size=3, attention_type="se", drop_prob=0.0,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU):
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
            
        # Attention Layer (Optimized Squeeze-and-Excitation with reduction ratio = 16)
        if attention_type.lower() == "se":
            self.attention = SqueezeExcitation(mid_channels, ratio=16, act_layer=act_layer)
        elif attention_type.lower() == "cbam":
            self.attention = CBAM(mid_channels, reduction=16, act_layer=act_layer)
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
    Inverted Bottleneck Block with Squeeze-and-Excitation (SE) or CBAM Attention,
    Stochastic Depth (DropPath), variable kernel size, and robust BatchNorm2d + SiLU.
    Reverting standard CNN stages back to BatchNorm2d restores training stability and speed.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, 
                 kernel_size=3, attention_type="se", drop_prob=0.0,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU):
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
        
        # 3. Attention Layer (Optimized Squeeze-and-Excitation with ratio = 16)
        if attention_type.lower() == "se":
            self.attention = SqueezeExcitation(mid_channels, ratio=16, act_layer=act_layer)
        elif attention_type.lower() == "cbam":
            self.attention = CBAM(mid_channels, reduction=16, act_layer=act_layer)
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


class TransformerBlock2D(nn.Module):
    """
    Highly optimized 2D Transformer Block with Pre-LayerNorm, 2D Multi-Head Self-Attention (MHSA),
    Stochastic Depth, a complete Feed-Forward Network (FFN), and a robust dual residual pathway.
    Fully compatible with progressive resizing (interpolates positional embeddings dynamically).
    """
    def __init__(self, channels, width=7, height=7, num_heads=8, mlp_ratio=4, dropout=0.1, drop_path_prob=0.0):
        super(TransformerBlock2D, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        # Pre-attention normalization
        self.norm1 = LayerNorm2d(channels)
        
        # Multi-Head Self-Attention layers
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Learnable 2D absolute position embeddings: shape (1, channels, height, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, height, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Pre-FFN normalization
        self.norm2 = LayerNorm2d(channels)
        
        # Feed-Forward Network (MLP)
        mid_features = channels * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, mid_features, kernel_size=1, bias=True),
            nn.SiLU(),  # Highly stable activation
            nn.Dropout(dropout),
            nn.Conv2d(mid_features, channels, kernel_size=1, bias=True),
            nn.Dropout(dropout)
        )
        
        # Stochastic Depth for regularization
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. Multi-Head Self-Attention Path with Pre-LayerNorm ---
        residual = x
        norm_x = self.norm1(x)
        
        # Interpolate positional embeddings dynamically to match spatial dimensions (resolves progressive resizing)
        if H != self.height or W != self.width:
            pos_embed = F.interpolate(
                self.pos_embed, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            pos_embed = self.pos_embed
            
        norm_x = norm_x + pos_embed
        
        # Compute Queries, Keys, Values: shape (B, 3*C, H, W)
        qkv = self.qkv_conv(norm_x)
        q, k, v = torch.chunk(qkv, 3, dim=1) # each is (B, C, H, W)
        
        # Reshape to (B, num_heads, head_dim, H*W) and transpose to (B, num_heads, H*W, head_dim)
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2) # (B, num_heads, N, head_dim)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2) # (B, num_heads, N, head_dim)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2) # (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Output calculation
        out = attn @ v
        
        # Transpose back to (B, num_heads, head_dim, N) and reshape to (B, C, H, W)
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        
        # Projection and Dropout
        out = self.proj_conv(out)
        out = self.proj_dropout(out)
        
        # First Residual + Stochastic Depth
        x = residual + self.drop_path(out)
        
        # --- 2. Feed-Forward Network Path with Pre-LayerNorm ---
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        return x


# Legacy placeholder alias for backwards compatibility
MHSA2D = TransformerBlock2D


class CBAM_EfficientNet(nn.Module):
    """
    CBAM-EfficientNet (v4 Ultra Hybrid): An elite vision hybrid CNN-Transformer model
    trained completely from scratch. Scaled block repetitions, optimized standard 
    stages using BatchNorm2d and SiLU, an integrated 2D Transformer Block with robust
    gradient highways, and a wider Pre-Classifier Projection head push representational 
    capacity to ~22.46M parameters to target 95% validation accuracy.
    """
    def __init__(self, attention_type="se", max_drop_path=0.2):
        super(CBAM_EfficientNet, self).__init__()
        
        # Stem: Input (3 x 224 x 224) -> Output (40 x 112 x 112)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU()
        )
        
        # Scaled Stage Configurations: We scale repeats to [2, 3, 5, 6, 3] to double representational capacity
        # Format: (in_channels, out_channels, num_repeats, stride, expand_ratio, kernel_size, block_type, norm_layer)
        self.stage_configs = [
            (40, 40, 2, 1, 1, 3, "fused", nn.BatchNorm2d),      # Stage 1
            (40, 80, 3, 2, 4, 3, "fused", nn.BatchNorm2d),      # Stage 2
            (80, 160, 5, 2, 4, 3, "fused", nn.BatchNorm2d),     # Stage 3
            (160, 320, 6, 2, 4, 5, "standard", nn.BatchNorm2d), # Stage 4: BatchNorm2d restores convergence speed
            (320, 480, 3, 2, 6, 5, "standard", nn.BatchNorm2d)  # Stage 5: BatchNorm2d restores convergence speed
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
                            act_layer=nn.SiLU
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
                            act_layer=nn.SiLU
                        )
                    )
                block_idx += 1
            stages.append(nn.Sequential(*stage_blocks))
            
        self.stages = nn.Sequential(*stages)
        
        # 2D Transformer Block with Pre-LN, Self-Attention, and FFN
        # Standardly placed at the deepest 7x7 spatial resolution before global pool.
        self.transformer = TransformerBlock2D(
            channels=480, width=7, height=7, num_heads=8, mlp_ratio=4, dropout=0.1, drop_path_prob=max_drop_path
        )
        
        # Pre-classifier Head Projection: Projects 480 to 1536 channels (+20% feature space capacity)
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=480, out_channels=1536, kernel_size=1, bias=False),
            nn.BatchNorm2d(1536),
            nn.SiLU()
        )
        
        # Global Avg Pooling: collapses spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Single-linear classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1536, out_features=1)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.transformer(x)
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
    print("Testing Upgraded CBAM_EfficientNet v4 (Ultra Hybrid) dimensions with a dummy batch...")
    
    # Test with SE attention (default)
    model_se = CBAM_EfficientNet(attention_type="se")
    print(f"Total model parameters (SE Attention): {sum(p.numel() for p in model_se.parameters()):,}")
    
    # Test with CBAM attention
    model_cbam = CBAM_EfficientNet(attention_type="cbam")
    print(f"Total model parameters (CBAM Attention): {sum(p.numel() for p in model_cbam.parameters()):,}")
    
    # Test multiple input shapes to verify progressive resizing / interpolation in Transformer
    for sz in [128, 192, 224]:
        dummy_input = torch.randn(4, 3, sz, sz) # Batch size of 4
        output = model_se(dummy_input)
        print(f"Input size: {sz}x{sz} | Output shape: {output.shape}")
        assert output.shape == (4, 1), f"Unexpected output shape: {output.shape}"
        
    print("✓ Model v4 dimension round-trip validation successful!")
