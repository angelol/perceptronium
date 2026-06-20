import torch
import torch.nn as nn

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
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ChannelAttention(nn.Module):
    """
    Upgraded Channel Attention (CBAM) using SiLU activation for better gradient flow.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.SiLU(inplace=True),
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


class MBConvBlock(nn.Module):
    """
    Upgraded Inverted Bottleneck Block with optional Squeeze-and-Excitation (SE)
    or CBAM Attention, Stochastic Depth (DropPath), and variable kernel size.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, 
                 kernel_size=3, attention_type="se", drop_prob=0.0):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio
        padding = kernel_size // 2
        
        # 1. Expansion: 1x1 Conv (skip if expand_ratio is 1)
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # 2. Depthwise Convolution: kernel_size x kernel_size grouped by channels
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
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
            nn.BatchNorm2d(out_channels)
        )
        
        # 5. Stochastic Depth / DropPath (only applied on residual connections)
        self.drop_path = DropPath(drop_prob) if self.use_residual and drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.attention(out)
        out = self.project_conv(out)
        
        if self.use_residual:
            return x + self.drop_path(out)
        return out


class CBAM_EfficientNet(nn.Module):
    """
    Upgraded CBAM-EfficientNet (v2): A deep, from-scratch vision CNN featuring
    multi-block MBConv stage configurations, Squeeze-and-Excitation / upgraded CBAM attention,
    5x5 depthwise kernels, linearly-decaying Stochastic Depth, and a 1x1 pre-classifier head projection.
    """
    def __init__(self, attention_type="se", max_drop_path=0.2):
        super(CBAM_EfficientNet, self).__init__()
        
        # Stem: Input (3 x 224 x 224) -> Output (32 x 112 x 112)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Define MBConv configurations
        # Format: (in_channels, out_channels, num_repeats, stride, expand_ratio, kernel_size)
        self.stage_configs = [
            (32, 32, 1, 1, 1, 3),   # Stage 1
            (32, 64, 2, 2, 4, 3),   # Stage 2
            (64, 128, 3, 2, 4, 3),  # Stage 3
            (128, 256, 3, 2, 4, 5), # Stage 4: using 5x5 depthwise filters
            (256, 512, 2, 2, 6, 5)  # Stage 5: using 5x5 depthwise filters
        ]
        
        # Calculate total number of MBConv blocks for Stochastic Depth scaling
        total_blocks = sum(cfg[2] for cfg in self.stage_configs)
        block_idx = 0
        
        # Build stages dynamically
        stages = []
        for in_c, out_c, repeats, stride, expand, kernel in self.stage_configs:
            stage_blocks = []
            for i in range(repeats):
                # Only downsample/change channels in the first block of the stage
                b_stride = stride if i == 0 else 1
                b_in_c = in_c if i == 0 else out_c
                
                # Linearly decaying drop path probability
                drop_prob = max_drop_path * (block_idx / (total_blocks - 1)) if total_blocks > 1 else 0.0
                
                stage_blocks.append(
                    MBConvBlock(
                        in_channels=b_in_c,
                        out_channels=out_c,
                        stride=b_stride,
                        expand_ratio=expand,
                        kernel_size=kernel,
                        attention_type=attention_type,
                        drop_prob=drop_prob
                    )
                )
                block_idx += 1
            stages.append(nn.Sequential(*stage_blocks))
            
        self.stages = nn.Sequential(*stages)
        
        # Pre-classifier Head Projection: 512 x 7 x 7 -> 1280 x 7 x 7
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
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
        x = self.head_conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


# Backwards compatibility aliases
CustomCNN = CBAM_EfficientNet
ResidualCNN = CBAM_EfficientNet
MBConvCBAM = MBConvBlock # Alias for compatibility with code imports


if __name__ == "__main__":
    # Dimension verification round-trip test
    print("Testing Upgraded CBAM_EfficientNet dimensions with a dummy batch...")
    
    # Test with SE attention (default)
    model_se = CBAM_EfficientNet(attention_type="se")
    print(f"Total model parameters (SE Attention): {sum(p.numel() for p in model_se.parameters()):,}")
    
    # Test with CBAM attention
    model_cbam = CBAM_EfficientNet(attention_type="cbam")
    print(f"Total model parameters (CBAM Attention): {sum(p.numel() for p in model_cbam.parameters()):,}")
    
    dummy_input = torch.randn(4, 3, 224, 224) # Batch size of 4
    output = model_se(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1), f"Unexpected output shape: {output.shape}"
    print("✓ Model dimension round-trip validation successful!")
