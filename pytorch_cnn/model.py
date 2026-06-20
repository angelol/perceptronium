import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Computes Channel Attention by taking both Average and Maximum spatial pooled outputs,
    passing them through a shared MLP, and scaling the feature maps.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # Shared MLP layers using Conv2d with kernel size 1
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
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
    Computes Spatial Attention by concatenating channel-average and channel-max pooling maps,
    convolving them with a 7x7 filter, and scaling the feature maps.
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
    Convolutional Block Attention Module (CBAM) combining Channel and Spatial Attention.
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class MBConvCBAM(nn.Module):
    """
    Inverted Bottleneck Block with depthwise separable convolution
    and integrated CBAM spatial/channel attention.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super(MBConvCBAM, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio
        
        # 1. Expansion: 1x1 Conv
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 2. Depthwise Convolution: 3x3 Conv grouped by expanded channels
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3. CBAM Attention Over Expanded Space
        self.cbam = CBAM(mid_channels, reduction=16)
        
        # 4. Pointwise Linear Projection: 1x1 Conv
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.cbam(out)
        out = self.project_conv(out)
        
        if self.use_residual:
            return out + x
        return out


class CBAM_EfficientNet(nn.Module):
    """
    CBAM-EfficientNet (Option C): A highly compressed (~1.27M params), from-scratch
    deep vision CNN leveraging Depthwise Separation, Inverted Bottlenecks (MBConv),
    and Convolutional Block Attention (CBAM) to maximize generalization and eliminate overfitting.
    """
    def __init__(self):
        super(CBAM_EfficientNet, self).__init__()
        
        # Stem: Input (3 x 224 x 224) -> Output (32 x 112 x 112)
        # We use stride=2 to downsample smoothly instead of immediate max pooling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: Input (32 x 112 x 112) -> Output (64 x 56 x 56)
        self.stage1 = MBConvCBAM(in_channels=32, out_channels=64, stride=2, expand_ratio=4)
        
        # Stage 2: Input (64 x 56 x 56) -> Output (128 x 28 x 28)
        self.stage2 = MBConvCBAM(in_channels=64, out_channels=128, stride=2, expand_ratio=4)
        
        # Stage 3: Input (128 x 28 x 28) -> Output (256 x 14 x 14)
        self.stage3 = MBConvCBAM(in_channels=128, out_channels=256, stride=2, expand_ratio=4)
        
        # Stage 4: Input (256 x 14 x 14) -> Output (512 x 7 x 7)
        self.stage4 = MBConvCBAM(in_channels=256, out_channels=512, stride=2, expand_ratio=4)
        
        # Global Pooling: collapses final spatial resolution to (512 x 1 x 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head: fully connected layers with Dropout regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=512, out_features=64),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=1)
        )
        
    def forward(self, x):
        # Initial stem
        x = self.stem(x)
        
        # Attentive MBConv stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Compress and flatten
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        # Output binary logit
        logits = self.classifier(x)
        return logits


# Backwards compatibility aliases
CustomCNN = CBAM_EfficientNet
ResidualCNN = CBAM_EfficientNet


if __name__ == "__main__":
    # Dimension verification round-trip test
    print("Testing CBAM_EfficientNet dimensions with a dummy batch...")
    model = CBAM_EfficientNet()
    dummy_input = torch.randn(4, 3, 224, 224) # Batch size of 4
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1), f"Unexpected output shape: {output.shape}"
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Model dimension round-trip validation successful!")
