import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Define a sequence of layers: Conv2d, BatchNorm, ReLU, Conv2d, BatchNorm, ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 3x3 convolution with padding
            nn.BatchNorm2d(mid_channels), # Batch normalization
            nn.ReLU(inplace=True), # ReLU activation
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # Second 3x3 convolution
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.ReLU(inplace=True) # ReLU activation
        )

    def forward(self, x):
        """Forward pass through the double convolution block."""
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define a sequence: MaxPool2d then DoubleConv
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # Max pooling with 2x2 kernel and stride 2
            DoubleConv(in_channels, out_channels) # Apply double convolution block
        )

    def forward(self, x):
        """Forward pass through the downscaling block."""
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling block: Upsample/ConvTranspose followed by DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Define the upsampling method
        if bilinear:
            # Bilinear upsampling followed by a convolution to halve channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution to upsample and halve channels
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the upscaling block.
        Args:
            x1: Feature map from the previous layer in the decoder.
            x2: Feature map from the corresponding layer in the encoder (skip connection).
        """
        # Upsample x1
        x1 = self.up(x1)

        # Calculate padding needed to concatenate skip connection (x2)
        # Input tensor format is (Batch, Channels, Height, Width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Apply padding to x1 to match the spatial dimensions of x2
        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2, # Left, Right padding
                        diffY // 2, diffY - diffY // 2]) # Top, Bottom padding

        # Concatenate the upsampled features (x1) and the skip connection features (x2) along the channel dimension
        x = torch.cat([x2, x1], dim=1)

        # Apply double convolution block
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution layer (1x1)"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # Define a single 1x1 convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the output convolution."""
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net Architecture.
    Processes input images (e.g., RGB) and outputs segmentation masks.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        Initialize the U-Net.
        Args:
            n_channels (int): Number of input channels (e.g., 3 for RGB).
            n_classes (int): Number of output classes (e.g., 1 for binary segmentation).
            bilinear (bool): Whether to use bilinear upsampling or transposed convolutions.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- Encoder Path ---
        self.inc = DoubleConv(n_channels, 64)      # Initial convolution block
        self.down1 = Down(64, 128)                 # First downscaling block
        self.down2 = Down(128, 256)                # Second downscaling block
        self.down3 = Down(256, 512)                # Third downscaling block
        factor = 2 if bilinear else 1              # Adjust channel counts based on upsampling method
        self.down4 = Down(512, 1024 // factor)     # Fourth downscaling block (bottleneck)

        # --- Decoder Path ---
        self.up1 = Up(1024, 512 // factor, bilinear) # First upscaling block
        self.up2 = Up(512, 256 // factor, bilinear)  # Second upscaling block
        self.up3 = Up(256, 128 // factor, bilinear)  # Third upscaling block
        self.up4 = Up(128, 64, bilinear)             # Fourth upscaling block

        # --- Output Layer ---
        self.outc = OutConv(64, n_classes)           # Final 1x1 convolution to get class scores/logits

    def forward(self, x):
        """Defines the forward pass of the U-Net."""
        # --- Encoder ---
        x1 = self.inc(x)    # Pass through initial block
        x2 = self.down1(x1) # Pass through first down block
        x3 = self.down2(x2) # Pass through second down block
        x4 = self.down3(x3) # Pass through third down block
        x5 = self.down4(x4) # Pass through bottleneck block

        # --- Decoder + Skip Connections ---
        x = self.up1(x5, x4) # Upsample bottleneck, concatenate with x4 (skip), apply convs
        x = self.up2(x, x3)  # Upsample, concatenate with x3 (skip), apply convs
        x = self.up3(x, x2)  # Upsample, concatenate with x2 (skip), apply convs
        x = self.up4(x, x1)  # Upsample, concatenate with x1 (skip), apply convs

        # --- Final Output ---
        logits = self.outc(x) # Apply final 1x1 convolution
        # For binary segmentation, typically apply Sigmoid activation *after* this in the loss function or evaluation
        return logits

# --- Example Usage (for testing the model definition) ---
if __name__ == '__main__':
    # Define parameters
    BATCH_SIZE = 4
    INPUT_CHANNELS = 3 # RGB Images
    NUM_CLASSES = 1    # Binary segmentation (polyp vs background)
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    # Create a dummy input tensor (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    # Instantiate the model
    model = UNet(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES, bilinear=True)

    # Perform a forward pass
    print("Performing forward pass...")
    with torch.no_grad(): # No need to calculate gradients for this test
        output_logits = model(dummy_input)

    # Print shapes to verify
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output_logits.shape}") # Should be (BATCH_SIZE, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH)

    # Check number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Optionally apply sigmoid to see output range (for binary case)
    output_probs = torch.sigmoid(output_logits)
    print(f"Output probabilities Min: {output_probs.min():.4f}, Max: {output_probs.max():.4f}")

