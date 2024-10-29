import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, noise_channels=100, img_channels=3):
        super().__init__()
        self.conv_layers = nn.Sequential(

            # 1st fractional strided convolution layer (upsample from 1*1 -> 4*4)
            # Projection layer, to convert the z of 100 inputs to 1024 * 4 * 4 (noise_channels = z_dim)
            # Each input (z) will be actually reshaped to 100 * 1 * 1 (100 channels)
            # (to ensure from 1x1 -> 4x4, with stride = 2 and kernal = 4, we need padding = 0 now (for a x4 increase))
            self._block(in_channels=noise_channels, out_channels=1024, kernel_size=4, stride=2, padding=0),

            # 2nd fractional strided convolution layer (upsample from 4*4 -> 8*8)
            self._block(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),

            # 3rd fractional strided convolution layer (upsample from 8*8 -> 16*16)
            self._block(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            
            # 4th fractional strided convolution layer (upsample from 16*16 -> 32*32)
            self._block(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),

            # Output fractional strided convolution layer (upsample from 32*32 -> 64*64)
            nn.ConvTranspose2d(in_channels=128, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ) if batch_norm else nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.conv_layers(z)


class Discriminator(nn.Module):

    def __init__(self, img_channels=3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            
            # 1st fractional strided convolution layer (downsample from 64*64 -> 32*32)
            self._block(in_channels=img_channels, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=False),

            # 2nd fractional strided convolution layer (downsample from 32*32 -> 16*16)
            self._block(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            
            # 3rd fractional strided convolution layer (downsample from 16*16 -> 8*8)
            self._block(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),

            # Output fractional strided convolution layer (downsample from 8*8 -> 4*4)
            self._block(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            
            # Classifier
            # No fully connected layer for DCGAN, use another way (instead of nn.Flatten(), nn.Linear(in_features=1024*4*4, out_features=1))
            # Use another convolutional layer (to ensure from 4x4 to 1x1, with stride = 2 and kernal = 4, we need padding = 0 now (for a x4 reduction))
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid() # ensure prediction is within [0, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        ) if batch_norm else nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        return self.conv_layers(x)