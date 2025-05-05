import torch
import torch.nn as nn

class BaseCritic(nn.Module):
    def __init__(self, img_channels):
        super(BaseCritic, self).__init__()
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            # self._block(256, 512, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class BaseGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(BaseGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # Input: N x latent_dim x 1 x 1
            self._block(latent_dim, 512, 4, 1, 0),  # img: 4x4
            # self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(512, 256, 4, 2, 1),  # img: 16x16
            self._block(256, 128, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)