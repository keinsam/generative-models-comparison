import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from .base_models import Diffusion,DDPM



class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.channels_img = channels_img
        self.features_d = features_d
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, latent_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels_img = channels_img
        self.features_g = features_g

        # Réseau pour traiter le vecteur latent
        self.latent_net = nn.Sequential(
            self._block(latent_dim, features_g * 8, 4, 1, 0),  # img: 4x4
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 8x8
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 16x16
        )

        # Réseau pour traiter l'image masquée
        self.masked_net = nn.Sequential(
            nn.Conv2d(channels_img, features_g * 2, kernel_size=4, stride=2, padding=1),  # img: 32x32
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            nn.Conv2d(features_g * 2, features_g * 4, kernel_size=4, stride=2, padding=1),  # img: 16x16
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
        )

        # Réseau pour fusionner et générer l'image finale
        self.final_net = nn.Sequential(
            self._block(features_g * 6, features_g * 4, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(features_g * 4, channels_img, kernel_size=4, stride=2, padding=1),  # img: 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise, masked_image):
        # Traiter le vecteur latent
        latent_features = self.latent_net(noise)

        # Traiter l'image masquée
        masked_features = self.masked_net(masked_image)

        # Ajuster les dimensions des tenseurs pour qu'ils puissent être concaténés
        if latent_features.shape[2:] != masked_features.shape[2:]:
            latent_features = nn.functional.interpolate(latent_features, size=masked_features.shape[2:], mode='bilinear', align_corners=False)

        # Fusionner les caractéristiques
        combined_features = torch.cat((latent_features, masked_features), dim=1)

        # Générer l'image finale
        output = self.final_net(combined_features)
        return output


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


