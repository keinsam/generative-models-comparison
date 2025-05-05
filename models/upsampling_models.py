import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """
    Bloc résiduel pour le générateur SR
    """

    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Connexion résiduelle
        return out


class PixelShuffle(nn.Module):
    """
    Couche d'upsampling utilisant PixelShuffle (équivalent à depth_to_space en TensorFlow)
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return F.pixel_shuffle(x, self.upscale_factor)


class Generator(nn.Module):
    """
    Générateur pour SRWGAN adapté à une architecture DCGAN à 4 couches

    Input:
        - Image basse résolution: forme par défaut (3, 64, 64)
          Valeurs dans l'intervalle (-1, 1)
    Output:
        - Image haute résolution: forme par défaut (3, 256, 256)
          Valeurs dans l'intervalle (-1, 1)
    """

    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()

        # Nombre de blocs d'upsampling basé sur le facteur d'échelle
        self.upsample_block_num = int(math.log(scale_factor, 2))

        # Première couche - extraction de caractéristiques
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Blocs résiduels - DCGAN utilise généralement 4 couches
        # Nous allons utiliser 4 blocs résiduels au lieu des 16 du code original
        residual_blocks = []
        for _ in range(4):
            residual_blocks.append(ResidualBlock(64))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # Couche de convolution après les blocs résiduels
        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Couches d'upsampling avec PixelShuffle
        upsampling = []
        for _ in range(self.upsample_block_num):
            upsampling.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                PixelShuffle(2),
                nn.PReLU()
            ])

        # Couche de sortie
        upsampling.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsampling = nn.Sequential(*upsampling)

        # Activation tanh pour la sortie
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.conv_input(x)
        out = self.residual_blocks(out1)
        out2 = self.conv_mid(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        return self.tanh(out)


# Fonction d'initialisation des poids pour une meilleure convergence
def weights_init_gen(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminateur pour SRWGAN adapté à une architecture DCGAN à 4 couches

    Input:
        - Image haute résolution: forme par défaut (3, 256, 256)
          Valeurs dans l'intervalle (-1, 1)
    Output:
        - Scalaire sans activation finale (pour distance Wasserstein)
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Paramètres standards de DCGAN
        self.ndf = 64  # Nombre de filtres de base (caractéristique de DCGAN)
        self.alpha = 0.2  # Pente négative pour LeakyReLU

        # Première couche - sans normalisation par batch (comme DCGAN)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(self.alpha, inplace=True)
        )

        # Deuxième couche
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(self.alpha, inplace=True)
        )

        # Troisième couche
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(self.alpha, inplace=True)
        )

        # Quatrième couche
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(self.alpha, inplace=True)
        )

        # Couche de sortie - pas d'activation pour la distance Wasserstein
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.ndf * 8 * 8 * 8, 1)  # Taille adaptée pour une entrée de 256x256
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x  # Pas d'activation finale pour Wasserstein


# Fonction d'initialisation des poids (typique pour DCGAN)
def weights_init_dis(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


