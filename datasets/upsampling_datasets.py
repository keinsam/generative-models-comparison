import numpy as np
from torchvision.transforms import Resize, InterpolationMode, RandomCrop, transforms
from .base_datasets import BaseCIFAR10
import torch
"""
class UpsamplingCIFAR10(BaseCIFAR10) :
    def __init__(self, root, train, transform=None, subset_size=None):
        super().__init__(root, train, transform, subset_size)
        self.downscale = Resize((16, 16), interpolation=InterpolationMode.BICUBIC)
        self.upscale = Resize((256, 256), interpolation=InterpolationMode.BICUBIC)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # Downscale the original image to 64x64
        low_res_image = self.downscale(image)

        # Upscale the low-res image to 256x256
        high_res_image = self.upscale(low_res_image)

        return low_res_image, high_res_image
"""
class UpsamplingCIFAR10(BaseCIFAR10):
    def __init__(self, root, train=True, transform=None, subset_size=None,hr_size=128):
        super().__init__(root, train, transform, subset_size=subset_size)
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # Pas besoin de normaliser à nouveau car les données sont déjà normalisées
        ])

        self.hr_size = hr_size
        self.subset_size = subset_size


    def __len__(self):
        return self.subset_size

    def __getitem__(self, idx):
        # Obtenir l'image CIFAR-10 (32x32) comme image LR
        lr_image, label = super().__getitem__(idx)
        # Créer une version HR "idéale" pour l'entraînement
        # (Utilisation de bicubic upsampling comme cible d'entraînement)
        hr_image = self.hr_transform(lr_image)

        return lr_image, hr_image


