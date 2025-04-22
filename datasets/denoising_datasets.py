import torch
from torch.utils.data import Dataset
from base_datasets import BaseCIFAR10

class DenoisingCIFAR10(BaseCIFAR10) :
    def __init__(self, root, train, noise, transform=None) :
        super().__init__(root, train, transform)
        assert noise > 0
        self.noise = noise

    def __getitem__(self, idx) :
        image, label = super().__getitem__(idx)
        noisy_image = torch.poisson(image * self.noise) / self.noise
        return noisy_image, image