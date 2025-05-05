import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets

class BaseCIFAR10(Dataset) :
    def __init__(self, root, train, transform=None, subset_size=None) :
        self.data = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        if subset_size is not None :
            self.data = torch.utils.data.Subset(self.data, range(subset_size))

    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        image, label = self.data[idx]
        if self.transform :
            image = self.transform(image)
        return image, label
