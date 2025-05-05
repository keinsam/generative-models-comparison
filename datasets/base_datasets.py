import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets

class BaseCIFAR10(Dataset) :
    def __init__(self, root, train, download, transform=None, subset_size=None) :
        super().__init__()
        self.download = download
        self.data = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        if subset_size is not None :
            self.data = Subset(self.data, range(subset_size))

    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        image, label = self.data[idx]
        if self.transform :
            image = self.transform(image)
        return image, label