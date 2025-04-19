import torch
from torch.utils.data import Dataset

class BaseCIFAR10(Dataset) :
    def __init__(self, root, train, transform=None)
        self.data = datasets.CIFAR10(root=root, train=train, download=False)
        self.transform = transform
    
    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, idx) :
        image, label = self.data[idx]
        if self.transform :
            image = self.transform(image)
        return image, label