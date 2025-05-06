import torch
from datasets.base_datasets import BaseCIFAR10

class UpsamplingCIFAR10(BaseCIFAR10):
    def __init__(self, root, train, download, transform=None, subset_size=None, scale_factor=2):
        super().__init__(root, train, download, transform, subset_size)
        self.scale_factor = scale_factor

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        low_res = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)
        low_res = torch.nn.functional.interpolate(low_res, size=image.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        return low_res, image  # input = low-res, target = high-res