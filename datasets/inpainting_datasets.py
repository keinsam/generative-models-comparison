import torch
from datasets.base_datasets import BaseCIFAR10

class InpaintingCIFAR10(BaseCIFAR10):
    def __init__(self, root, train, download, transform=None, subset_size=None, mask_ratio=0.25):
        super().__init__(root, train, download, transform, subset_size)
        self.mask_ratio = mask_ratio

    def _create_mask(self, image):
        C, H, W = image.shape
        mask_H = int(H * self.mask_ratio ** 0.5)
        mask_W = int(W * self.mask_ratio ** 0.5)
        top = (H - mask_H) // 2
        left = (W - mask_W) // 2
        mask = torch.ones_like(image)
        mask[:, top:top+mask_H, left:left+mask_W] = 0
        return mask

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        mask = self._create_mask(image)
        masked_image = image * mask
        return masked_image, image, mask  # input, target, mask
