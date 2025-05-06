import torch
from datasets.base_datasets import BaseCIFAR10

class OutpaintingCIFAR10(BaseCIFAR10):
    def __init__(self, root, train, download, transform=None, subset_size=None, visible_ratio=0.75):
        super().__init__(root, train, download, transform, subset_size)
        self.visible_ratio = visible_ratio

    def _create_mask(self, image):
        C, H, W = image.shape
        visible_H = int(H * (self.visible_ratio ** 0.5))
        visible_W = int(W * (self.visible_ratio ** 0.5))
        top = (H - visible_H) // 2
        left = (W - visible_W) // 2
        mask = torch.zeros_like(image)
        mask[:, top:top+visible_H, left:left+visible_W] = 1
        return mask

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        
        mask = self._create_mask(image)
        masked_image = image * mask
        return image, masked_image, mask
