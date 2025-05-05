import torch
from datasets.base_datasets import BaseCIFAR10

class OutpaintingCIFAR10(BaseCIFAR10):
    def __init__(self, root, train, download, transform=None, subset_size=None, visible_ratio=0.75):
        super().__init__(root, train, download, transform, subset_size)
        self.visible_ratio = visible_ratio  # e.g. 0.75 keeps 75% center

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform is not None :
            image = self.transform(image)

        C, H, W = image.shape

        # Compute size of visible center patch
        visible_H = int(H * (self.visible_ratio ** 0.5))
        visible_W = int(W * (self.visible_ratio ** 0.5))
        top = (H - visible_H) // 2
        left = (W - visible_W) // 2

        # Create masked input
        masked_image = torch.zeros_like(image)
        masked_image[:, top:top+visible_H, left:left+visible_W] = image[:, top:top+visible_H, left:left+visible_W]

        return masked_image, image
