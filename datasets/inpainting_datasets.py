import os
import shutil
import torch
import random
from torchvision.transforms import ToPILImage
from .base_datasets import BaseCIFAR10
from torchvision.utils import save_image

class InpaintingCIFAR10(BaseCIFAR10):
    def __init__(self, root, train, download, transform=None, subset_size=None, mask_folder=None):
        super().__init__(root, train,download, transform, subset_size)
        self.mask_folder = mask_folder
        self.counter = 0
        if self.mask_folder:
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            else:
                shutil.rmtree(mask_folder)
                os.makedirs(mask_folder)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        # Generate a masked image with a single random block
        masked_image,mask = self.apply_random_mask(image,self.counter)
        self.counter += 1
        return masked_image, image, mask

    def apply_random_mask(self, image, idx):
        _, height, width = image.shape
        mask = torch.ones_like(image)

        # Randomly define the size of the block to mask
        max_block_height = height // 2
        max_block_width = width // 2
        block_height = random.randint(height // 8, max_block_height)
        block_width = random.randint(width // 8, max_block_width)

        # Randomly choose the top-left corner of the block
        top = random.randint(0, height - block_height)
        left = random.randint(0, width - block_width)

        # Apply the mask
        mask[:, top:top + block_height, left:left + block_width] = 0

        if self.mask_folder:
            mask_path = os.path.join(self.mask_folder, f'{idx}.png')
            self.save_mask(mask, mask_path)

        masked_image = image * mask
        return masked_image, mask

    def save_mask(self, mask, path):
        # Convert the mask to a PIL image and save it
        mask = mask * 255
        mask_image = ToPILImage()(mask[0].byte())  # Take the first channel and convert to PIL image
        mask_image.save(path)