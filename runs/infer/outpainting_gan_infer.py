import os
import sys
import torch
import matplotlib.pyplot as plt
import shutil
import yaml
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets.outpainting_datasets import OutpaintingCIFAR10
from models.outpainting_models import OutpaintingGenerator
from utils.metrics import calculate_fid, calculate_ssim, calculate_l1_masked

# === Load config ===
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

GAN_MODEL_NAME = Path(paths["outpainting_gan_name"])
WEIGHT_DIR = Path(paths["weight_dir"])
WEIGHT_PATH = WEIGHT_DIR.joinpath(GAN_MODEL_NAME).with_suffix('.pth')
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
OUTPUT_REAL_DIR = Path(paths["output_real_dir"])
OUTPUT_GEN_DIR = Path(paths["output_gen_dir"])

with open("configs/outpainting_hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

gan_hparams = hparams["gan"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Define transforms ===
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(gan_hparams["img_channels"])],
        [0.5 for _ in range(gan_hparams["img_channels"])]
    ),
])

# === Dataset ===
dataset_val = OutpaintingCIFAR10(
    root="data/",
    train=False,
    download=False,
    transform=transforms,
    subset_size=10000,
    visible_ratio=0.75
)
dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

# === Model ===
generator = OutpaintingGenerator(
    latent_dim=gan_hparams["latent_dim"],
    img_channels=gan_hparams["img_channels"]
).to(DEVICE)


def save_images(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    for i, img in enumerate(images):
        save_image(img, os.path.join(folder, f"{i}.png"))


def test_outpainting_gan(generator, dataloader, device, weight_path, output_real_folder, output_gen_folder, num_samples=8):
    generator.load_state_dict(torch.load(weight_path, map_location=device))
    generator.eval()

    fixed_noise = torch.randn(64, generator.latent_dim, device=device)

    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_masked, test_target, test_mask = [x.to(device) for x in test_batch]

        # Generate samples
        samples = generator(fixed_noise, test_masked)
        samples = samples * (1 - test_mask) + test_masked * test_mask

        # Denormalize for visualization
        test_target = torch.clamp((test_target.cpu() + 1) / 2.0, 0, 1)
        test_masked = torch.clamp((test_masked.cpu() + 1) / 2.0, 0, 1)
        samples = torch.clamp((samples.cpu() + 1) / 2.0, 0, 1)

        # Save images
        save_images(test_target, output_real_folder)
        save_images(samples, output_gen_folder)

        # Display
        fig, axs = plt.subplots(3, num_samples, figsize=(15, 12))
        for i in range(num_samples):
            axs[0, i].imshow(np.transpose(test_target[i].numpy(), (1, 2, 0)))
            axs[0, i].axis('off')
            axs[0, i].set_title('Original')

            axs[1, i].imshow(np.transpose(test_masked[i].numpy(), (1, 2, 0)))
            axs[1, i].axis('off')
            axs[1, i].set_title('Masked')

            axs[2, i].imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
            axs[2, i].axis('off')
            axs[2, i].set_title('Outpainting')

        plt.suptitle('Outpainting with GAN')
        plt.show()


if __name__ == "__main__":
    test_outpainting_gan(
        generator=generator,
        dataloader=dataloader,
        device=DEVICE,
        weight_path=WEIGHT_PATH,
        output_real_folder=OUTPUT_REAL_DIR,
        output_gen_folder=OUTPUT_GEN_DIR,
    )

    fid = calculate_fid(OUTPUT_GEN_DIR, OUTPUT_REAL_DIR)
    avg_ssim = calculate_ssim(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR)
    avg_l1_masked = calculate_l1_masked(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR, OUTPUT_MASK_DIR)

    print(f"Average SSIM: {avg_ssim}")
    print(f"Average L1 (masked): {avg_l1_masked}")
    print(f"FID: {fid}")