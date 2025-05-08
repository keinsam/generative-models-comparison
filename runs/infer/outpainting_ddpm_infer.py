import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import shutil
import yaml
import torch
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.outpainting_models import OutpaintingUNetEpsilon, OutpaintingDDPM
from utils.metrics import calculate_fid, calculate_ssim, calculate_l1_masked
from datasets.outpainting_datasets import OutpaintingCIFAR10

# Chargement des chemins
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
DDPM_MODEL_NAME = Path(paths["outpainting_ddpm_name"])
WEIGHT_DIR = Path(paths["weight_dir"])
WEIGHT_PATH = WEIGHT_DIR.joinpath(DDPM_MODEL_NAME).with_suffix('.pth')
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
OUTPUT_REAL_DIR = Path(paths["output_real_dir"])
OUTPUT_GEN_DIR = Path(paths["output_gen_dir"])

# Chargement des hyperparamètres
with open("configs/outpainting_hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)
ddpm_hparams = hparams["ddpm"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformations
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * ddpm_hparams["img_channels"], [0.5] * ddpm_hparams["img_channels"])
])

# Dataset et DataLoader
dataset_val = OutpaintingCIFAR10(
    root="data/", train=False, download=False,
    transform=transforms, subset_size=10000, visible_ratio=0.75,
)
dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

# Modèle
eps_model = OutpaintingUNetEpsilon(
    n_channel=ddpm_hparams["img_channels"],
    time_dim=ddpm_hparams["time_dim"]
).to(DEVICE)

model = OutpaintingDDPM(
    eps_model,
    betas=(ddpm_hparams["beta_start"], ddpm_hparams["beta_end"]),
    n_T=ddpm_hparams["noise_steps"],
    criterion=torch.nn.MSELoss()
).to(DEVICE)

def save_images(images, folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    for i, img in enumerate(images):
        save_image(img, os.path.join(folder, f"{i}.png"))

def test_outpainting_ddpm(model, dataloader, device, weight_path, output_real_folder, output_gen_folder, num_samples=8):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_masked, test_target, test_mask = [x.to(device) for x in test_batch]

        samples = model.sample(test_masked, test_mask)

        test_target = torch.clamp((test_target.cpu() + 1) / 2.0, 0, 1)
        test_masked = torch.clamp((test_masked.cpu() + 1) / 2.0, 0, 1)
        samples = torch.clamp((samples.cpu() + 1) / 2.0, 0, 1)

        save_images(test_target, output_real_folder)
        save_images(samples, output_gen_folder)

        fig, axs = plt.subplots(3, num_samples, figsize=(15, 12))
        for i in range(num_samples):
            axs[1, i].imshow(np.transpose(test_masked[i].numpy(), (1, 2, 0)))
            axs[1, i].axis('off')
            axs[1, i].set_title('Masked')

            axs[2, i].imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
            axs[2, i].axis('off')
            axs[2, i].set_title('Generated')

            axs[0, i].imshow(np.transpose(test_target[i].numpy(), (1, 2, 0)))
            axs[0, i].axis('off')
            axs[0, i].set_title('Target')

        plt.suptitle('Outpainting with DDPM')
        plt.show()


if __name__ == "__main__":
    test_outpainting_ddpm(
        model=model,
        dataloader=dataloader,
        device=DEVICE,
        weight_path=WEIGHT_PATH,
        output_real_folder=OUTPUT_REAL_DIR,
        output_gen_folder=OUTPUT_GEN_DIR,
    )

    fid = calculate_fid(OUTPUT_GEN_DIR, OUTPUT_REAL_DIR)
    avg_ssim = calculate_ssim(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR)
    avg_l1 = calculate_l1_masked(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR, OUTPUT_MASK_DIR)

    print(f"Average SSIM: {avg_ssim}")
    print(f"Average L1 (outpainted): {avg_l1}")
    print(f"FID: {fid}")
