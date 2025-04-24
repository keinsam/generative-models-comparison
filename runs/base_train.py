import yaml
import sys
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.base_datasets import BaseCIFAR10
from models.base_models import DDPM, Diffusion

with open("configs/paths.yaml", "r") as f :
    paths = yaml.safe_load(f)
DDPM_MODEL_NAME = Path(paths["base_ddpm_name"])
GAN_MODEL_NAME = Path(paths["base_gan_name"])
LOG_DIR = Path(paths["log_dir"])
WEIGHTS_DIR = Path(paths["weight_dir"])
DDPM_MODEL_PATH = WEIGHTS_DIR.joinpath(f"{DDPM_MODEL_NAME}.pth")
GAN_MODEL_PATH = WEIGHTS_DIR.joinpath(f"{GAN_MODEL_NAME}.pth")
LOG_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/base_hparams.yaml", "r") as f :
    hparams = yaml.safe_load(f)
ddpm_hparams = hparams["ddpm"]
# wgan_hparams = hparams["gan"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_ddpm(model, diffusion, dataloader, optimizer, device, epochs, path):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:    
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model.forward(x_t, t)
            loss = nn.MSELoss()(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def generate_and_visualize_samples(model, diffusion, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        # Generate samples
        samples = diffusion.sample(model, n=num_samples)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()

        # Plot the samples
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i])
            ax.axis("off")
        plt.show()

if __name__ == "__main__" :
    # Dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = BaseCIFAR10(root="data/", train=True, transform=transform, subset_size=5000)
    dataloader = DataLoader(dataset, batch_size=ddpm_hparams["train"]["batch_size"], shuffle=True)

    # DDPM model
    ddpm = DDPM(in_channels=ddpm_hparams["model"]["in_channels"],
                 out_channels=ddpm_hparams["model"]["out_channels"],
                 time_dim=ddpm_hparams["model"]["time_dim"],
                 ).to(DEVICE)
    diffusion = Diffusion(noise_steps=ddpm_hparams["model"]["noise_steps"],
                          beta_start=ddpm_hparams["model"]["beta_start"],
                          beta_end=ddpm_hparams["model"]["beta_end"],
                          img_size=ddpm_hparams["model"]["img_size"],
                          device=DEVICE)
    
    # Optimizer
    optimizer = optim.RMSprop(ddpm.parameters(),
                              lr=ddpm_hparams["train"]["learning_rate"],
                              alpha=0.99,
                              momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ddpm_hparams["train"]["nb_epochs"])

    train_ddpm(ddpm, diffusion, dataloader, optimizer,
               device = DEVICE,
               epochs=ddpm_hparams["train"]["nb_epochs"],
               path=DDPM_MODEL_PATH)
    generate_and_visualize_samples(ddpm, diffusion, DEVICE)