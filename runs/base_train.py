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

def train_ddpm(model, diffusion, dataloader, optimizer, device, epochs, path, writer=None):
    model.train()
    global_step = 0
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix({"loss": loss.item()})
    
        if writer is not None:
            if epoch % 5 == 0:
                # Save checkpoint
                # checkpoint_path = path.parent / f"{path.stem}_epoch{epoch}.pth"
                # torch.save(model.state_dict(), checkpoint_path)
                # print(f"Checkpoint saved to {checkpoint_path}")
                with torch.no_grad():
                    samples = diffusion.sample(model, n=8)
                    writer.add_images("Samples", (samples + 1) / 2.0, epoch)
    
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
    dataset = BaseCIFAR10(root="data/", train=True, transform=transform, subset_size=10000)
    dataloader = DataLoader(dataset, batch_size=ddpm_hparams["batch_size"], shuffle=True)

    # DDPM model
    ddpm = DDPM(in_channels=ddpm_hparams["in_channels"],
                 out_channels=ddpm_hparams["out_channels"],
                 time_dim=ddpm_hparams["time_dim"],
                 ).to(DEVICE)
    diffusion = Diffusion(noise_steps=ddpm_hparams["noise_steps"],
                          beta_start=ddpm_hparams["beta_start"],
                          beta_end=ddpm_hparams["beta_end"],
                          img_size=ddpm_hparams["img_size"],
                          device=DEVICE)
    
    # Optimizer
    # optimizer = optim.RMSprop(ddpm.parameters(),
    #                           lr=ddpm_hparams["learning_rate"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ddpm_hparams["nb_epochs"])
    optimizer = optim.Adam(ddpm.parameters(),
                           lr=ddpm_hparams["learning_rate"])
    
    # Tensorboard
    writer = SummaryWriter(log_dir=LOG_DIR.joinpath(DDPM_MODEL_NAME))
    writer.add_hparams(ddpm_hparams, {})

    train_ddpm(ddpm, diffusion, dataloader, optimizer,
               device = DEVICE,
               epochs=ddpm_hparams["nb_epochs"],
               path=DDPM_MODEL_PATH,
               writer=writer)
    generate_and_visualize_samples(ddpm, diffusion, DEVICE)