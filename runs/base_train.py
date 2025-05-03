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
from models.base_models import DDPM, Diffusion, GAN_Generator, GAN_Critic

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
gan_hparams = hparams["gan"]

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

            if writer is not None :
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
                    writer.add_images("DDPM Samples", (samples + 1) / 2.0, epoch)
    
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")

def train_wgan(gen, critic, dataloader, opt_gen, opt_critic, device, epochs, path, 
              n_critic=5, clip_value=0.01, writer=None):
    gen.train()
    critic.train()
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for real_images, _ in pbar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # ===== Entraînement du Critic =====
            critic_loss_total = 0
            for _ in range(n_critic):
                # Bruit latent
                z = torch.randn(batch_size, gan_hparams["latent_dim"], device=device)
                
                # Génération d'images
                fake_images = gen(z).detach()
                
                # Calcul des scores
                real_scores = critic(real_images)
                fake_scores = critic(fake_images)
                
                # Loss du critic
                critic_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
                
                # Mise à jour
                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()
                
                # Clip des poids
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                
                critic_loss_total += critic_loss.item()
            
            # ===== Entraînement du Générateur =====
            z = torch.randn(batch_size, gan_hparams["latent_dim"], device=device)
            fake_images = gen(z)
            gen_loss = -torch.mean(critic(fake_images))
            
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            # ===== Logging =====
            avg_critic_loss = critic_loss_total / n_critic
            if writer:
                writer.add_scalar("Loss/critic", avg_critic_loss, global_step)
                writer.add_scalar("Loss/generator", gen_loss.item(), global_step)
                writer.add_scalar("Scores/real", real_scores.mean().item(), global_step)
                writer.add_scalar("Scores/fake", fake_scores.mean().item(), global_step)
            
            global_step += 1
            pbar.set_postfix({
                "C_loss": avg_critic_loss,
                "G_loss": gen_loss.item()
            })

        # ===== Visualisation =====
        if writer is not None:
            if epoch % 5 == 0:
                with torch.no_grad():
                    z = torch.randn(8, gan_hparams["latent_dim"], device=device)
                    samples = gen(z)
                    samples = (samples.clamp(-1, 1) + 1) / 2  # [0,1]
                    writer.add_images("GAN_samples", samples, epoch)
    
    # ===== Sauvegarde =====
    torch.save({
        'gen': gen.state_dict(),
        'critic': critic.state_dict()
    }, path)
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

def generate_and_visualize_samples(generator, device, num_samples=8):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.latent_dim, device=device)

        samples = generator(z).cpu()
        samples = (samples + 1) / 2

        samples = samples.permute(0, 2, 3, 1).numpy()

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
    ddpm_dataloader = DataLoader(dataset, batch_size=ddpm_hparams["batch_size"], shuffle=True)
    gan_dataloader = DataLoader(dataset, batch_size=gan_hparams["batch_size"], shuffle=True)


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
    # DDPM optimizer
    ddpm_optimizer = optim.Adam(ddpm.parameters(),
                           lr=ddpm_hparams["learning_rate"])
    
    # GAN model
    generator = GAN_Generator(
        latent_dim=gan_hparams["latent_dim"],
        out_channels=gan_hparams["out_channels"]
    ).to(DEVICE)
    critic = GAN_Critic(
        in_channels=gan_hparams["in_channels"],
    ).to(DEVICE)
    # GAN optimizer
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=gan_hparams["learning_rate"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=gan_hparams["learning_rate"])
    
    # Tensorboard
    ddpm_writer = SummaryWriter(log_dir=LOG_DIR.joinpath(DDPM_MODEL_NAME))
    ddpm_writer.add_hparams(ddpm_hparams, {})
    gan_writer = SummaryWriter(log_dir=LOG_DIR.joinpath(GAN_MODEL_NAME))
    gan_writer.add_hparams(gan_hparams, {})

    # Train DDPM
    # train_ddpm(ddpm, diffusion, ddpm_dataloader, ddpm_optimizer,
    #            device = DEVICE,
    #            epochs=ddpm_hparams["nb_epochs"],
    #            path=DDPM_MODEL_PATH,
    #            writer=ddpm_writer)
    # generate_and_visualize_samples(ddpm, diffusion, DEVICE)

    # Train GAN
    train_wgan(generator, critic, gan_dataloader, generator_optimizer, critic_optimizer,
        device=DEVICE,
        epochs=gan_hparams["nb_epochs"],
        path=GAN_MODEL_PATH,
        writer=gan_writer
    )
    # generate_and_visualize_samples(generator, DEVICE)