import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from models.base_gan_models import initialize_weights
from datasets.outpainting_datasets import OutpaintingCIFAR10
from models.outpainting_models import OutpaintingGenerator, OutpaintingCritic

# Load hyperparameters
with open("configs/outpainting_hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)
gan_hparams = hparams["gan"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(gan_hparams["img_channels"])], 
        [0.5 for _ in range(gan_hparams["img_channels"])]
    ),
])

# Dataset and dataloader
dataset = OutpaintingCIFAR10(root="data/", train=True, download=False, transform=transform, subset_size=10000, visible_ratio=0.75)
dataloader = DataLoader(dataset, batch_size=gan_hparams["batch_size"], shuffle=False)

# Models
generator = OutpaintingGenerator(gan_hparams["latent_dim"], gan_hparams["img_channels"]).to(DEVICE)
critic = OutpaintingCritic(gan_hparams["img_channels"]).to(DEVICE)

initialize_weights(generator)
initialize_weights(critic)

# Optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=gan_hparams["learning_rate"])
critic_optimizer = optim.Adam(critic.parameters(), lr=gan_hparams["learning_rate"])

# Tensorboard
writer = SummaryWriter(log_dir="./logs/outpainting_gan_v1")
writer.add_hparams(gan_hparams, {})


fixed_real_images, fixed_masked_images, fixed_masks = next(iter(dataloader))
fixed_masks = fixed_masks[:8].to(DEVICE)
fixed_real_images = fixed_real_images[:8].to(DEVICE)
fixed_masked_images = fixed_masked_images[:8].to(DEVICE)


def train_outpainting_gan(
    generator, 
    critic, 
    dataloader, 
    generator_optimizer, 
    critic_optimizer, 
    device, 
    nb_epochs, 
    path,
    nb_critic_itr=5,
    weight_clip=0.01,
    writer=None
):
    generator.train()
    critic.train()
    step = 0
    
    # Fixed noise for consistent samples
    fixed_noise = torch.randn(8, generator.latent_dim, device=device)
    
    for epoch in range(nb_epochs):
        for batch_idx, (real_images, masked_images, masks) in enumerate(tqdm(dataloader)):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            
            # Train Critic
            critic_losses = []
            for _ in range(nb_critic_itr):
                # Generate fake images
                noise = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_images = generator(noise, masked_images)
                
                # Combine with known pixels
                completed_images = fake_images * (1 - masks) + masked_images * masks
                
                # Critic loss
                critic_real = critic(real_images, masked_images).reshape(-1)
                critic_fake = critic(completed_images.detach(), masked_images).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                
                # Optimize critic
                critic.zero_grad()
                loss_critic.backward()
                critic_optimizer.step()
                
                # Clip critic weights
                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)
                
                critic_losses.append(loss_critic.item())
            
            # Train Generator
            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_images = generator(noise, masked_images)
            completed_images = fake_images * (1 - masks) + masked_images * masks
            # generator_loss = -torch.mean(critic(completed_images, masked_images).reshape(-1))
            reconstruction_loss = torch.nn.functional.l1_loss(completed_images, real_images)
            lambda_recon = gan_hparams["lambda_recon"]
            generator_loss = -torch.mean(critic(completed_images, masked_images).reshape(-1)) + lambda_recon * reconstruction_loss
            
            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()
            
            # Logging
            if writer is not None:
                avg_critic_loss = sum(critic_losses) / len(critic_losses)
                writer.add_scalar("OutpaintingGAN/Critic_Loss", avg_critic_loss, global_step=step)
                writer.add_scalar("OutpaintingGAN/Generator_Loss", generator_loss.item(), global_step=step)
                
                wasserstein_distance = torch.mean(critic_real) - torch.mean(critic_fake)
                writer.add_scalar("OutpaintingGAN/Wasserstein_Distance", wasserstein_distance.item(), global_step=step)
                
                if step % 5 == 0:
                    with torch.no_grad():
                        samples = generator(fixed_noise, fixed_masked_images)
                        completed_samples = samples * (1 - fixed_masks) + fixed_masked_images * fixed_masks

                        grid_input = torchvision.utils.make_grid(fixed_masked_images, nrow=8, normalize=True)
                        grid_output = torchvision.utils.make_grid(completed_samples, nrow=8, normalize=True)
                        grid_real = torchvision.utils.make_grid(fixed_real_images, nrow=8, normalize=True)
                        grid = torch.cat([grid_input, grid_output, grid_real], dim=1)

                        writer.add_image("OutpaintingGAN/Samples", grid, global_step=step)
            
            step += 1
        
        # Epoch logging
        print(
            f"Epoch [{epoch+1}/{nb_epochs}] "
            f"Critic Loss: {avg_critic_loss:.4f} "
            f"Generator Loss: {generator_loss.item():.4f}"
        )
    
    # Save final model
    torch.save(generator.state_dict(), path)

if __name__ == "__main__":
    train_outpainting_gan(
        generator=generator,
        critic=critic,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        device=DEVICE,
        nb_epochs=gan_hparams["nb_epochs"],
        path="weights/outpainting_gan_v1.pth",
        nb_critic_itr=gan_hparams["nb_critic_itr"],
        weight_clip=gan_hparams["weight_clip"],
        writer=writer
    )