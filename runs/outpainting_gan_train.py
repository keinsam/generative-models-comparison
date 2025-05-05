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
dataloader = DataLoader(dataset, batch_size=gan_hparams["batch_size"], shuffle=True)

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

def create_mask(batch_size, channels, height, width, mask_ratio=0.25, device="cuda"):
    """Create a mask that keeps left 25% of the image"""
    mask = torch.zeros((batch_size, channels, height, width), device=device)
    mask_width = int(width * mask_ratio)
    mask[:, :, :, :mask_width] = 1
    return mask

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
        for batch_idx, (real_images, _) in enumerate(tqdm(dataloader)):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            
            # Create masks (keep left 25%)
            masks = create_mask(
                batch_size, 
                gan_hparams["img_channels"], 
                real_images.shape[2], 
                real_images.shape[3], 
                device=device
            )
            masked_images = real_images * masks
            
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
            
            generator_loss = -torch.mean(critic(completed_images, masked_images).reshape(-1))
            
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
                
                # Visualize samples periodically
                if step % 5 == 0:
                    with torch.no_grad():
                        # Generate samples
                        fixed_masked = real_images[:8] * masks[:8]
                        samples = generator(fixed_noise, fixed_masked)
                        completed_samples = samples * (1 - masks[:8]) + fixed_masked * masks[:8]
                        
                        # Create comparison grid - simplified approach
                        grid_input = fixed_masked  # Already has 3 channels
                        grid_output = completed_samples
                        grid_real = real_images[:8]
                        
                        # Stack images vertically: masked input | generated output | real image
                        grid = torch.cat([grid_input, grid_output, grid_real], dim=0)
                        
                        # Normalize to [0, 1] range
                        grid = (grid + 1) / 2  # Assuming tanh output (-1 to 1)
                        
                        # Make grid with 3 rows (input, output, real) and 8 columns
                        grid = torchvision.utils.make_grid(grid, nrow=8, normalize=False)
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