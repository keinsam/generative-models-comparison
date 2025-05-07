import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from models.base_gan_models import initialize_weights
from datasets.inpainting_datasets import InpaintingCIFAR10
from models.inpainting_models import InpaintingGenerator, InpaintingCritic

# Load hyperparameters
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WGAN_MODEL_NAME = Path(paths["inpainting_gan_name"])
WEIGHT_DIR = Path(paths["weight_dir"])
WEIGHT_PATH = WEIGHT_DIR.joinpath(WGAN_MODEL_NAME).with_suffix('.pth')
LOG_DIR = Path(paths["log_dir"])
LOG_DIR = LOG_DIR.joinpath(WGAN_MODEL_NAME)
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/inpainting_hparams.yaml", "r") as f:
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
dataset = InpaintingCIFAR10(root="data/", train=True, download=False, transform=transform, subset_size=10000)  # Use inpainting dataset
dataloader = DataLoader(dataset, batch_size=gan_hparams["batch_size"], shuffle=False)

# Models
generator = InpaintingGenerator(gan_hparams["latent_dim"], gan_hparams["img_channels"]).to(DEVICE)
critic = InpaintingCritic(gan_hparams["img_channels"]).to(DEVICE)

initialize_weights(generator)
initialize_weights(critic)

# Optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=gan_hparams["generator_learning_rate"])
critic_optimizer = optim.Adam(critic.parameters(), lr=gan_hparams["critic_learning_rate"])

# Tensorboard
writer = SummaryWriter(log_dir=LOG_DIR)


def train_inpainting_gan(
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
        for batch_idx, (masked_images, real_images, masks) in enumerate(tqdm(dataloader)):
            masked_images = masked_images.to(DEVICE)
            real_images = real_images.to(DEVICE)
            masks = masks.to(DEVICE)

            batch_size = real_images.shape[0]

            # Train Critic
            critic_losses = []
            for _ in range(nb_critic_itr):
                # Generate fake images
                noise = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_images = generator(noise, masked_images)

                # Combine with known pixels
                completed_images = fake_images * masks + masked_images * (1 - masks)

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
            completed_images = fake_images * masks + masked_images * (1 - masks)
            reconstruction_loss = torch.nn.functional.l1_loss(completed_images, real_images)
            lambda_recon = gan_hparams["lambda_recon"]
            generator_loss = -torch.mean(critic(completed_images, masked_images).reshape(-1)) + lambda_recon * reconstruction_loss

            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            # Logging
            if writer is not None:
                avg_critic_loss = sum(critic_losses) / len(critic_losses)
                writer.add_scalar("InpaintingGAN/Critic_Loss", avg_critic_loss, global_step=step)
                writer.add_scalar("InpaintingGAN/Generator_Loss", generator_loss.item(), global_step=step)

                wasserstein_distance = torch.mean(critic_real) - torch.mean(critic_fake)
                writer.add_scalar("InpaintingGAN/Wasserstein_Distance", wasserstein_distance.item(), global_step=step)

                if step % 5 == 0:
                    with torch.no_grad():
                        # Visualize some examples
                        test_batch = next(iter(dataloader))
                        test_masked, test_target, test_mask = [x[:8].to(device) for x in test_batch]

                        # Generate samples
                        samples = generator(fixed_noise, test_masked)
                        completed_samples = samples * test_mask + test_masked * (1 - test_mask)

                        # Create grid of: masked images | generated samples | target images
                        grid_input = torchvision.utils.make_grid(test_masked, nrow=8, normalize=True)
                        grid_output = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
                        grid_target = torchvision.utils.make_grid(test_target, nrow=8, normalize=True)
                        grid = torch.cat([grid_input, grid_output, grid_target], dim=1)

                        writer.add_image("InpaintingWGAN/Samples", grid, global_step=step)

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
    print(DEVICE)
    train_inpainting_gan(
        generator=generator,
        critic=critic,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        device=DEVICE,
        nb_epochs=gan_hparams["nb_epochs"],
        path=WEIGHT_PATH,
        nb_critic_itr=gan_hparams["nb_critic_itr"],
        weight_clip=gan_hparams["weight_clip"],
        writer=writer
    )