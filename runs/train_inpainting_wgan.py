import os
import sys

import torchvision
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from pathlib import Path
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.inpainting_models import Discriminator, Generator, initialize_weights
from datasets.inpainting_datasets import InpaintingCIFAR10

with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
WGAN_MODEL_NAME = Path(paths["inpainting_wgan_name"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(WGAN_MODEL_NAME).with_suffix('.pth')
LOG_DIR = Path(paths["log_dir"])
LOG_DIR = LOG_DIR.joinpath(WGAN_MODEL_NAME)
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/inpainting_hparams.yaml", "r") as f :
    hparams = yaml.safe_load(f)
wgan_hparams = hparams["wgan"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transforms = transforms.Compose(
    [
        # transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(wgan_hparams["channels_img"])], [0.5 for _ in range(wgan_hparams["channels_img"])]
        ),
    ]
)

dataset = InpaintingCIFAR10(root="data/", train=True, transform=transforms, subset_size=10000) # added
loader = DataLoader(dataset, batch_size=wgan_hparams["batch_size"], shuffle=True)

# initialize gen and disc/critic
generator = Generator(wgan_hparams["latent_dim"],
                wgan_hparams["channels_img"],
                wgan_hparams["features_g"]).to(DEVICE)
critic = Discriminator(wgan_hparams["channels_img"],
                        wgan_hparams["features_d"]).to(DEVICE)

initialize_weights(generator)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(generator.parameters(), lr=wgan_hparams["learning_rate"])
opt_critic = optim.Adam(critic.parameters(), lr=wgan_hparams["learning_rate"])


writer = SummaryWriter(log_dir=LOG_DIR)


def train_wgan(generator, critic, dataloader, generator_optimizer, critic_optimizer, device, nb_epochs, path,
              nb_critic_itr=5, weight_clip=0.01, writer=None):
    generator.train()
    critic.train()
    step = 0
    for epoch in range(nb_epochs):
        for batch_idx, (masked_image, original_image, mask) in enumerate(tqdm(dataloader)):
            masked_image = masked_image.to(device)
            original_image = original_image.to(device)
            mask = mask.to(device)  # Assurez-vous que le masque est également transféré sur le bon appareil
            batch_size = masked_image.shape[0]

            # Train Critic: max (E[critic(real)] - E[critic(fake)]) <-> min -(E[critic(real)] - E[critic(fake)])
            for _ in range(nb_critic_itr):
                noise = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
                fake = generator(noise, masked_image)

                # Combiner l'image générée avec l'image masquée d'origine
                inpainted_image = masked_image + (1 - mask) * fake

                critic_real = critic(original_image).reshape(-1)
                critic_fake = critic(inpainted_image.detach()).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                critic_optimizer.step()
                # Clip Critic weights between -0.01 and 0.01
                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            # Train Generator: max E[critic(generator_fake)] <-> min -E[critic(generator_fake)]
            generator_fake = critic(inpainted_image).reshape(-1)
            loss_generator = -torch.mean(generator_fake)

            # Ajouter une perte de reconstruction L1 masquée
            loss_reconstruction = torch.mean(torch.abs(fake - original_image) * (1 - mask))
            loss_generator += loss_reconstruction

            generator.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            step += 1
            generator.train()
            critic.train()

        # Logging
        if epoch % 5 == 0:
            generator.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{nb_epochs}]\
                Loss Critic: {loss_critic:.4f}, Loss Generator: {loss_generator:.4f}"
            )
            if writer:
                writer.add_scalar("Loss/Critic", loss_critic, global_step=step)
                writer.add_scalar("Loss/Generator", loss_generator, global_step=step)
                writer.add_scalar("Scores/Real", critic_real.mean().item(), global_step=step)
                writer.add_scalar("Scores/Fake", critic_fake.mean().item(), global_step=step)
                wasserstein_distance = critic_real.mean() - critic_fake.mean()
                writer.add_scalar("Distance/Wasserstein", wasserstein_distance.item(), global_step=step)

                with torch.no_grad():
                    # Ajouter des images réelles, masquées, générées et des masques à TensorBoard
                    real_samples = torchvision.utils.make_grid(original_image[:8], normalize=True)
                    masked_samples = torchvision.utils.make_grid(masked_image[:8], normalize=True)
                    inpainted_samples = torchvision.utils.make_grid(inpainted_image[:8], normalize=True)
                    mask_samples = torchvision.utils.make_grid((mask[:8] * 255).byte(), normalize=False)
                    writer.add_image("Real Samples", real_samples, global_step=step)
                    writer.add_image("Masked Samples", masked_samples, global_step=step)
                    writer.add_image("Inpainted Samples", inpainted_samples, global_step=step)
                    writer.add_image("Mask Samples", mask_samples, global_step=step)

        torch.cuda.empty_cache()

    torch.save(generator, path)


# Exemple d'utilisation
train_wgan(
    generator=generator,
    critic=critic,
    dataloader=loader,
    generator_optimizer=opt_gen,
    critic_optimizer=opt_critic,
    device=DEVICE,
    nb_epochs=wgan_hparams["nb_epochs"],
    path=MODEL_PATH,
    nb_critic_itr=wgan_hparams["nb_critic_itr"],
    weight_clip=wgan_hparams["weight_clip"],
    writer=writer
)
