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
from models.upsampling_models import Discriminator, Generator
from datasets.upsampling_datasets import UpsamplingCIFAR10

import torch
import torchvision
from tqdm import tqdm


with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
WGAN_MODEL_NAME = Path(paths["upsampling_wgan_name"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(WGAN_MODEL_NAME).with_suffix('.pth')
LOG_DIR = Path(paths["log_dir"])
LOG_DIR = LOG_DIR.joinpath(WGAN_MODEL_NAME)
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/upsampling_hparams.yaml", "r") as f :
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

generator = Generator(scale_factor=4).to(DEVICE)  # 32x32 → 128x128
discriminator = Discriminator().to(DEVICE)

dataset = UpsamplingCIFAR10(root="data/", train=True, transform=transforms, subset_size=10000) # added
loader = DataLoader(dataset, batch_size=wgan_hparams["batch_size"], shuffle=True)



# initializate optimizer
opt_gen = optim.Adam(generator.parameters(), lr=wgan_hparams["learning_rate"])
opt_critic = optim.Adam(discriminator.parameters(), lr=wgan_hparams["learning_rate"])


writer = SummaryWriter(log_dir=LOG_DIR)


def train_wgan(generator, discriminator, dataloader, gen_optimizer, disc_optimizer,
               device, nb_epochs, path, nb_critic_iter=5, weight_clip=0.01, writer=None):
    """
    Entraîne un WGAN pour super-résolution d'images CIFAR-10.
    """
    generator.train()
    discriminator.train()
    step = 0

    for epoch in range(nb_epochs):
        for batch_idx, (low_res_image, high_res_image) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{nb_epochs}")):
            low_res_image = low_res_image.to(device)
            high_res_image = high_res_image.to(device)
            batch_size = low_res_image.shape[0]

            # Entraîner le Discriminateur: max (E[D(real)] - E[D(fake)]) <-> min -(E[D(real)] - E[D(fake)])
            for _ in range(nb_critic_iter):
                with torch.no_grad():
                    fake_high_res = generator(low_res_image)

                disc_real = discriminator(high_res_image).reshape(-1)
                disc_fake = discriminator(fake_high_res.detach()).reshape(-1)

                # Perte du discriminateur selon WGAN
                loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))

                discriminator.zero_grad()
                loss_disc.backward()
                disc_optimizer.step()

                # Clipper les poids du discriminateur entre -weight_clip et weight_clip
                for p in discriminator.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            # Entraîner le Générateur: max E[D(G(z))] <-> min -E[D(G(z))]
            fake_high_res = generator(low_res_image)
            outputs = discriminator(fake_high_res).reshape(-1)

            # Perte adversariale du générateur
            loss_gen_adv = -torch.mean(outputs)

            # Perte de contenu - souvent utile pour la super-résolution
            # Elle aide à maintenir le contenu de l'image d'origine
            content_loss = torch.nn.functional.mse_loss(fake_high_res, high_res_image)

            # Combiner les pertes (le facteur 100 donne plus d'importance à la reconstruction)
            loss_gen = loss_gen_adv + 100 * content_loss

            generator.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()

            step += 1


        if epoch % 5 == 0:
            generator.eval()
            discriminator.eval()
            print(
                f"Epoch [{epoch}/{nb_epochs}]\
                Loss Critic: {loss_disc:.4f}, Loss Generator: {loss_gen:.4f}"
            )

            if writer:
                # Journaliser les pertes
                writer.add_scalar("Loss/Discriminator", loss_disc.item(), global_step=step)
                writer.add_scalar("Loss/Generator/Total", loss_gen.item(), global_step=step)

                # Journaliser les scores du discriminateur
                writer.add_scalar("Scores/Real", disc_real.mean().item(), global_step=step)
                writer.add_scalar("Scores/Fake", disc_fake.mean().item(), global_step=step)

                # Distance Wasserstein (métrique principale pour WGAN)
                wasserstein_distance = disc_real.mean() - disc_fake.mean()
                writer.add_scalar("Distance/Wasserstein", wasserstein_distance.item(), global_step=step)

                # Journaliser les images
                with torch.no_grad():
                    # Images basse résolution d'origine (CIFAR-10)
                    low_res_samples = torchvision.utils.make_grid(low_res_image[:8], normalize=True)
                    writer.add_image("Original Low-Res (32x32)", low_res_samples, global_step=step)

                    # Images haute résolution attendues (bicubic upscaled)
                    high_res_samples = torchvision.utils.make_grid(high_res_image[:8], normalize=True)
                    writer.add_image("Expected High-Res (Bicubic)", high_res_samples, global_step=step)

                    # Images haute résolution générées par notre modèle
                    generated_samples = torchvision.utils.make_grid(fake_high_res[:8], normalize=True)
                    writer.add_image("Generated High-Res (Model)", generated_samples, global_step=step)

        torch.cuda.empty_cache()

    torch.save(generator, path)


# Initialiser les poids
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

train_wgan(
    generator=generator,
    discriminator=discriminator,
    dataloader=loader,
    gen_optimizer=opt_gen,
    disc_optimizer=opt_critic,
    device=DEVICE,
    nb_epochs=wgan_hparams["nb_epochs"],
    path=MODEL_PATH,
    nb_critic_iter=wgan_hparams["nb_critic_itr"],
    weight_clip=wgan_hparams["weight_clip"],
    writer=writer
)





