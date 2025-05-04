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
from models.base_gan_models import Discriminator, Generator, initialize_weights
from datasets.base_datasets import BaseCIFAR10


with open("configs/base_hparams.yaml", "r") as f :
    hparams = yaml.safe_load(f)
# ddpm_hparams = hparams["ddpm"]
gan_hparams = hparams["gan"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(gan_hparams["channels_img"])], [0.5 for _ in range(gan_hparams["channels_img"])]
        ),
    ]
)

dataset = BaseCIFAR10(root="data/", train=True, transform=transforms, subset_size=5000)
dataloader = DataLoader(dataset, batch_size=gan_hparams["batch_size"], shuffle=True)


generator = Generator(gan_hparams["latent_dim"],
                gan_hparams["channels_img"],
                gan_hparams["features_g"]).to(DEVICE)
critic = Discriminator(gan_hparams["channels_img"],
                        gan_hparams["features_d"]).to(DEVICE)

initialize_weights(generator)
initialize_weights(critic)

generator_optimizer = optim.Adam(generator.parameters(), lr=gan_hparams["learning_rate"])
critic_optimizer = optim.Adam(critic.parameters(), lr=gan_hparams["learning_rate"])

gan_writer = SummaryWriter(log_dir="./logs/base_gan_v0")
gan_writer.add_hparams(gan_hparams, {})





def train_gan(generator, critic, dataloader, generator_optimizer, critic_optimizer, device, nb_epochs, path,
              nb_critic_itr=5, weight_clip=0.01, writer=None) :
    generator.train()
    critic.train()
    step = 0
    fixed_noise = torch.randn(8, generator.latent_dim, 1, 1).to(device)
    for epoch in range(nb_epochs):
        for batch_idx, (image, _) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            batch_size = image.shape[0]

            # Train Critic: max (E[critic(real)] - E[critic(fake)]) <-> min -(E[critic(real)] - E[critic(fake)])
            for _ in range(nb_critic_itr):
                noise = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
                fake = generator(noise) # with torch.no_grad() may be better
                critic_real = critic(image).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                # critic_optimizer.zero_grad() # may be better
                loss_critic.backward(retain_graph=True) # retain_graph=True may be useless
                critic_optimizer.step()
                # Clip Critic weights between -0.01 and 0.01
                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            # Train Generator: max E[critic(generator_fake)] <-> min -E[critic(generator_fake)]
            generator_fake = critic(fake).reshape(-1)
            loss_generator = -torch.mean(generator_fake)
            generator.zero_grad()
            # generator_optimizer.zero_grad() # may be better
            loss_generator.backward()
            generator_optimizer.step()

            # Logging
            if writer is not None :
                # generator.eval()
                # critic.eval()
                writer.add_scalar("BaseGAN/Critic_Loss", loss_critic.item(), global_step=step)
                writer.add_scalar("BaseGAN/Generator_Loss", loss_generator.item(), global_step=step)
                # writer.add_scalar("BaseGAN/Real_Score", critic_real.mean().item(), global_step=step)
                # writer.add_scalar("BaseGAN/Fake_Score", critic_fake.mean().item(), global_step=step)
                wasserstein_distance = critic_real.mean() - critic_fake.mean()
                writer.add_scalar("BaseGAN/Wasserstein_Distance", wasserstein_distance.item(), global_step=step)


            step += 1
            generator.train()
            critic.train()

        # Logging
        print(
            f"Epoch [{epoch}/{nb_epochs}]\
            Loss Critic: {loss_critic:.4f}, Loss Generator: {loss_generator:.4f}"
        )
        if writer is not None and epoch % 5 == 0 :
            with torch.no_grad():
                samples = generator(fixed_noise)
                grid = torchvision.utils.make_grid(samples, normalize=True)
                writer.add_image("BaseGAN/Samples", grid, global_step=step)

    # Save model
    torch.save(generator.state_dict(), path)


if __name__ == "__main__" :

    train_gan(
            generator=generator,
            critic=critic,
            dataloader=dataloader,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            device=DEVICE,
            nb_epochs=gan_hparams["nb_epochs"],
            path="weights/base_gan_v0.pth",
            nb_critic_itr=gan_hparams["nb_critic_itr"],
            weight_clip=gan_hparams["weight_clip"],
            writer=gan_writer
    )