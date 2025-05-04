import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from tqdm import tqdm
import torch
# import torch.nn as nn
import torch.optim as optim
import torchvision
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.base_models2 import Discriminator, Generator, initialize_weights
from datasets.base_datasets import BaseCIFAR10

# Hyperparameters etc
# device = "cuda" if torch.cuda.is_available() else "cpu"
# LEARNING_RATE = 3e-4 # 5e-5
# BATCH_SIZE = 16 # 64
# IMAGE_SIZE = 32 # 64
# CHANNELS_IMG = 3 # 1
# Z_DIM = 128
# NUM_EPOCHS = 11
# FEATURES_CRITIC = 64
# FEATURES_GEN = 64
# CRITIC_ITERATIONS = 5
# WEIGHT_CLIP = 0.01
with open("configs/base_hparams.yaml", "r") as f :
    hparams = yaml.safe_load(f)
# ddpm_hparams = hparams["ddpm"]
gan_hparams = hparams["gan"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transforms = transforms.Compose(
    [
        # transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(gan_hparams["channels_img"])], [0.5 for _ in range(gan_hparams["channels_img"])]
        ),
    ]
)

dataset = BaseCIFAR10(root="data/", train=True, transform=transforms, subset_size=10000) # added
# dataset = datasets.MNIST(root="data/", transform=transforms, download=True)
#comment mnist and uncomment below if you want to train on CelebA dataset
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=gan_hparams["batch_size"], shuffle=True)

# initialize gen and disc/critic
generator = Generator(gan_hparams["latent_dim"],
                gan_hparams["channels_img"],
                gan_hparams["features_g"]).to(DEVICE)
critic = Discriminator(gan_hparams["channels_img"],
                        gan_hparams["features_d"]).to(DEVICE)

initialize_weights(generator)
initialize_weights(critic)

# initializate optimizer
# opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
# opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
opt_gen = optim.Adam(generator.parameters(), lr=gan_hparams["learning_rate"])
opt_critic = optim.Adam(critic.parameters(), lr=gan_hparams["learning_rate"])

# for tensorboard plotting
# fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
# writer_real = SummaryWriter(f"logs/real")
# writer_fake = SummaryWriter(f"logs/fake")
writer = SummaryWriter(log_dir="logs/base_gan_v0")
# step = 0

# gen.train()
# critic.train()

# for epoch in range(NUM_EPOCHS):
#     # Target labels not needed! <3 unsupervised
#     for batch_idx, (data, _) in enumerate(tqdm(loader)):
#         data = data.to(device)
#         cur_batch_size = data.shape[0]

#         # Train Critic: max E[critic(real)] - E[critic(fake)]
#         for _ in range(CRITIC_ITERATIONS):
#             noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
#             fake = gen(noise)
#             critic_real = critic(data).reshape(-1)
#             critic_fake = critic(fake).reshape(-1)
#             loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
#             critic.zero_grad()
#             loss_critic.backward(retain_graph=True)
#             opt_critic.step()

#             # clip critic weights between -0.01, 0.01
#             for p in critic.parameters():
#                 p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

#         # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
#         gen_fake = critic(fake).reshape(-1)
#         loss_gen = -torch.mean(gen_fake)
#         gen.zero_grad()
#         loss_gen.backward()
#         opt_gen.step()

#         # Print losses occasionally and print to tensorboard
#         if batch_idx % 5 == 0 and batch_idx > 0:
#             gen.eval()
#             critic.eval()
#             print(
#                 f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
#                   Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
#             )

#             with torch.no_grad():
#                 fake = gen(noise)
#                 # take out (up to) 32 examples
#                 img_grid_real = torchvision.utils.make_grid(
#                     data[:8], normalize=True
#                 )
#                 img_grid_fake = torchvision.utils.make_grid(
#                     fake[:8], normalize=True
#                 )

#                 writer_real.add_image("Real", img_grid_real, global_step=step)
#                 writer_fake.add_image("Fake", img_grid_fake, global_step=step)

#             step += 1
#             gen.train()
#             critic.train()






def train_gan(generator, critic, dataloader, generator_optimizer, critic_optimizer, device, nb_epochs, path,
              nb_critic_itr=5, weight_clip=0.01, writer=None) :
    generator.train()
    critic.train()
    step = 0
    for epoch in range(nb_epochs):
        for batch_idx, (image, _) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            batch_size = image.shape[0]

            # Train Critic: max (E[critic(real)] - E[critic(fake)]) <-> min -(E[critic(real)] - E[critic(fake)])
            for _ in range(nb_critic_itr):
                noise = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
                fake = generator(noise)
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

            step += 1
            generator.train()
            critic.train()

        # Logging
        if epoch % 5 == 0 :
            generator.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{nb_epochs}]\
                Loss Critic: {loss_critic:.4f}, Loss Generator: {loss_generator:.4f}"
            )
            if writer :
                writer.add_scalar("Loss/Critic", loss_critic, global_step=step)
                writer.add_scalar("Loss/Generator", loss_generator, global_step=step)
                writer.add_scalar("Scores/Real", critic_real.mean().item(), global_step=step)
                writer.add_scalar("Scores/Fake", critic_fake.mean().item(), global_step=step)
                wasserstein_distance = critic_real.mean() - critic_fake.mean()
                writer.add_scalar("Distance/Wasserstein", wasserstein_distance.item(), global_step=step)

                with torch.no_grad():
                    samples = torchvision.utils.make_grid(fake[:8], normalize=True)
                    writer.add_image("GAN Samples", samples, global_step=step)



train_gan(
        generator=generator,
        critic=critic,
        dataloader=loader,
        generator_optimizer=opt_gen,
        critic_optimizer=opt_critic,
        device=DEVICE,
        nb_epochs=gan_hparams["nb_epochs"],
        path="models/base_gan.pth",
        nb_critic_itr=gan_hparams["nb_critic_itr"],
        weight_clip=gan_hparams["weight_clip"],
        writer=None
)