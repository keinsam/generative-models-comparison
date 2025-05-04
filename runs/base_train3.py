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
from models.base_models3 import DummyEpsModel, DDPM
from datasets.base_datasets import BaseCIFAR10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ddpm = DDPM(eps_model=DummyEpsModel(3), betas=(1e-4, 0.02), n_T=200)
ddpm.to(DEVICE)

tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
dataset = BaseCIFAR10(root="data/", train=True, transform=tf, subset_size=1000) # added
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #, num_workers=20)
optim = torch.optim.Adam(ddpm.parameters(), lr=3e-4)

def train_ddpm(model, dataloader, optimizer, device, nb_epoch, path, writer=None):
    model.train()
    loss_ema = None
    for epoch in range(nb_epoch):
        for batch_idx, (image, _) in enumerate(tqdm(dataloader)) :
            image = image.to(device)
            optimizer.zero_grad()
            loss = model(image)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            optimizer.step()

        model.eval()
        # with torch.no_grad():
        #     # xh = ddpm.sample(16, (1, 28, 28), device)
        #     xh = ddpm.sample(16, (3, 32, 32), device)
        #     grid = make_grid(xh, nrow=4, normalize=True)
        #     save_image(grid, f"./bin/ddpm_sample_{i}.png")

        #     # save model
        #     torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")


if __name__ == "__main__":
    train_ddpm(n_epoch=10, device="cpu")