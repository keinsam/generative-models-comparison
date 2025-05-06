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
from models.base_ddpm_models import UNetEpsilon, DDPM
from datasets.base_datasets import BaseCIFAR10


with open("configs/base_hparams.yaml", "r") as f :
    hparams = yaml.safe_load(f)
ddpm_hparams = hparams["ddpm"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(ddpm_hparams["img_channels"])], [0.5 for _ in range(ddpm_hparams["img_channels"])]
        ),
    ]
)
# dataset = BaseCIFAR10(root="data/", train=True, download=False, transform=transforms, subset_size=10000)
dataset = torchvision.datasets.MNIST(root="data/", train=True, transform=transforms, download=False)
# dataset = torch.utils.data.Subset(dataset, range(1000))
dataloader = DataLoader(dataset, batch_size=ddpm_hparams["batch_size"], shuffle=True)


# ddpm = DDPM(eps_model=DummyEpsModel(ddpm_hparams["img_channels"], ddpm_hparams["time_dim"]), 
#             betas=(ddpm_hparams["beta_start"], ddpm_hparams["beta_end"]), n_T=ddpm_hparams["noise_steps"]).to(DEVICE)
ddpm = DDPM(eps_model=UNetEpsilon(ddpm_hparams["img_channels"], ddpm_hparams["time_dim"]), 
            betas=(ddpm_hparams["beta_start"], ddpm_hparams["beta_end"]), n_T=ddpm_hparams["noise_steps"]).to(DEVICE)

optimizer = optim.Adam(ddpm.parameters(), lr=ddpm_hparams["learning_rate"])

ddpm_writer = SummaryWriter(log_dir="./logs/base_ddpm_v8")
ddpm_writer.add_hparams(ddpm_hparams, {})





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
            
            if writer is not None:
                writer.add_scalar("BaseDDPM/Loss", loss_ema, epoch)

            optimizer.step()

        print(f"Epoch [{epoch}/{nb_epoch}]\
               Loss: {loss_ema:.4f}"
        )
        if writer is not None and epoch % 5 == 0 : # comment epoch % 5 == 0 for testing
            # model.eval()
            with torch.no_grad() :
                # samples = model.sample(2, (3, 32, 32), device)
                samples = model.sample(2, (1, 32, 32), device)
                grid = torchvision.utils.make_grid(samples, nrow=1, normalize=True)
                writer.add_image("BaseDDPM/Samples", grid, epoch)

    # Save model
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    train_ddpm(
        model=ddpm,
        dataloader=dataloader,
        optimizer=optimizer,
        device=DEVICE,
        nb_epoch=ddpm_hparams["nb_epochs"],
        path="./weights/base_ddpm_v8.pth",
        writer=ddpm_writer,
    )