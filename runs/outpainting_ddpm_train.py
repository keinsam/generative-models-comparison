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
from datasets.outpainting_datasets import OutpaintingCIFAR10
from models.outpainting_models import OutpaintingDDPM, OutpaintingUNetEpsilon

# === Hyperparams ===
with open("configs/outpainting_hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)
ddpm_hparams = hparams["ddpm"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "outpainting_ddpm_v0"

# === Dataset ===
transforms = transforms.Compose(
    [
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(ddpm_hparams["img_channels"])], [0.5 for _ in range(ddpm_hparams["img_channels"])]
        ),
    ]
)
dataset = OutpaintingCIFAR10(root="data", train=True, download=False, transform=transforms, subset_size=25000)
dataloader = DataLoader(dataset, batch_size=ddpm_hparams["batch_size"], shuffle=True)

# === Model ===
eps_model = OutpaintingUNetEpsilon(n_channel=ddpm_hparams["img_channels"], time_dim=ddpm_hparams["time_dim"]).to(DEVICE)
model = OutpaintingDDPM(eps_model,
                        betas=(ddpm_hparams["beta_start"], ddpm_hparams["beta_end"]),
                        n_T=ddpm_hparams["noise_steps"],
                        criterion=torch.nn.MSELoss()).to(DEVICE)

# === Optimizer ===
optimizer = torch.optim.Adam(model.parameters(), lr=ddpm_hparams["learning_rate"])

# === Logging === 
writer = SummaryWriter(log_dir=f"./logs/{MODEL_NAME}")
writer.add_hparams(ddpm_hparams, {})

# === Training ===
def train_outpainting_ddpm(model, dataloader, optimizer, device, nb_epoch, path, writer=None):
    model.train()
    loss_ema = None

    for epoch in range(nb_epoch):
        pbar = tqdm(dataloader)
        for batch_idx, (masked_image, target_image, mask) in enumerate(pbar):
            masked_image = masked_image.to(device)
            target_image = target_image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            loss = model(target_image, mask)  # Note: we pass the target image, not masked_image
            loss.backward()
            optimizer.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            
            pbar.set_description(f"Epoch {epoch+1}/{nb_epoch}, Loss: {loss_ema:.4f}")

            if writer is not None:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("OutpaintingDDPM/Loss", loss_ema, step)

        if writer is not None and epoch % 1 == 0:
            with torch.no_grad():
                # Visualize some examples
                test_batch = next(iter(dataloader))
                test_masked, test_target, test_mask = [x[:8].to(device) for x in test_batch]
                
                # Generate samples
                samples = model.sample(test_masked, test_mask)
                
                # Create grid of: masked images | generated samples | target images
                grid_input = torchvision.utils.make_grid(test_masked, nrow=8, normalize=True)
                grid_output = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
                grid_target = torchvision.utils.make_grid(test_target, nrow=8, normalize=True)
                grid = torch.cat([grid_input, grid_output, grid_target], dim=1)
                
                writer.add_image("OutpaintingDDPM/Samples", grid, global_step=step)
                torchvision.utils.save_image(grid, f"./samples/{MODEL_NAME}.png")

    torch.save(model.state_dict(), path)


# === Run ===
if __name__ == "__main__":
    train_outpainting_ddpm(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=DEVICE,
        nb_epoch=ddpm_hparams["nb_epochs"],
        path=f"./weights/{MODEL_NAME}.pth",
        writer=writer
    )
