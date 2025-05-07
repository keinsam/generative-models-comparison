import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from datasets.inpainting_datasets import InpaintingCIFAR10
from models.inpainting_models import InpaintingDDPM, InpaintingUNetEpsilon

# === Hyperparams ===

with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DDPM_MODEL_NAME = Path(paths["inpainting_ddpm_name"])
WEIGHT_DIR = Path(paths["weight_dir"])
WEIGHT_PATH = WEIGHT_DIR.joinpath(DDPM_MODEL_NAME).with_suffix('.pth')
LOG_DIR = Path(paths["log_dir"])
LOG_DIR = LOG_DIR.joinpath(DDPM_MODEL_NAME)
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/inpainting_hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)
ddpm_hparams = hparams["ddpm"]
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
dataset = InpaintingCIFAR10(root="data", train=True, download=False, transform=transforms, subset_size=10000)
dataloader = DataLoader(dataset, batch_size=ddpm_hparams["batch_size"], shuffle=True)

# === Model ===
eps_model = InpaintingUNetEpsilon(n_channel=ddpm_hparams["img_channels"], time_dim=ddpm_hparams["time_dim"]).to(DEVICE)
model = InpaintingDDPM(eps_model,
                        betas=(ddpm_hparams["beta_start"], ddpm_hparams["beta_end"]),
                        n_T=ddpm_hparams["noise_steps"],
                        criterion=torch.nn.MSELoss()).to(DEVICE)

# === Optimizer ===
optimizer = torch.optim.Adam(model.parameters(), lr=ddpm_hparams["learning_rate"])

# === Logging ===
writer = SummaryWriter(log_dir=LOG_DIR)


# === Training ===
def train_inpainting_ddpm(model, dataloader, optimizer, device, nb_epoch, path, writer=None):
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

            pbar.set_description(f"Epoch {epoch + 1}/{nb_epoch}, Loss: {loss_ema:.4f}")

            if writer is not None:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("InpaintingDDPM/Loss", loss_ema, step)

        if writer is not None and epoch % 5 == 0:
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

                writer.add_image("InpaintingDDPM/Samples", grid, global_step=step)

    torch.save(model.state_dict(), path)


# === Run ===
if __name__ == "__main__":
    train_inpainting_ddpm(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=DEVICE,
        nb_epoch=ddpm_hparams["nb_epochs"],
        path=WEIGHT_PATH,
        writer=writer
    )