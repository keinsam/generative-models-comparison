import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.base_datasets import BaseCIFAR10
from models.base_models import DDPM, Diffusion

def train_ddpm(model, diffusion, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            # ADD THESE DEBUG PRINTS RIGHT AT THE START OF THE LOOP
            # print("\nDEBUG - Raw input shape:", images.shape)  # Should be [64, 3, 32, 32]
            
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            # ADD THIS TO CHECK NOISE GENERATION
            # print("DEBUG - Noised images shape:", x_t.shape)  # Should match input
            # print("DEBUG - Noise shape:", noise.shape)  # Should match input
            
            predicted_noise = model.forward(x_t, t)
            
            # CRITICAL SIZE CHECK
            # print("DEBUG - Model output shape:", predicted_noise.shape)  # Must match noise shape
            # print("DEBUG - Target noise shape:", noise.shape)  # Must match output
            
            loss = nn.MSELoss()(noise, predicted_noise)
            # print("DEBUG - Current loss:", loss.item())  # Monitor loss values

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def generate_and_visualize_samples(model, diffusion, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        # Generate samples
        samples = diffusion.sample(model, n=num_samples)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()

        # Plot the samples
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i])
            ax.axis("off")
        plt.show()

if __name__ == "__main__" :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = BaseCIFAR10(root="data/", train=True, transform=transform, subset_size=5000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DDPM().to(device)
    diffusion = Diffusion(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_ddpm(model, diffusion, dataloader, optimizer, device, epochs)
    save_model(model, "weights/ddpm_base.pth")
    generate_and_visualize_samples(model, diffusion, device)