import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import shutil

from utils.metrics import *
import yaml
import torch
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.inpainting_datasets import InpaintingCIFAR10

with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
WGAN_MODEL_NAME = Path(paths["inpainting_wgan_name"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(WGAN_MODEL_NAME).with_suffix('.pth')
OUTPUT_MASK_DIR = Path(paths["output_mask_dir"])
OUTPUT_REAL_DIR = Path(paths["output_real_dir"])
OUTPUT_GEN_DIR = Path(paths["output_gen_dir"])
with open("configs/inpainting_hparams.yaml", "r") as f:
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

dataset_val = InpaintingCIFAR10(root="data/", train=False, transform=transforms, subset_size=10000, mask_folder=OUTPUT_MASK_DIR) # added


model = torch.load(MODEL_PATH,weights_only=False)


def save_images(images, folder):
    shutil.rmtree(folder)
    """
    Sauvegarde les images dans un répertoire donné avec un préfixe spécifié.

    :param images: Tenseur d'images à sauvegarder
    :param folder: Chemin du répertoire où sauvegarder les images
    :param prefix: Préfixe pour les noms de fichiers
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    for i, img in enumerate(images):
        save_image(img, os.path.join(folder, f"{i}.png"))


def generate_and_save_inpainting(generator, device, dataset_val, latent_dim, output_real_folder, output_gen_folder,
                                 num_images=5):
    generator.eval()
    with torch.no_grad():
        # Charger quelques images du dataset pour l'inpainting
        sample_loader = DataLoader(dataset_val, batch_size=num_images, shuffle=True)
        masked_imgs, real_imgs, _ = next(iter(sample_loader))
        real_imgs = real_imgs.to(device)
        masked_imgs = masked_imgs.to(device)

        # Reconstruire le masque à partir des images (valeurs 0 dans les zones masquées)
        # On suppose que les zones masquées sont remplacées par 0 → créer un masque binaire
        mask = (masked_imgs == 0).float()

        # Générer vecteurs latents
        z = torch.randn(num_images, latent_dim, 1, 1, device=device)
        inpainted_imgs = generator(z, masked_imgs)

        # Fusionner : on remplace seulement les zones masquées
        completed_imgs = masked_imgs * (1 - mask) + inpainted_imgs * mask

        # Dénormalisation [-1, 1] → [0, 1]
        real_imgs = (real_imgs.cpu() + 1) / 2.0
        masked_imgs = (masked_imgs.cpu() + 1) / 2.0
        completed_imgs = (completed_imgs.cpu() + 1) / 2.0

        # Sauvegarder
        save_images(real_imgs, output_real_folder)
        save_images(completed_imgs, output_gen_folder)

        # Affichage
        fig, axs = plt.subplots(3, num_images, figsize=(15, 12))
        for i in range(num_images):
            axs[0, i].imshow(np.transpose(real_imgs[i].numpy(), (1, 2, 0)))
            axs[0, i].axis('off')
            axs[0, i].set_title('Original')

            axs[1, i].imshow(np.transpose(masked_imgs[i].numpy(), (1, 2, 0)))
            axs[1, i].axis('off')
            axs[1, i].set_title('Masked')


            axs[2, i].imshow(np.transpose(completed_imgs[i].numpy(), (1, 2, 0)))
            axs[2, i].axis('off')
            axs[2, i].set_title('Final Inpainting')

        plt.suptitle('Inpainting with Region Fusion')
        plt.show()




if __name__ == "__main__":
    generate_and_save_inpainting(model, DEVICE, dataset_val, wgan_hparams['latent_dim'],OUTPUT_REAL_DIR,OUTPUT_GEN_DIR)
    fid = calculate_fid(OUTPUT_GEN_DIR, OUTPUT_REAL_DIR)
    avg_ssim = calculate_ssim_for_directories(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR)
    avg_l1_masked = calculate_l1_masked_for_directories(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR, OUTPUT_MASK_DIR)

    print(f"Average SSIM: {avg_ssim}")
    print(f"Average L1 (masked): {avg_l1_masked}")
    print(f"FID : {fid}")