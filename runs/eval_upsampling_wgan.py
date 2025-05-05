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
from datasets.upsampling_datasets import UpsamplingCIFAR10

with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
WGAN_MODEL_NAME = Path(paths["upsampling_wgan_name"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(WGAN_MODEL_NAME).with_suffix('.pth')
OUTPUT_REAL_DIR = Path(paths["output_real_dir"])
OUTPUT_GEN_DIR = Path(paths["output_gen_dir"])
with open("configs/upsampling_hparams.yaml", "r") as f:
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

dataset_val = UpsamplingCIFAR10(root="data/", train=False, transform=transforms, subset_size=10000) # added


model = torch.load(MODEL_PATH,weights_only=False)


def save_images(images, folder):
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


def generate_and_save_upsampling(generator, device, dataset_val, latent_dim, output_real_folder, output_gen_folder,
                                 num_images=5):
    print(MODEL_PATH)
    generator.eval()
    with torch.no_grad():
        # Charger quelques images du dataset pour l'upsampling
        sample_loader = DataLoader(dataset_val, batch_size=num_images, shuffle=True)
        low_res_image, high_res_image = next(iter(sample_loader))
        low_res_image = low_res_image.to(device)
        high_res_image = high_res_image.to(device)

        upsampled_imgs = generator(low_res_image)
        print(upsampled_imgs)
        upsampled_imgs= upsampled_imgs.cpu()
        # Dénormaliser les images
        high_res_image = (high_res_image.cpu() + 1) / 2.0
        upsampled_imgs = (upsampled_imgs + 1) / 2.0

        # Sauvegarder les images
        save_images(high_res_image, output_real_folder)
        save_images(upsampled_imgs, output_gen_folder)

        # Afficher les images
        fig, axs = plt.subplots(2, num_images, figsize=(15, 9))
        for i in range(num_images):
            axs[0, i].imshow(np.transpose(high_res_image[i].numpy(), (1, 2, 0)))
            axs[0, i].axis('off')
            axs[0, i].set_title('Image Expected')

            axs[1, i].imshow(np.transpose(upsampled_imgs[i].numpy(), (1, 2, 0)))
            axs[1, i].axis('off')
            axs[1, i].set_title('Upsampled')

        plt.suptitle('Upsampling Results')
        plt.show()



if __name__ == "__main__":
    generate_and_save_upsampling(model, DEVICE, dataset_val, wgan_hparams['latent_dim'],OUTPUT_REAL_DIR,OUTPUT_GEN_DIR)
    fid = calculate_fid(OUTPUT_GEN_DIR, OUTPUT_REAL_DIR)
    avg_ssim = calculate_ssim_for_directories(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR)
    avg_psnr = calculate_psnr_for_directories(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR)
    avg_lpips = calculate_lpips_for_directories(OUTPUT_REAL_DIR, OUTPUT_GEN_DIR)



    print(f"Average SSIM: {avg_ssim}")
    print(f"Average PSNR : {avg_psnr}")
    print(f"Average LPIPS : {avg_lpips}")
    print(f"FID : {fid}")
