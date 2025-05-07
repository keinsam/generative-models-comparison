import os
import subprocess
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import lpips

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_mask(path):
    """Load a mask from a file path."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask

def calculate_psnr(original_dir, generated_dir):
    original_files = sorted(os.listdir(original_dir))
    generated_files = sorted(os.listdir(generated_dir))

    psnr_scores = []

    for original_file, generated_file in zip(original_files, generated_files):
        original_img_path = os.path.join(original_dir, original_file)
        generated_img_path = os.path.join(generated_dir, generated_file)

        img1 = load_image(original_img_path)
        img2 = load_image(generated_img_path)
        psnr_scores.append(peak_signal_noise_ratio(img1, img2))

    return np.mean(psnr_scores)

def calculate_ssim(original_dir, generated_dir):
    original_files = sorted(os.listdir(original_dir))
    generated_files = sorted(os.listdir(generated_dir))


    ssim_scores = []

    for original_file, generated_file in zip(original_files, generated_files):
        original_img_path = os.path.join(original_dir, original_file)
        generated_img_path = os.path.join(generated_dir, generated_file)

        img1 = load_image(original_img_path)
        img2 = load_image(generated_img_path)
        ssim_scores.append(structural_similarity(img1, img2, channel_axis=-1))

    return np.mean(ssim_scores)

# def calculate_lpips(original_dir, generated_dir):
#     original_files = sorted(os.listdir(original_dir))
#     generated_files = sorted(os.listdir(generated_dir))


#     lpips_scores = []
#     loss_fn = lpips.LPIPS(net='alex')  # You can choose 'alex', 'vgg', or 'squeeze'

#     for original_file, generated_file in zip(original_files, generated_files):
#         original_img_path = os.path.join(original_dir, original_file)
#         generated_img_path = os.path.join(generated_dir, generated_file)

#         img1 = load_image(original_img_path)
#         img2 = load_image(generated_img_path)
#         img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#         img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#         lpips_scores.append(loss_fn(img1_tensor, img2_tensor).item())

#     return np.mean(lpips_scores)

def calculate_l1_masked(original_dir, generated_dir, mask_dir):
    original_files = sorted(os.listdir(original_dir))
    generated_files = sorted(os.listdir(generated_dir))
    mask_files = sorted(os.listdir(mask_dir))

    l1_masked_scores = []

    for original_file, generated_file, mask_file in zip(original_files, generated_files, mask_files):
        original_img_path = os.path.join(original_dir, original_file)
        generated_img_path = os.path.join(generated_dir, generated_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img1 = load_image(original_img_path)
        img2 = load_image(generated_img_path)
        mask = load_mask(mask_path)

        # Normalisation du masque
        mask = mask.astype(np.float32) / 255.0
        mask = 1 - mask
        diff = np.abs(img1 - img2)
        masked_diff = diff * mask[..., np.newaxis]
        l1_masked_scores.append(np.sum(masked_diff) / np.sum(mask))

    return np.mean(l1_masked_scores)


def calculate_fid(output_gen_dir, output_real_dir):
    command = ["python3", "-m", "pytorch_fid", output_gen_dir,output_real_dir]
    result = subprocess.run(command, capture_output=True, text=True)

    return result.stdout