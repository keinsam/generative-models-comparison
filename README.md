# Comparative Review of WGAN and DDPM on multiple tasks

**Goal :** Compare WGAN and DDPM on at least 4 tasks : denoising, upscaling (super-resolution), inpainting, outpainting.

**Base :** Basic dataset, the model generates images.

**Denoising :** Change the dataset by introducing gaussian or poisson noise, the model has to generate images without the noise.

**Upscaling (super-resolution) :** Change the dataset by downsampling images, the model has to generate higher resolution images.

**Inpainting :** Change the dataset by introduction random black patches in the images, the model has to generate images with the black patch painted.

**Outpainting :** Change the dataset by introducing a black rectangle around the images, the model has to generate images with the black rectangle painted.

**TODO :**
- Implement WGAN and DDPM base model (with same architecture and same number of parameters for fair comparison)
- Code Dataset for each tasks
- Check for which tasks the base model needs to be recoded
- Implement evaluation logic with specialized metrics


### **TODO List for Fair WGAN vs DDPM Comparison**  

#### **1. Base Models (Keep Identical Architectures)**  
- [ ] **WGAN-GP**  
  - Generator: 4-layer CNN (like DCGAN)  
  - Discriminator: 4-layer CNN (mirror of Generator)  
  - **Loss**: Wasserstein
- [ ] **DDPM**  
  - **Same 4-layer CNN** as WGAN’s Generator, but:  
  - Replace final layer with noise prediction  
  - **Loss**: MSE on noise

#### **2. Datasets (For All Tasks)**  
| Task          | Input → Target                  | Notes                          |  
|---------------|---------------------------------|--------------------------------|  
| **Denoising** | Noisy image → Clean image       | Add 15% Gaussian noise        |  
| **Upscaling** | Low-res (64x64) → High-res (256x256) | Bicubic downsampling       |  
| **Inpainting** | Masked image (25% black patches) → Full image | Random square masks |  
| **Outpainting** | Image with black borders → Full image | Extend canvas by 25%      |  

#### **3. Evaluation (Same for Both Models)**  
Task | Metrics

Denoising | PSNR, SSIM, MSE

Super-resolution | PSNR, SSIM, LPIPS, FID

Inpainting | L1 (masked), SSIM, FID

Outpainting | L1 (border), SSIM, FID