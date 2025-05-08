# Generative Models Comparison: WGAN vs DDPM for Image Restoration

This repository contains the implementation and experiments comparing Wasserstein Generative Adversarial Networks (WGAN) and Denoising Diffusion Probabilistic Models (DDPM) on three image restoration tasks: inpainting, outpainting, and super-resolution. The report is available [here](https://keinsam.github.io/generative-models-comparison/report.pdf).

## Structure

```
.
├── configs/              # Hyperparameters and paths
├── data/                 # Data files
├── datasets/             # Custom dataset implementations
├── docs/                 # Project documentation
├── logs/                 # Training logs (TensorBoard)
├── models/               # Model implementations
├── runs/                 # Training and inference scripts
├── samples/              # Generated output samples
├── utils/                # Utility functions and metrics
└── weights/              # Trained model weights
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/keinsam/generative-models-comparison.git
   cd generative-models-comparison
   ```

2. Train a model:
   ```bash
   python runs/train/inpainting_gan_train.py
   python runs/train/inpainting_ddpm_train.py
   python runs/train/outpainting_gan_train.py
   python runs/train/outpainting_ddpm_train.py
   ```

3. Evaluate a model and generate samples:
   ```bash
   python runs/infer/inpainting_gan_train.py
   python runs/infer/inpainting_ddpm_train.py
   python runs/infer/outpainting_gan_train.py
   python runs/infer/outpainting_ddpm_train.py
   ```

## Configurations

Modify YAML files in `configs/` to adjust:
- Model hyperparameters
- Training parameters
- Paths

## Results

Sample outputs are saved in `samples/` directory. Results are logged in TensorBoard (in `logs/`).

## References

This implementation is based on:
- WGAN: https://arxiv.org/abs/1701.07875
- DDPM: https://arxiv.org/abs/2006.11239