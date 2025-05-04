# from tqdm import tqdm
# import torch
# import torch.nn as nn

# class Diffusion:
#     def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cpu"):
#         self.noise_steps = noise_steps
#         self.beta_start = beta_start
#         self.beta_end = beta_end
#         self.img_size = img_size
#         self.device = device

#         self.beta = self.prepare_noise_schedule().to(device)
#         self.alpha = 1. - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)

#     def prepare_noise_schedule(self):
#         return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

#     def noise_images(self, x, t):
#         sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
#         sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
#         eps = torch.randn_like(x)
#         return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

#     def sample_timesteps(self, n):
#         return torch.randint(low=1, high=self.noise_steps, size=(n,))

#     def sample(self, model, n):
#         model.eval()
#         with torch.no_grad():
#             x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
#             for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
#                 t = (torch.ones(n) * i).long().to(self.device)
#                 predicted_noise = model(x, t)
#                 alpha = self.alpha[t][:, None, None, None]
#                 alpha_hat = self.alpha_hat[t][:, None, None, None]
#                 beta = self.beta[t][:, None, None, None]
#                 if i > 1:
#                     noise = torch.randn_like(x)
#                 else:
#                     noise = torch.zeros_like(x)
#                 x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
#         model.train()
#         x = (x.clamp(-1, 1) + 1) / 2
#         x = (x * 255).type(torch.uint8)
#         return x

# class DDPM_Down(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(1, out_channels),
#             nn.GELU(),
#             nn.MaxPool2d(2)
#         )
#         self.emb_proj = nn.Linear(emb_dim, out_channels)

#     def forward(self, x, t):
#         x = self.conv(x)
#         emb = self.emb_proj(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) #.expand_as(x)
#         return x + emb

# class DDPM_Up(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(1, out_channels),
#             nn.GELU()
#         )
#         self.emb_proj = nn.Linear(emb_dim, out_channels)

#     def forward(self, x, t):
#         x = self.up(x)
#         x = self.conv(x)
#         emb = self.emb_proj(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) #.expand_as(x)
#         return x + emb

# class DDPM(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, time_dim=256):
#         super().__init__()
#         self.time_dim = time_dim
        
#         # Downsample
#         self.down1 = DDPM_Down(in_channels, 64, time_dim)
#         self.down2 = DDPM_Down(64, 128, time_dim)
#         self.down3 = DDPM_Down(128, 256, time_dim)
        
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.GroupNorm(1, 256),
#             nn.GELU()
#         )
        
#         # Upsample
#         self.up1 = DDPM_Up(256, 128, time_dim)
#         self.up2 = DDPM_Up(128, 64, time_dim)
#         self.up3 = DDPM_Up(64, out_channels, time_dim)
        
#         # Output
#         self.out = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#             nn.Tanh()
#         )

#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
#         pos_enc = torch.cat([
#             torch.sin(t * inv_freq),
#             torch.cos(t * inv_freq)
#         ], dim=-1)
#         #     pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         #     pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(self, x, t):
#         # Time embedding
#         t = t.unsqueeze(-1).float()
#         t = self.pos_encoding(t, self.time_dim)
        
#         # Downsample
#         x1 = self.down1(x, t)
#         x2 = self.down2(x1, t)
#         x3 = self.down3(x2, t)
        
#         # Bottleneck
#         x = self.bottleneck(x3)
        
#         # Upsample
#         x = self.up1(x, t)
#         x = self.up2(x, t)
#         x = self.up3(x, t)
        
#         return self.out(x)



# class GAN_Critic(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 16x16
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # 8x8
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # 4x4
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 1)
#         )

#     def forward(self, x):
#         return self.model(x)

# class GAN_Generator(nn.Module):
#     def __init__(self, latent_dim=100, out_channels=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 256 * 4 * 4),
#             nn.ReLU(True),
#             nn.Unflatten(1, (256, 4, 4)),

#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 32x32
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z)

# def initialize_weights(self):
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight)

#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)

#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight)
#             nn.init.constant_(m.bias, 0)