import torch
import torch.nn as nn
from models.base_gan_models import BaseCritic, BaseGenerator
from models.base_ddpm_models import DDPM, UNetEpsilon

class OutpaintingCritic(BaseCritic):
    def __init__(self, img_channels):
        # Double input channels to handle [image, mask] concatenation
        super().__init__(img_channels * 2)
        
    def forward(self, x, masked_x):
        # Concatenate along channel dimension
        x = torch.cat([x, masked_x], dim=1)
        return super().forward(x)
    
class OutpaintingGenerator(BaseGenerator):
    def __init__(self, latent_dim, img_channels):
        super().__init__(latent_dim, img_channels)
        
        # Replace the entire network to ensure proper 32x32 output
        self.net = nn.Sequential(
            # Input will be [noise_features + mask_features] x 1 x 1
            self._block(latent_dim + 64, 512, 4, 1, 0),  # 4x4
            self._block(512, 256, 4, 2, 1),               # 8x8
            self._block(256, 128, 4, 2, 1),               # 16x16
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1),  # 32x32
            nn.Tanh()
        )
        
        # Keep the mask projection
        self.mask_projection = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Reduce to 1x1 to match noise
        )
        
    def forward(self, noise, masked_x):
        # Process masked input to 1x1 features
        mask_features = self.mask_projection(masked_x)
        mask_features = mask_features.view(mask_features.size(0), -1)
        
        # Combine with noise
        x = torch.cat([noise, mask_features], dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        
        return self.net(x)


class OutpaintingDDPM(DDPM):
    def __init__(
        self,
        eps_model,
        betas,
        n_T,
        criterion=nn.MSELoss(),
    ):
        super().__init__(eps_model, betas, n_T, criterion)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Convert RGB mask to single channel by taking first channel
        mask_single = mask[:, 0:1, :, :]  # [B, 1, H, W]
        
        _ts = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # [B, 3, H, W]

        # Create noisy version
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )

        x_t = mask * x + (1 - mask) * x_t

        # Predict noise for the entire image
        # Concatenate single-channel mask (not RGB mask)
        pred_eps = self.eps_model(torch.cat([x_t, mask_single], dim=1), _ts / self.n_T)
        
        pred_eps = pred_eps[:, :3, :, :]  # Take first 3 channels
        
        # Only compute loss on unknown regions (using single-channel mask)
        # Broadcast mask_single to match eps channels
        mask_broadcast = mask_single.expand_as(eps)
        loss = self.criterion(eps * (1 - mask_broadcast), pred_eps * (1 - mask_broadcast))
        return loss

    @torch.no_grad()
    def sample(self, x_start: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Convert RGB mask to single channel
        mask_single = mask[:, 0:1, :, :]  # [B, 1, H, W]
        
        # Start from random noise
        x_i = torch.randn_like(x_start).to(x_start.device)
        # Apply mask immediately to preserve known regions
        x_i = mask * x_start + (1 - mask) * x_i

        for i in range(self.n_T, 0, -1):
            t = torch.full((x_i.size(0),), i, device=x_i.device, dtype=torch.long)
            # Concatenate single-channel mask
            eps = self.eps_model(torch.cat([x_i, mask_single], dim=1), t)
            
            # Take only first 3 channels
            eps = eps[:, :3, :, :]
            
            # Update only the unknown regions
            x_i_unknown = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
            if i > 1:
                z = torch.randn_like(x_i)
                x_i_unknown += self.sqrt_beta_t[i] * z
            
            # Combine known and unknown regions
            x_i = mask * x_start + (1 - mask) * x_i_unknown

        return x_i


class OutpaintingUNetEpsilon(UNetEpsilon):
    def __init__(self, n_channel, time_dim):
        # Input channels = image channels (3) + mask channel (1)
        super().__init__(n_channel + 1, time_dim)  # +1 for single-channel mask
        # Modify final layer to output n_channel (3) instead of n_channel+1
        self.final_conv = nn.Conv2d(64, n_channel, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x contains concatenated [image, mask]
        return super().forward(x, t)