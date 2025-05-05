import torch
import torch.nn as nn
from models.base_gan_models import BaseCritic, BaseGenerator

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

# class OutpaintingGenerator(BaseGenerator):
#     def __init__(self, latent_dim, img_channels):
#         super().__init__(latent_dim, img_channels)
        
#         # Additional processing for conditional generation
#         self.mask_projection = nn.Sequential(
#             nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
        
#         # Modify first block to accept concatenated inputs
#         original_first = self.net[0]
#         self.net[0] = self._block(
#             latent_dim + 64,  # Now accepts noise + mask features
#             original_first[0].out_channels,
#             original_first[0].kernel_size,
#             original_first[0].stride,
#             original_first[0].padding
#         )
        
#     def forward(self, noise, masked_x):
#         # Process masked input
#         mask_features = self.mask_projection(masked_x)
        
#         # Reshape noise to spatial dimensions and concatenate
#         noise = noise.view(noise.size(0), -1, 1, 1)
#         noise = noise.expand(-1, -1, mask_features.size(2), mask_features.size(3))
        
#         # Combine inputs
#         x = torch.cat([noise, mask_features], dim=1)
#         return super().forward(x)