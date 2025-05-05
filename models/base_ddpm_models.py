import torch
import torch.nn as nn
from typing import Dict, Tuple


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DummyEpsModel(nn.Module) :
    def __init__(self, n_channel, time_dim) :
        super().__init__()
        self.n_channel = n_channel
        self.time_dim = time_dim
        self.conv = nn.Sequential(  # with batchnorm
            self._block(n_channel, 64),
            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 512),
            self._block(512, 256), #upblock
            self._block(256, 128), #upblock
            self._block(128, 64), #upblock
            nn.Conv2d(64, n_channel, 3, padding=1),
            nn.Tanh(),
        )
        self.emb_proj = nn.Linear(time_dim, n_channel)

    def _block(self, in_channels, out_channels) :
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    # def _up_block(self, in_channels, out_channels) :
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, padding=3),
    #         nn.BatchNorm2d(out_channels),
    #         nn.LeakyReLU(),
    #     )
    
    def positional_encoding(self, t, channels) :
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc = torch.cat([
            torch.sin(t * inv_freq),
            torch.cos(t * inv_freq)
        ], dim=-1)
        return pos_enc

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        t = t.unsqueeze(-1).float()
        t = self.positional_encoding(t, self.time_dim)
        emb = self.emb_proj(t)[:, :, None, None] # .repeat(1, 1, x.shape[-2], x.shape[-1]) #.expand_as(x)
        emb = emb.expand(-1, -1, x.shape[-2], x.shape[-1])
        # emb = self.emb_proj(t).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + emb
        x = self.conv(x)
        return x

class UNetEpsModel(nn.Module):
    def __init__(self, n_channel, time_dim):
        super().__init__()
        self.n_channel = n_channel
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4)
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(n_channel, 64, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = self._down_block(64, 128)
        self.down2 = self._down_block(128, 256)
        self.down3 = self._down_block(256, 512)
        
        # Middle
        self.middle = nn.Sequential(
            self._block(512, 512)
        )
        
        # Upsampling path with skip connections
        self.up1 = self._up_block(512, 256)
        self.up1_conv = self._block(512, 256)  # For handling skip connection
        
        self.up2 = self._up_block(256, 128)
        self.up2_conv = self._block(256, 128)  # For handling skip connection
        
        self.up3 = self._up_block(128, 64)
        self.up3_conv = self._block(128, 64)   # For handling skip connection
        
        # Final output
        self.final_conv = nn.Conv2d(64, n_channel, kernel_size=3, padding=1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._block(in_channels, out_channels)
        )
    
    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self._block(in_channels, out_channels)
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU()
        )
    
    def positional_encoding(self, t, channels):
        """
        Enhanced positional encoding for better temporal information
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.positional_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Initial conv
        x1 = self.init_conv(x)
        
        # Downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Middle processing with time embedding
        x4 = x4 + t_emb[:, :, None, None].expand(-1, -1, x4.shape[-2], x4.shape[-1])
        x4 = self.middle(x4)
        
        # Upsampling with skip connections
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.up1_conv(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.up2_conv(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.up3_conv(x)
        
        # Output
        return self.final_conv(x)

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            # eps = self.eps_model(x_i, i / self.n_T)
            t = torch.full((x_i.size(0),), i, device=x_i.device, dtype=torch.long)     # added
            eps = self.eps_model(x_i, t)                                               # added
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i