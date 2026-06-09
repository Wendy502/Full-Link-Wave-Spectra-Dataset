import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, out_dim=256, weights=None):
        super().__init__()

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class HsResNetRegression(nn.Module):
    """
    Significant Wave Height regression model.

    Inputs:
        wave_spec       [B, 1, 256, 256] — wave spectrum
        sar_spec        [B, 1, 256, 256] — SAR spectrum
        incidence_angle [B, 1, 256, 256] — incidence angle matrix

    Output:
        hs  [B] — predicted significant wave height (meters)
    """
    def __init__(self):
        super().__init__()

        self.wave_encoder = ResNetEncoder(out_dim=256)
        self.sar_encoder  = ResNetEncoder(out_dim=256)
        self.angle_encoder = ResNetEncoder(out_dim=256)

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, wave_spec, sar_spec, incidence_angle):
        # f_wave  = self.wave_encoder(wave_spec)
        f_sar   = self.sar_encoder(sar_spec)
        f_angle = self.angle_encoder(incidence_angle)

        # Use SAR + incidence angle features (not wave features, which serve as ground truth reference)
        fused = torch.cat([f_sar, f_angle], dim=1)
        hs = self.regressor(fused)
        return hs.squeeze(1)
