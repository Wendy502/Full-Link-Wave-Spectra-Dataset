"""
Quick sanity check: forward + backward pass with random inputs.
Run: python test_backward.py
"""
import torch
import torch.nn as nn
from model.ResNet_Encoder import HsResNetRegression

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = HsResNetRegression().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

wave = torch.randn(4, 1, 256, 256).to(device)
sar  = torch.randn(4, 1, 256, 256).to(device)
inc  = torch.randn(4, 1, 256, 256).to(device)
hs_gt = torch.randn(4).to(device)

pred = net(wave, sar, inc)
loss = criterion(pred, hs_gt)

print("Loss:", loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Backward success! OK")
