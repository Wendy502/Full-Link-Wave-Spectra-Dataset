"""
Train HsResNetRegression for significant wave height retrieval from SAR.

Usage:
    python train.py --data_path ./dataset/train --valid_path ./dataset/val
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model.ResNet_Encoder import HsResNetRegression
from setting import parse_opts
from data_loader.Dataset import SARWaveHsDataset


opt = parse_opts()

train_set = SARWaveHsDataset(opt.data_path)
train_loader = DataLoader(
    train_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    drop_last=True
)

valid_set = SARWaveHsDataset(opt.valid_path)
valid_loader = DataLoader(
    valid_set,
    batch_size=opt.valid_batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    drop_last=True
)

device = 'cuda' if opt.cuda else 'cpu'
net = HsResNetRegression().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
total_epochs = opt.niter
decay_start  = opt.decay_epoch

def lr_lambda(epoch):
    if epoch < decay_start:
        return 1.0
    else:
        return max(
            0.0,
            1.0 - (epoch - decay_start) / (total_epochs - decay_start)
        )
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

def run_epoch(epoch, loader, train=True):
    net.train() if train else net.eval()
    mode = 'Train' if train else 'Valid'

    total_loss = 0
    preds, gts = [], []

    with torch.set_grad_enabled(train):
        for wave, sar, inc, hs in tqdm(loader, ncols=80, leave=True,
                                        desc='[{:d}/{:d}] {}'.format(epoch, opt.niter, mode)):
            wave, sar, inc, hs = wave.to(device), sar.to(device), inc.to(device), hs.to(device)
            pred = net(wave, sar, inc)
            # If the network outputs multi-dim, take the 0th dim as Hs
            if pred.ndim > 1 and pred.shape[1] > 1:
                loss = criterion(pred[:, 0], hs)
                hs_pred = pred[:, 0]
            else:
                loss = criterion(pred, hs)
                hs_pred = pred

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds.append(hs_pred.detach().cpu())
            gts.append(hs.cpu())

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    rmse = torch.sqrt(((preds - gts) ** 2).mean())
    return total_loss / len(loader), rmse.item()


if __name__ == '__main__':

    log_records = []
    for epoch in range(1, opt.niter + 1):
        train_loss, train_rmse = run_epoch(epoch, train_loader, True)
        valid_loss, valid_rmse = run_epoch(epoch, valid_loader, False)

        print(f"[{epoch}/{opt.niter}] "
              f"Train RMSE: {train_rmse:.4f}, Train Loss: {train_loss:.4f} \n"
              f"[{epoch}/{opt.niter}] "
              f"Valid RMSE: {valid_rmse:.4f}, Valid Loss: {valid_loss:.4f} \n"
              f"[{epoch}/{opt.niter}] "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}"
              )
        scheduler.step()

        log_records.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'valid_loss': valid_loss,
            'valid_rmse': valid_rmse
        })

        torch.save(net.state_dict(), f"{opt.outf}/net_Latest.pth")
        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(),
                       f"{opt.outf}/net_{epoch}.pth")

    df = pd.DataFrame(log_records)
    csv_path = f"{opt.log_path}/training_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training log saved to {csv_path}")
