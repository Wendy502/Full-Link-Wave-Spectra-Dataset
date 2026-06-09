"""
Evaluate HsResNetRegression on a test set and save predictions as .mat files.

Usage:
    python test.py --test_path ./dataset/test --model_path ./checkpoint/net_Latest.pth
"""
import torch
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from model.ResNet_Encoder import HsResNetRegression
from data_loader.Dataset import SARWaveHsDataset


def evaluate_full(loader, net, device, has_hs_mask):
    """
    Run inference over the full test set.

    Args:
        has_hs_mask: bool list, whether each sample has ground truth Hs
    Returns:
        rmse, bias, preds, gts — rmse/bias may be None if no ground truth
    """
    net.eval()
    preds_list = []
    gts_list = []

    with torch.no_grad():
        for i, (wave, sar, inc, hs) in enumerate(tqdm(loader, ncols=80, desc="Testing")):
            wave, sar, inc = wave.to(device), sar.to(device), inc.to(device)
            pred = net(wave, sar, inc)

            # Multi-dim output → take 0th dim as Hs
            if pred.ndim > 1 and pred.shape[1] > 1:
                pred_hs = pred[:, 0]
            else:
                pred_hs = pred

            preds_list.append(pred_hs.cpu())

            # Only add to gts_list if sample has ground truth
            for j in range(hs.size(0)):
                global_idx = i * loader.batch_size + j
                if global_idx < len(has_hs_mask) and has_hs_mask[global_idx]:
                    gts_list.append(hs[j:j+1].cpu())

    preds = torch.cat(preds_list)

    # Compute metrics (only on samples with ground truth)
    if len(gts_list) > 0:
        gts = torch.cat(gts_list)
        valid_indices = [i for i, has in enumerate(has_hs_mask) if has]
        preds_valid = preds[valid_indices]
        rmse = torch.sqrt(((preds_valid - gts) ** 2).mean()).item()
        bias = (preds_valid - gts).mean().item()
        return rmse, bias, preds, gts
    else:
        print("Warning: No sample has Hs ground truth; cannot compute RMSE/Bias.")
        return None, None, preds, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Hs regression model')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--outf', type=str, default='./mat_folder/test_result',
                        help='Output folder for .mat predictions')
    opt = parser.parse_args()

    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')

    # Initialize network
    net = HsResNetRegression().to(device)
    net.eval()

    # Load checkpoint
    checkpoint = torch.load(opt.model_path, map_location=device)
    net.load_state_dict(checkpoint)
    print(f"Loaded model from {opt.model_path}")

    # Load test set
    test_set = SARWaveHsDataset(opt.test_path)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    num_with_hs = sum(test_set.has_hs)
    print(f"Test set: {len(test_set)} samples, {num_with_hs} with Hs ground truth.")

    # Evaluate
    rmse, bias, preds, gts = evaluate_full(test_loader, net, device, test_set.has_hs)
    if rmse is not None:
        print(f"Test RMSE: {rmse:.4f} m")
        print(f"Test Mean Bias: {bias:.4f} m")

    # Save predictions as .mat
    os.makedirs(opt.outf, exist_ok=True)

    for idx, mat_path in enumerate(test_set.sample_files):
        file_name = os.path.basename(mat_path)
        hs_pred = preds[idx].item()

        save_dict = {'Hs_Retrieved': hs_pred}

        # Also save ground truth if available
        if gts is not None and test_set.has_hs[idx]:
            # Count how many ground truth samples appeared before this index
            gt_idx = sum(test_set.has_hs[:idx])
            hs_true = gts[gt_idx].item()
            save_dict['Hs'] = hs_true

        save_path = os.path.join(opt.outf, file_name)
        import scipy.io as sio
        sio.savemat(save_path, save_dict)

    print(f"Predictions saved to {opt.outf}")
