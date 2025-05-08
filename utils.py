import os
import torch
import numpy as np
from pytorch_msssim import ssim

def calculate_psnr(pred, target, data_range=1.0):
    """
    Compute PSNR between predicted and target images.
    Args:
        pred, target: shape (batch, channel, height, width), torch.Tensor
        data_range: max pixel value, typically 1.0 or 255
    Returns:
        Average PSNR over the batch (float)
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # per sample
    data_range_tensor = torch.tensor(data_range, device=pred.device, dtype=pred.dtype)
    psnr = 20 * torch.log10(data_range_tensor) - 10 * torch.log10(mse + 1e-8)
    return psnr.mean().item()

def calculate_ssim(pred, target, data_range=1.0):
    """
    Compute SSIM between predicted and target images.
    Args:
        pred, target: shape (batch, channel, height, width), torch.Tensor
        data_range: max pixel value, typically 1.0 or 255
    Returns:
        Average SSIM over the batch (float)
    """
    data_range_tensor = torch.tensor(data_range, device=pred.device, dtype=pred.dtype)
    ssim_val = ssim(pred, target, data_range=data_range_tensor.item(), size_average=True)
    return ssim_val.item()

def save_checkpoint(state, is_best, checkpoint_dir, filename='last_checkpoint.pth'):
    """
    Save model checkpoint.
    Args:
        state: dict, includes epoch, model state_dict, optimizer state_dict
        is_best: bool, if True also save as 'best_checkpoint.pth'
        checkpoint_dir: directory to save files
        filename: name of the last checkpoint file
    """
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """
    Load model and optimizer from checkpoint.
    Args:
        model: model instance
        optimizer: optimizer instance
        checkpoint_path: file path to checkpoint
        device: 'cuda' or 'cpu'
    Returns:
        model, optimizer, epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch
