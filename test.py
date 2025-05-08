import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model import UNet
from utils import calculate_psnr, calculate_ssim

# 配置
VAL_DIR = '/home/tony_hu/data/Val'
CHECKPOINT_PATH = './checkpoints/best_checkpoint.pth'
BATCH_SIZE = 8
MASK_TYPE = 'cartesian'
ACCELERATION = 4

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=24, out_channels=24).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 构造数据集
from mask import create_mask_func
mask_func = create_mask_func(MASK_TYPE, ACCELERATION, center_fraction=0.1)
val_files = sorted(glob.glob(os.path.join(VAL_DIR, '*.h5')))
val_dataset = MRIDataset(val_files, mask_func)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 评估
val_loss = 0
val_psnr = 0
val_ssim = 0
criterion = torch.nn.L1Loss()
count = 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        val_loss += loss.item() * inputs.size(0)

        # 注意：需要用实际范围（未归一化时）
        max_val = targets.max().item()
        psnr = calculate_psnr(outputs, targets, data_range=max_val)
        ssim = calculate_ssim(outputs, targets, data_range=max_val)
        val_psnr += psnr * inputs.size(0)
        val_ssim += ssim * inputs.size(0)

        count += inputs.size(0)

val_loss /= count
val_psnr /= count
val_ssim /= count

print(f"Test Loss: {val_loss:.4f}")
print(f"Test PSNR: {val_psnr:.2f} dB")
print(f"Test SSIM: {val_ssim:.4f}")
