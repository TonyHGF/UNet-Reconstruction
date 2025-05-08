import os
import glob
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from loss import CombinedLoss
from dataset import MRIDataset
from mask import create_mask_func
from utils import calculate_psnr, calculate_ssim, save_checkpoint

# argparse 设置
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--acceleration', type=int, default=4, help='Acceleration factor')
parser.add_argument('--mask-type', type=str, default='cartesian', help='Mask type (cartesian or radial)')
args = parser.parse_args()

# 配置参数
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs
acceleration = args.acceleration
mask_type = args.mask_type
in_channels = 24  # 2 * Nc (real + imag), Nc=12
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据目录
train_dir = '/home/tony_hu/data/Train'
val_dir = '/home/tony_hu/data/Val'

if __name__ == '__main__':
    print("\n========== Training Configuration ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("===========================================\n")


    # 自动搜集 .h5 文件
    train_files = sorted(glob.glob(os.path.join(train_dir, '*.h5')))
    val_files = sorted(glob.glob(os.path.join(val_dir, '*.h5')))
    print(f'Found {len(train_files)} training files, {len(val_files)} validation files.')

    # 准备 mask function
    mask_func = create_mask_func(mask_type, acceleration, center_fraction=0.1)

    # 准备数据集和 dataloader
    train_dataset = MRIDataset(train_files, mask_func)
    val_dataset = MRIDataset(val_files, mask_func)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, (inputs, targets) in enumerate(train_loader):
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        break  # 只看第一个 batch，防止跑完全部

    

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型
    model = UNet(in_channels=in_channels, out_channels=in_channels).to(device)

    # 损失函数
    loss_fn = CombinedLoss(alpha=0.8, data_range=1.0)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_psnr += calculate_psnr(outputs, targets, data_range=1.0) * inputs.size(0)
                val_ssim += calculate_ssim(outputs, targets, data_range=1.0) * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)
        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val PSNR: {val_psnr:.2f} dB | '
              f'Val SSIM: {val_ssim:.4f}')

        # ---------- Save Checkpoints ----------
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, checkpoint_dir=save_dir)
