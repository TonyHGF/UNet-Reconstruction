import numpy as np
from abc import ABC, abstractmethod

class BaseMaskFunc(ABC):
    """抽象类：所有 mask function 都要继承它"""
    def __init__(self, acceleration, center_fraction=0.08):
        """
        acceleration: 下采样倍数
        center_fraction: 保留低频区域的比例
        """
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    @abstractmethod
    def __call__(self, ky, kz):
        pass

class RandomMaskFunc(BaseMaskFunc):
    """完全随机 mask"""
    def __call__(self, ky, kz):
        mask = np.zeros((ky, kz), dtype=np.float32)
        num_samples = int(ky * kz / self.acceleration)
        sampled_indices = np.random.choice(ky * kz, num_samples, replace=False)
        mask.flat[sampled_indices] = 1
        return mask

class CartesianMaskFunc(BaseMaskFunc):
    """1D Cartesian mask (只沿 ky 随机下采样)"""
    def __call__(self, ky, kz):
        mask = np.zeros((ky, kz), dtype=np.float32)
        num_low_freqs = int(ky * self.center_fraction)
        center = ky // 2
        mask[center - num_low_freqs // 2:center + num_low_freqs // 2, :] = 1

        # 随机采样其他位置
        num_high_freqs = int((ky - num_low_freqs) / self.acceleration)
        prob = float(num_high_freqs) / (ky - num_low_freqs)
        random_mask = (np.random.rand(ky, 1) < prob).astype(np.float32)
        mask = np.maximum(mask, np.tile(random_mask, (1, kz)))
        return mask

def create_mask_func(name, acceleration, center_fraction=0.08):
    if name == 'random':
        return RandomMaskFunc(acceleration, center_fraction)
    elif name == 'cartesian':
        return CartesianMaskFunc(acceleration, center_fraction)
    else:
        raise ValueError(f'Unknown mask type: {name}')
