import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

import torch.nn.functional as F

def pad_to_multiple(x, multiple=8):
    """ Pad (C, H, W) tensor to nearest multiple of `multiple`. """
    _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

class MRIDataset(Dataset):
    def __init__(self, file_list, mask_func, transform=None):
        """
        file_list: list of .h5 file paths
        mask_func: function to generate undersampling mask
        transform: optional transform function
        """
        self.file_list = file_list
        self.mask_func = mask_func
        self.transform = transform
        self.examples = self._prepare_examples()

    def _prepare_examples(self):
        examples = []
        for fname in self.file_list:
            with h5py.File(fname, 'r') as f:
                num_slices = f['kspace'].shape[0]
                for slice_idx in range(num_slices):
                    examples.append((fname, slice_idx))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, slice_idx = self.examples[idx]
        with h5py.File(fname, 'r') as f:
            raw = f['kspace'][slice_idx]  # (ky, kz, 24)
        real = raw[..., :12]
        imag = raw[..., 12:]
        kspace = real + 1j * imag  # (ky, kz, Nc)

        # Trim kz from 180 → 170 if needed
        if kspace.shape[1] == 180:
            kspace = kspace[:, 5:-5, :]  # (ky, 170, Nc)

        # Generate mask
        mask = self.mask_func(kspace.shape[0], kspace.shape[1])  # (ky, kz)
        mask = np.expand_dims(mask, axis=-1)  # (ky, kz, 1)
        undersampled_kspace = kspace * mask

        # IFFT to get images
        zero_filled = np.fft.ifft2(undersampled_kspace, norm='ortho', axes=(0,1))
        target = np.fft.ifft2(kspace, norm='ortho', axes=(0,1))

        # Convert complex to (2, ky, kz, Nc)
        def complex_to_chans(x):
            return np.stack([np.real(x), np.imag(x)], axis=0)  # (2, ky, kz, Nc)

        input_chans = complex_to_chans(zero_filled).transpose(3,0,1,2).reshape(-1, kspace.shape[0], kspace.shape[1])
        target_chans = complex_to_chans(target).transpose(3,0,1,2).reshape(-1, kspace.shape[0], kspace.shape[1])

        # Convert to tensor
        input_tensor = torch.from_numpy(input_chans).float()
        target_tensor = torch.from_numpy(target_chans).float()

        # Pad to multiple of 8
        input_tensor = pad_to_multiple(input_tensor, multiple=8)
        target_tensor = pad_to_multiple(target_tensor, multiple=8)

        # Normalize
        max_val = np.abs(target).max()
        input_tensor /= max_val
        target_tensor /= max_val

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return input_tensor, target_tensor
