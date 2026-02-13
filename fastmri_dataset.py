import random
import pathlib
import numpy as np
import scipy.io as sio
import torch
import h5py
from torch.utils.data import Dataset
from utils import normalize_zero_to_one
import hashlib

def center_crop_or_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop or zero-pad a 2D array to (target_h, target_w)."""
    h, w = img.shape
    # Crop
    if h > target_h:
        top = (h - target_h) // 2
        img = img[top:top + target_h, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        img = img[:, left:left + target_w]
        w = target_w
    # Pad
    if h < target_h or w < target_w:
        out = np.zeros((target_h, target_w), dtype=img.dtype)
        top = (target_h - h) // 2
        left = (target_w - w) // 2
        out[top:top + h, left:left + w] = img
        img = out
    return img

def _split_theta_lambda(mask_under_2ch: torch.Tensor, base_seed: int, epoch: int, key: str, theta_frac: float):
    """
    mask_under_2ch: (H,W,2) float {0,1}
    returns mask_theta_2ch, mask_lambda_2ch with:
      theta ∩ lambda = ∅
      theta ∪ lambda = under
    """
    # use 1-channel for sampling indices
    u = (mask_under_2ch[..., 0] > 0.5).cpu().numpy()
    coords = np.argwhere(u)  # (N,2)
    N = coords.shape[0]
    if N == 0:
        # degenerate case
        return mask_under_2ch.clone(), mask_under_2ch.clone() * 0.0

    n_theta = int(round(theta_frac * N))
    n_theta = max(1, min(N-1, n_theta))  # keep both non-empty

    # deterministic seed from (base_seed, epoch, file+slice key)
    h = hashlib.md5(f"{base_seed}_{epoch}_{key}".encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)  # 32-bit
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    theta_idx = perm[:n_theta]
    lambda_idx = perm[n_theta:]

    H, W = u.shape
    theta = np.zeros((H, W), dtype=np.float32)
    lam   = np.zeros((H, W), dtype=np.float32)
    theta[tuple(coords[theta_idx].T)] = 1.0
    lam[tuple(coords[lambda_idx].T)] = 1.0

    theta_2ch = torch.from_numpy(np.stack([theta, theta], axis=-1))
    lam_2ch   = torch.from_numpy(np.stack([lam, lam], axis=-1))
    return theta_2ch, lam_2ch

class FastMRIData(Dataset):
    """
    fastMRI singlecoil adapter.
    Matches IXIData __getitem__ signature:
      label, mask_under, mask_net_up, mask_net_down, file_name, slice_id

    label is taken from reconstruction_esc if present else reconstruction_rss.
    """
    def __init__(self, data_path, u_mask_path, s_mask_up_path, s_mask_down_path, sample_rate,
                 file_list=None,
                 label_key_preference=("reconstruction_esc", "reconstruction_rss"),
                 seed=30, ssdu=False, ssdu_theta_frac=0.5):
        super().__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        self.sample_rate = sample_rate
        self.label_key_preference = label_key_preference
        self.seed = int(seed)
        self.ssdu = bool(ssdu)
        self.ssdu_theta_frac = float(ssdu_theta_frac)
        self.epoch = 0

        # Collect .h5 files (either from file_list or from directory)
        if file_list is not None:
            with open(file_list, "r", encoding="utf-8") as fh:
                files = [pathlib.Path(line.strip()) for line in fh if line.strip()]
        else:
            files = sorted(list(pathlib.Path(self.data_path).glob("*.h5")))

        if len(files) == 0:
            raise FileNotFoundError("No .h5 files found (check data_path or file_list)")

        # Build (file, slice_id) examples using number of slices in kspace
        self.examples = []
        for f in files:
            with h5py.File(str(f), "r") as hf:
                if "kspace" in hf:
                    n_slices = hf["kspace"].shape[0]
                else:
                    # fall back to reconstruction keys
                    k = next((kk for kk in self.label_key_preference if kk in hf), None)
                    if k is None:
                        raise KeyError(f"{f.name}: no 'kspace' and no recon keys {self.label_key_preference}")
                    n_slices = hf[k].shape[0]
            self.examples += [(f, s) for s in range(int(n_slices))]

        if self.sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = max(1, round(len(self.examples) * self.sample_rate))
            self.examples = self.examples[:num_examples]

        # Load masks from .mat variable name 'mask'
        self.mask_under = np.array(sio.loadmat(self.u_mask_path)["mask"])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)["mask"])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)["mask"])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        # Stack to (..., 2) and convert to torch float (matches IXI_dataset)
        self.mask_under = torch.from_numpy(np.stack((self.mask_under, self.mask_under), axis=-1)).float()
        self.mask_net_up = torch.from_numpy(np.stack((self.mask_net_up, self.mask_net_up), axis=-1)).float()
        self.mask_net_down = torch.from_numpy(np.stack((self.mask_net_down, self.mask_net_down), axis=-1)).float()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        f, slice_id = self.examples[idx]
        slice_id = int(slice_id)

        # Mask dims define the expected spatial size in this repo
        mh, mw = int(self.mask_under.shape[0]), int(self.mask_under.shape[1])

        with h5py.File(str(f), "r") as hf:
            label_key = next((kk for kk in self.label_key_preference if kk in hf), None)

            if label_key is None:
                # No ground truth in this file (common in fastMRI test split)
                # Return a dummy label of correct shape + flag so forward() can skip metrics/loss.
                label = np.zeros((mh, mw), dtype=np.float32)
                has_label = False
            else:
                label = np.asarray(hf[label_key][slice_id])  # (H, W)
                label = center_crop_or_pad(label, mh, mw)
                has_label = True

            kspace = np.asarray(hf["kspace"][slice_id])  # complex64, shape (H, W) for singlecoil
            # crop/pad real and imag separately to mask size
            kspace_r = center_crop_or_pad(kspace.real.astype(np.float32), mh, mw)
            kspace_i = center_crop_or_pad(kspace.imag.astype(np.float32), mh, mw)
            kspace_2ch = np.stack([kspace_r, kspace_i], axis=-1)  # (H, W, 2)
            kspace_2ch = torch.from_numpy(kspace_2ch).float()

        # Keep tensor shape consistent for batching: (H, W, 1)
        label = normalize_zero_to_one(label, eps=1e-6)
        label = torch.from_numpy(label).unsqueeze(-1).float()

        if self.ssdu:
            key = f"{f.name}:{slice_id}"
            mask_theta, mask_lambda = _split_theta_lambda(
            self.mask_under, self.seed, self.epoch, key, self.ssdu_theta_frac
            )
            mask_theta = mask_theta.float()
            mask_lambda = mask_lambda.float()
            return label, self.mask_under, mask_theta, mask_lambda, f.name, slice_id, has_label


        return label, self.mask_under, self.mask_net_up, self.mask_net_down, f.name, slice_id, has_label


