import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import rasterio

from .transform import Transforms


IMG_EXTS = (".tif", ".tiff")


def make_dataset(dir_path: str, filelist: Optional[List[str]] = None):
    if filelist is None:
        fnames = sorted(os.listdir(dir_path))
    else:
        fnames = [f.strip() for f in filelist if f.strip()]

    paths, names = [], []
    for f in fnames:
        base, ext = os.path.splitext(f)
        if ext:
            p = os.path.join(dir_path, f)
            n = f
        else:
            p = os.path.join(dir_path, base + ".tif")
            n = base + ".tif"
        paths.append(p)
        names.append(n)
    return paths, names


def _percentile_stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        vmin, vmax = np.percentile(arr, [1, 99])
        vmax = max(vmax, vmin + 1.0)
        out = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
        return (out * 255.0).round().astype(np.uint8)
    else:
        out = np.empty_like(arr, dtype=np.uint8)
        for c in range(arr.shape[-1]):
            vmin, vmax = np.percentile(arr[..., c], [1, 99])
            vmax = max(vmax, vmin + 1.0)
            ch = np.clip((arr[..., c] - vmin) / (vmax - vmin + 1e-6), 0, 1)
            out[..., c] = (ch * 255.0).round().astype(np.uint8)
        return out


def read_redd_tif_first3_as_uint8(path: str, out_hw: Tuple[int, int]) -> Image.Image:
    with rasterio.open(path) as src:
        data = src.read()  # (C,H,W) or (H,W)
        if data.ndim == 2:
            data = data[None, ...]  # (1,H,W)
    data = np.transpose(data, (1, 2, 0))  # (H,W,C)

    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    elif data.shape[-1] >= 3:
        data = data[..., :3]
    else:
        data = np.repeat(data, 3, axis=-1)

    data = _percentile_stretch_to_uint8(data)

    H, W = out_hw
    img = Image.fromarray(data)  
    img = img.resize((W, H), Image.BILINEAR)
    return img


def read_label_as_pil(path: str, out_hw: Tuple[int, int]) -> Image.Image:
    """
    Read single-channel label and resize with NEAREST. No remap here.
    """
    with rasterio.open(path) as src:
        lab = src.read(1)  # (H,W)
    lab = Image.fromarray(lab.astype(np.uint8))
    H, W = out_hw
    lab = lab.resize((W, H), Image.NEAREST)
    return lab


class Identity:
    def __call__(self, sample):
        return sample

class Load_Dataset(Dataset):
    """
    For REDDCD:
      - Read .tif -> first 3 bands (no reordering)
      - Per-channel 1–99% percentile stretch -> uint8 (align with your overfit code)
      - Resize(BILINEAR), label resize(NEAREST)
      - Apply paired Transforms()
      - ToTensor + ImageNet Normalize (once)
      - Ensure label is {0,1}
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_size = int(getattr(opt, "input_size", 256))

        if opt.dataset == 'REDDCD':
            self.base_dir = opt.dataroot
            split_dir = os.path.join(self.base_dir, 'splits')
            os.makedirs(split_dir, exist_ok=True)

            split_file = os.path.join(split_dir, f"{opt.phase}.txt")
            if not os.path.exists(os.path.join(split_dir, 'train.txt')):
                print("Splitting dataset into train/val/test...")
                all_files = sorted(os.listdir(os.path.join(self.base_dir, '2017')))
                random.shuffle(all_files)
                n = len(all_files)
                n_train = int(0.7 * n)
                n_val = int(0.15 * n)

                train_list = all_files[:n_train]
                val_list   = all_files[n_train:n_train + n_val]
                test_list  = all_files[n_train + n_val:]

                with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
                    for name in train_list: f.write(name + '\n')
                with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
                    for name in val_list: f.write(name + '\n')
                with open(os.path.join(split_dir, 'test.txt'), 'w') as f:
                    for name in test_list: f.write(name + '\n')
                print("Splitting done.")

            filelist = open(split_file).read().splitlines()

            self.dir1 = os.path.join(self.base_dir, '2017')
            self.dir2 = os.path.join(self.base_dir, '2023')
            self.dirL = os.path.join(self.base_dir, 'label')

            self.t1_paths, self.fnames = make_dataset(self.dir1, filelist)
            self.t2_paths, _ = make_dataset(self.dir2, filelist)
            self.lab_paths, _ = make_dataset(self.dirL, filelist)

        elif (opt.dataset == 'LEVIR_256_split') or (opt.dataset == 'SYSU_256'):
            self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'A')
            self.t1_paths, self.fnames = make_dataset(self.dir1)

            self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'B')
            self.t2_paths, _ = make_dataset(self.dir2)

            self.dirL = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label')
            self.lab_paths, _ = make_dataset(self.dirL)

        else:
            self.fnames = open(os.path.join(opt.dataroot, opt.dataset, "list", f"{opt.phase}.txt")).read().splitlines()
            self.dir1 = os.path.join(opt.dataroot, opt.dataset, "A")
            self.t1_paths = [os.path.join(self.dir1, n) for n in self.fnames]

            self.dir2 = os.path.join(opt.dataroot, opt.dataset, "B")
            self.t2_paths = [os.path.join(self.dir2, n) for n in self.fnames]

            self.dirL = os.path.join(opt.dataroot, opt.dataset, "label")
            self.lab_paths = [os.path.join(self.dirL, n) for n in self.fnames]

        self.dataset_size = len(self.t1_paths)

        if getattr(self.opt, "phase", "train") == "train":
            self.transform = transforms.Compose([Transforms()]) 
        else:
            self.transform = Identity()

        # ToTensor + ImageNet Normalize
        self.tf_img = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        name = self.fnames[index]

        if self.opt.dataset == 'REDDCD':
            A_img = read_redd_tif_first3_as_uint8(self.t1_paths[index], (self.input_size, self.input_size))
            B_img = read_redd_tif_first3_as_uint8(self.t2_paths[index], (self.input_size, self.input_size))
            L_img = read_label_as_pil(self.lab_paths[index], (self.input_size, self.input_size))
        else:
            # Generic reader for other datasets (assumes PNG/JPEG or uint8 TIF)
            A_img = Image.open(self.t1_paths[index]).convert("RGB")
            B_img = Image.open(self.t2_paths[index]).convert("RGB")
            L_img = Image.open(self.lab_paths[index]).convert("L")
            # Ensure target size
            A_img = A_img.resize((self.input_size, self.input_size), Image.BILINEAR)
            B_img = B_img.resize((self.input_size, self.input_size), Image.BILINEAR)
            L_img = L_img.resize((self.input_size, self.input_size), Image.NEAREST)

        # Apply paired transforms (expects keys: img1, img2, cd_label)
        sample = {'img1': A_img, 'img2': B_img, 'cd_label': L_img}
        sample = self.transform(sample)
        A_img = sample['img1']
        B_img = sample['img2']
        L_img = sample['cd_label']

        # ToTensor + Normalize
        A_t = self.tf_img(A_img)
        B_t = self.tf_img(B_img)

        # Label -> {0,1} long
        L_np = np.array(L_img, dtype=np.uint8)
        L_np = (L_np > 0).astype(np.uint8)
        L_t = torch.from_numpy(L_np).long()

        return {
            'img1': A_t,
            'img2': B_t,
            'cd_label': L_t,
            'name': name,
            'fname': name,
        }



class DataLoader(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=(opt.phase == 'train'),
            pin_memory=True,
            drop_last=(opt.phase == 'train'),
            num_workers=int(opt.num_workers),
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
