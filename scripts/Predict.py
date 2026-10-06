"""

Outputs in --out_dir:
    logits_val.npy   (10664, 14) float32   logits BEFORE the sigmoid
    logits_test.npy  (25596, 14) float32
    labels_val.npy   (10664, 14) float32   0/1
    labels_test.npy  (25596, 14) float32
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia',
]
EXPECTED = {'val': (10664, 14), 'test': (25596, 14)}
BATCH_SIZE = 32
NUM_WORKERS = 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                    help='folder containing images_224.npy, meta.csv, val.csv, test.csv')
    ap.add_argument('--ckpt', default='best.pt', help='path to best.pt')
    ap.add_argument('--out_dir', default='preds')
    ap.add_argument('--copy_to_local', action='store_true',
                    help='copy images_224.npy to the local Colab disk first (much faster than reading from Drive)')
    return ap.parse_args()


class CXRDataset(Dataset):
    """Reads images by the idx column of each CSV row; sample order = CSV row order."""

    def __init__(self, df, images_npy):
        self.images_npy = images_npy
        self.idx = df['idx'].to_numpy(dtype=np.int64)
        self.y = df[DISEASES].to_numpy(dtype=np.float32)
        self.arr = None  # memory map is opened lazily inside each worker

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.images_npy, mmap_mode='r')
        a = np.array(self.arr[self.idx[i]])
        if a.ndim == 2:                      # (H, W) -> (1, H, W)
            a = a[None]
        elif a.shape[-1] in (1, 3):          # (H, W, C) -> (C, H, W)
            a = a.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(self.y[i])


def add_idx(df, meta, n_imgs, name):
    """The CSV files from the data-split block have no idx column: idx = position of the image
    in images_224.npy (the npy stores images in the same row order as meta.csv)."""
    if 'idx' in df.columns:
        return df
    pos = {img: i for i, img in enumerate(meta['image'])}
    df = df.copy()
    df['idx'] = df['image'].map(pos)
    assert df['idx'].notna().all(), f'{name}.csv has images that are not in meta.csv'
    df['idx'] = df['idx'].astype(int)
    assert df['idx'].min() >= 0 and df['idx'].max() < n_imgs, f'idx of {name} is out of range'
    return df


@torch.no_grad()
def predict(model, loader, prep, device, use_amp):
    model.eval()
    out_l, out_y = [], []
    for x, y in tqdm(loader, leave=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(prep(x))
        out_l.append(logits.float().cpu().numpy())
        out_y.append(y.numpy())
    return np.concatenate(out_l), np.concatenate(out_y)


def macro_auroc(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    aucs = []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max():   # class has a single value: AUROC undefined
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
    return float(np.nanmean(aucs)), aucs


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print('Device:', device)

    # --- Images ---
    images_npy = os.path.join(args.data_dir, 'images_224.npy')
    assert os.path.exists(images_npy), f'Cannot find {images_npy}'
    if args.copy_to_local:
        local = '/content/images_224.npy'
        if not (os.path.exists(local) and os.path.getsize(local) == os.path.getsize(images_npy)):
            print('Copying images_224.npy to the local disk (first time only)...')
            shutil.copyfile(images_npy, local)
        images_npy = local
    imgs = np.load(images_npy, mmap_mode='r')
    scale = 255.0 if (imgs.dtype == np.uint8 or float(np.asarray(imgs[:64]).max()) > 1.5) else 1.0
    print('Image array:', imgs.shape, imgs.dtype, '| scale divisor:', scale)

    # --- CSV files ---
    meta = pd.read_csv(os.path.join(args.data_dir, 'meta.csv'))
    assert len(meta) == len(imgs), f'meta.csv has {len(meta)} rows, image array has {len(imgs)} images'
    val_df = add_idx(pd.read_csv(os.path.join(args.data_dir, 'val.csv')), meta, len(imgs), 'val')
    test_df = add_idx(pd.read_csv(os.path.join(args.data_dir, 'test.csv')), meta, len(imgs), 'test')
    print(f'val: {len(val_df)} images | test: {len(test_df)} images')

    # --- Preprocessing (identical to training) ---
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def prep(x):
        x = x.to(device, non_blocking=True).float() / scale
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - mean) / std

    def make_loader(df):
        return DataLoader(CXRDataset(df, images_npy), batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model: load best.pt ---
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['model'])
    model.to(device)
    print(f"Loaded {args.ckpt}: epoch {ck['epoch'] + 1}, val macro AUROC when saved = {ck['val_auroc']:.4f}")

    # --- Predict ---
    v_logits, v_labels = predict(model, make_loader(val_df), prep, device, use_amp)
    t_logits, t_labels = predict(model, make_loader(test_df), prep, device, use_amp)

    # --- Checks ---
    for split, lg, lb, df in [('val', v_logits, v_labels, val_df), ('test', t_logits, t_labels, test_df)]:
        assert lg.shape == lb.shape == EXPECTED[split], f'{split}: {lg.shape} / {lb.shape}, expected {EXPECTED[split]}'
        assert np.isfinite(lg).all(), f'{split}: logits contain NaN or inf'
        assert np.array_equal(lb, df[DISEASES].to_numpy(dtype=np.float32)), f'{split}: labels do not match the CSV row order'
        print(f'{split}: shape {lg.shape} OK')

    auc_val, _ = macro_auroc(v_logits, v_labels)
    print(f"Recomputed val macro AUROC = {auc_val:.4f} (when saved: {ck['val_auroc']:.4f})")
    assert abs(auc_val - ck['val_auroc']) < 1e-3, 'Does not match the value saved with best.pt: wrong checkpoint or wrong preprocessing'

    # --- Save the 4 files ---
    os.makedirs(args.out_dir, exist_ok=True)
    for name, arr in [('logits_val', v_logits), ('labels_val', v_labels),
                      ('logits_test', t_logits), ('labels_test', t_labels)]:
        path = os.path.join(args.out_dir, name + '.npy')
        np.save(path, arr.astype(np.float32))
        back = np.load(path)
        assert back.shape == arr.shape and np.array_equal(back, arr.astype(np.float32)), name
        print(f'Saved {path}: {back.shape}, {back.dtype}')


if __name__ == '__main__':
    main()
