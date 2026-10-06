
import os, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from google.colab import drive

drive.mount('/content/drive')


DATA_DIR = '/content/drive/MyDrive/CXR_Project/data/'          # contains images_224.npy, meta.csv, val.csv, test.csv
BEST_CKPT = '/content/drive/MyDrive/CXR_Project/runs/densenet121_baseline/best.pt'   # or the path of best.pt 
OUT_DIR = '/content/preds/'                                    # where the 4 output files are saved

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia',
]
EXPECTED = {'val': (10664, 14), 'test': (25596, 14)}
BATCH_SIZE, NUM_WORKERS = 32, 2
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'
print('Device:', device)

# 1. Images
src = DATA_DIR + 'images_224.npy'
IMAGES_NPY = '/content/images_224.npy'
assert os.path.exists(src), f'Cannot find {src}'
if not (os.path.exists(IMAGES_NPY) and os.path.getsize(IMAGES_NPY) == os.path.getsize(src)):
    print('Copying images_224.npy to the local Colab disk (first time only)...')
    shutil.copyfile(src, IMAGES_NPY)
imgs = np.load(IMAGES_NPY, mmap_mode='r')
SCALE = 255.0 if (imgs.dtype == np.uint8 or float(np.asarray(imgs[:64]).max()) > 1.5) else 1.0
print('Image array:', imgs.shape, imgs.dtype, '| scale divisor:', SCALE)
# 2. CSV files + idx column 
meta = pd.read_csv(DATA_DIR + 'meta.csv')
assert len(meta) == len(imgs), f'meta.csv has {len(meta)} rows, image array has {len(imgs)} images'
pos = {name: i for i, name in enumerate(meta['image'])}

def load_csv(name):
    d = pd.read_csv(DATA_DIR + f'{name}.csv')
    if 'idx' not in d.columns:                      # idx = position of the image in images_224.npy
        d['idx'] = d['image'].map(pos)
        assert d['idx'].notna().all(), f'{name}.csv has images that are not in meta.csv'
        d['idx'] = d['idx'].astype(int)
    assert d['idx'].min() >= 0 and d['idx'].max() < len(imgs), f'idx of {name} is out of range'
    return d

val_df, test_df = load_csv('val'), load_csv('test')
print(f'val: {len(val_df)} images | test: {len(test_df)} images')

#  3. Dataset / loader / preprocessing (identical to training) 
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

class CXRDataset(Dataset):
    def __init__(self, df):
        self.idx = df['idx'].to_numpy(dtype=np.int64)
        self.y = df[DISEASES].to_numpy(dtype=np.float32)
        self.arr = None                              # memory map is opened lazily inside each worker
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(IMAGES_NPY, mmap_mode='r')
        a = np.array(self.arr[self.idx[i]])
        if a.ndim == 2:
            a = a[None]                              # (H,W) -> (1,H,W)
        elif a.shape[-1] in (1, 3):
            a = a.transpose(2, 0, 1)                 # (H,W,C) -> (C,H,W)
        return torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(self.y[i])

def make_loader(df):
    return DataLoader(CXRDataset(df), batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

def prep(x):
    x = x.to(device, non_blocking=True).float() / SCALE
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return (x - MEAN) / STD

@torch.no_grad()
def predict(model, loader):
    model.eval()
    out_l, out_y = [], []
    for x, y in tqdm(loader, leave=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(prep(x))
        out_l.append(logits.float().cpu().numpy())
        out_y.append(y.numpy())
    return np.concatenate(out_l), np.concatenate(out_y)

def macro_auroc(logits, labels):
    probs = 1 / (1 + np.exp(-logits))                # sigmoid: logit -> probability
    aucs = [np.nan if labels[:, c].min() == labels[:, c].max()
            else roc_auc_score(labels[:, c], probs[:, c]) for c in range(labels.shape[1])]
    return float(np.nanmean(aucs)), aucs

# 4. Load best.pt
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))   # no sigmoid: outputs logits
ck = torch.load(BEST_CKPT, map_location=device)
model.load_state_dict(ck['model'])
model.to(device)
print(f"Loaded best.pt: epoch {ck['epoch'] + 1}, val macro AUROC when saved = {ck['val_auroc']:.4f}")

# 5. Predict on val and test
v_logits, v_labels = predict(model, make_loader(val_df))
t_logits, t_labels = predict(model, make_loader(test_df))

#6. Checks before saving
for split, lg, lb, df in [('val', v_logits, v_labels, val_df), ('test', t_logits, t_labels, test_df)]:
    assert lg.shape == lb.shape == EXPECTED[split], f'{split}: {lg.shape} / {lb.shape}, expected {EXPECTED[split]}'
    assert np.isfinite(lg).all(), f'{split}: logits contain NaN or inf'
    assert np.array_equal(lb, df[DISEASES].to_numpy(dtype=np.float32)), f'{split}: labels do not match the CSV row order'
    print(f'{split}: shape {lg.shape} OK')

auc_val, auc_each = macro_auroc(v_logits, v_labels)
print(f"Recomputed val macro AUROC = {auc_val:.4f} (when saved: {ck['val_auroc']:.4f})")
assert abs(auc_val - ck['val_auroc']) < 1e-3, 'Does not match the value saved with best.pt: wrong checkpoint or wrong preprocessing'

#7. Save the 4 files 
for name, arr in [('logits_val', v_logits), ('labels_val', v_labels),
                  ('logits_test', t_logits), ('labels_test', t_labels)]:
    arr = arr.astype(np.float32)
    np.save(OUT_DIR + name + '.npy', arr)
    back = np.load(OUT_DIR + name + '.npy')
    assert back.shape == arr.shape and np.array_equal(back, arr), name
    print(f'{name}.npy: {back.shape}, {back.dtype}')

print('Validation AUROC per finding:')
print(pd.Series(auc_each, index=DISEASES).round(4).to_string())

#8. Probabilities from logits, for threshold selection / calibration
probs_val = 1 / (1 + np.exp(-v_logits))              # (10664, 14), each value in [0, 1]
pred_val = probs_val >= 0.5                          # binary predictions at threshold 0.5
print('Validation probabilities: min %.4f, max %.4f' % (probs_val.min(), probs_val.max()))
