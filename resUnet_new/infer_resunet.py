import os
import argparse
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def list_from_txt(path):
    with open(path, 'r') as f:
        items = [ln.strip().split()[0] for ln in f if ln.strip()]
    return items

def read_tiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return arr, profile, transform, crs

def ensure_chw(arr):
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4, 6, 8, 12):
        arr = np.transpose(arr, (2, 0, 1))
    return arr

def load_pair(root, name, use_rgb_only=False):
    pre, p_prof, p_tr, p_crs = read_tiff(os.path.join(root, '2017', f'{name}'))
    post, _, _, _ = read_tiff(os.path.join(root, '2023', f'{name}'))
    pre = ensure_chw(pre); post = ensure_chw(post)
    if use_rgb_only:
        pre = pre[:3]; post = post[:3]
    x = np.concatenate([pre, post], axis=0).transpose(1, 2, 0).astype(np.float32)
    return x, (p_prof, p_tr, p_crs)

def parse_mean_std_arg(val):
    if not val:
        return None
    val = val.strip()
    if val.endswith(".npy") and os.path.isfile(val):
        return np.load(val).astype(np.float32)
    parts = [p for p in val.split(",") if p != ""]
    return np.array([float(p) for p in parts], dtype=np.float32)

def normalize_per_channel(x, mean, std):
    return (x - mean) / (std + 1e-7)


def percentile_stretch01(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    H, W, C = x.shape
    out = np.empty_like(x, dtype=np.float32)
    eps = 1e-6
    for c in range(C):
        ch = x[..., c]
        vmin, vmax = np.percentile(ch, [p_low, p_high])
        if not np.isfinite(vmin): vmin = np.nanmin(ch)
        if not np.isfinite(vmax): vmax = np.nanmax(ch)
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = vmin + 1.0
        if vmax <= vmin + 1e-12:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((ch - vmin) / (vmax - vmin + eps), 0.0, 1.0)
    return out

def sliding_window_predict(model, x, patch=256, overlap=0.5, batch_size=4):
    assert 0 <= overlap < 1.0
    H, W, C = x.shape
    ov = int(round(patch * overlap))
    ov -= ov % 2
    stride = patch - ov
    if stride <= 0:
        raise ValueError("overlap too big, stride <= 0")

    step_h = (stride - H % stride) % stride
    step_w = (stride - W % stride) % stride
    pad = ((ov // 2, ov // 2 + step_h), (ov // 2, ov // 2 + step_w), (0, 0))
    x_pad = np.pad(x, pad, mode='symmetric')

    k1 = (H + step_h) // stride
    k2 = (W + step_w) // stride

    patches, coords = [], []
    for i in range(k1):
        for j in range(k2):
            xs = x_pad[i * stride:i * stride + patch, j * stride:j * stride + patch, :]
            patches.append(xs)
            coords.append((i, j))
    patches = np.stack(patches, axis=0).astype(np.float32)

    pr0 = model.predict(patches[:1], verbose=0)
    if pr0.ndim != 4 or pr0.shape[-1] != 2:
        raise ValueError(f"not 2（softmax）: {pr0.shape}")

    probs = np.zeros((k1 * stride, k2 * stride, 2), dtype=np.float32)
    for s in range(0, len(patches), batch_size):
        batch = patches[s:s + batch_size]
        pr = model.predict(batch, verbose=0)
        for b, (i, j) in enumerate(coords[s:s + batch_size]):
            prb = pr[b]
            core = prb[ov // 2:ov // 2 + stride, ov // 2:ov // 2 + stride, :]
            probs[i * stride:(i * stride + stride), j * stride:(j * stride + stride), :] = core

    probs = probs[:k1 * stride - step_h, :k2 * stride - step_w, :]
    return probs

def save_geotiff_like(reference_profile, transform, crs, array, out_path, dtype, nodata=None):
    prof = reference_profile.copy()
    prof.update({
        'driver': 'GTiff',
        'height': array.shape[0],
        'width': array.shape[1],
        'count': 1,
        'dtype': dtype,
        'transform': transform,
        'crs': crs,
        'compress': 'deflate'
    })
    if nodata is not None:
        prof['nodata'] = nodata
    with rasterio.open(out_path, 'w', **prof) as dst:
        if dtype == 'uint8':
            dst.write(array.astype(np.uint8), 1)
        elif dtype == 'float32':
            dst.write(array.astype(np.float32), 1)
        else:
            dst.write(array, 1)

def save_png(mask, out_path):
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

def per_class_confusion(y_true, y_pred):
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    from sklearn.metrics import confusion_matrix, cohen_kappa_score
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    tn, fp, fn, tp = per_class_confusion(y_true, y_pred)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec1 = tp / max(1, tp + fp); rec1 = tp / max(1, tp + fn)
    f11 = 2 * prec1 * rec1 / max(1e-7, (prec1 + rec1)); iou1 = tp / max(1, tp + fp + fn)
    prec0 = tn / max(1, tn + fn); rec0 = tn / max(1, tn + fp)
    f10 = 2 * prec0 * rec0 / max(1e-7, (prec0 + rec0)); iou0 = tn / max(1, tn + fp + fn)
    kappa = cohen_kappa_score(y_true.reshape(-1), y_pred.reshape(-1), labels=[0, 1])
    return {'acc': acc, 'kappa': kappa,
            'f1_mean': (f10 + f11) / 2, 'f1_0': f10, 'f1_1': f11,
            'prec_mean': (prec0 + prec1) / 2, 'prec_0': prec0, 'prec_1': prec1,
            'rec_mean': (rec0 + rec1) / 2, 'rec_0': rec0, 'rec_1': rec1,
            'iou_mean': (iou0 + iou1) / 2, 'iou_0': iou0, 'iou_1': iou1}

def str2bool(v): return str(v).lower() in ['1','true','yes','y']

def auto_get_mean_std(args, model_path, ids, data_root, use_rgb_only, patch_size):
    mean = parse_mean_std_arg(args.mean)
    std  = parse_mean_std_arg(args.std)
    if mean is not None and std is not None:
        print(f"[Norm] Using mean/std from args:\n  mean={args.mean}\n  std={args.std}")
        return mean.astype(np.float32), std.astype(np.float32)

    model_dir = os.path.dirname(os.path.abspath(model_path))
    mpath = os.path.join(model_dir, "train_mean.npy")
    spath = os.path.join(model_dir, "train_std.npy")
    if os.path.isfile(mpath) and os.path.isfile(spath):
        mean = np.load(mpath).astype(np.float32)
        std  = np.load(spath).astype(np.float32)
        print(f"[Norm] Loaded mean/std from model dir:\n  {mpath}\n  {spath}")
        return mean, std

    print("[Warn] mean/std not found")
    sample = min(50, len(ids))
    xs = []
    for name in ids[:sample]:
        x, _ = load_pair(data_root, name, use_rgb_only=use_rgb_only)
        x = percentile_stretch01(x)
        xs.append(x)
    X = np.stack(xs, axis=0)
    mean = X.mean(axis=(0, 1, 2)).astype(np.float32)
    std = (X.std(axis=(0, 1, 2)) + 1e-7).astype(np.float32)
    return mean, std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    ap.add_argument('--use_rgb_only', type=str2bool, default=True)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--overlap', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--mean', type=str, default='', help='comma-separated or .npy path')
    ap.add_argument('--std',  type=str, default='', help='comma-separated or .npy path')
    ap.add_argument('--out_dir', type=str, default='./infer_out')
    ap.add_argument('--save_tif', type=str2bool, default=True)
    ap.add_argument('--save_png', type=str2bool, default=True)
    ap.add_argument('--no_save', type=str2bool, default=False)
    args = ap.parse_args()

    if args.no_save:
        args.save_tif = False
        args.save_png = False

    ids = list_from_txt(os.path.join(args.data_root, 'splits', f'{args.split}.txt'))
    if len(ids) == 0:
        raise FileNotFoundError(f"No ids found in splits/{args.split}.txt")

    if (args.save_tif or args.save_png):
        os.makedirs(args.out_dir, exist_ok=True)


    model = load_model(args.model_path, compile=False)

    mean, std = auto_get_mean_std(args, args.model_path, ids, args.data_root,
                                  args.use_rgb_only, args.patch_size)

    exp_C = 6 if args.use_rgb_only else 8
    if mean.shape[0] != exp_C or std.shape[0] != exp_C:
        print(f"[Warn] mean/std shape:({mean.shape[0]}")
        if mean.shape[0] >= exp_C and std.shape[0] >= exp_C:
            mean = mean[:exp_C]; std = std[:exp_C]
        else:
            mean = np.resize(mean, (exp_C,)).astype(np.float32)
            std  = np.resize(std,  (exp_C,)).astype(np.float32)

    metrics_list = []
    for name in tqdm(ids, desc=f'Infer {args.split}'):
        x, (prof, tr, crs) = load_pair(args.data_root, name, use_rgb_only=args.use_rgb_only)
        x = percentile_stretch01(x).astype(np.float32)
        x = normalize_per_channel(x, mean, std)

        probs = sliding_window_predict(
            model, x,
            patch=args.patch_size, overlap=args.overlap,
            batch_size=args.batch_size
        )  # (H,W,2)

        prob1 = probs[..., 1]
        pred = (prob1 >= 0.5).astype(np.uint8)

        if args.save_tif or args.save_png:
            out_base = os.path.join(args.out_dir, name)
            if args.save_tif:
                save_geotiff_like(prof, tr, crs, pred, out_base + '_mask.tif', 'uint8', nodata=0)
                save_geotiff_like(prof, tr, crs, prob1.astype(np.float32), out_base + '_prob.tif', 'float32')
            if args.save_png:
                os.makedirs(os.path.dirname(out_base), exist_ok=True)
                save_png(pred, out_base + '_mask.png')

        lab_path = os.path.join(args.data_root, 'label', f'{name}')
        if os.path.exists(lab_path):
            with rasterio.open(lab_path) as src:
                y_true = src.read(1).astype(np.uint8)
            if y_true.max() > 1:
                y_true = (y_true > 0).astype(np.uint8)
            m = compute_metrics(y_true, pred)
            metrics_list.append((name, m))

    if metrics_list:
        import csv
        if args.save_tif or args.save_png:
            csv_path = os.path.join(args.out_dir, f'summary_{args.split}.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                header = ['name','acc','kappa','f1_mean','f1_0','f1_1',
                          'prec_mean','prec_0','prec_1','rec_mean','rec_0','rec_1',
                          'iou_mean','iou_0','iou_1']
                w.writerow(header)
                for name, m in metrics_list:
                    w.writerow([name, m['acc'], m['kappa'], m['f1_mean'], m['f1_0'], m['f1_1'],
                                m['prec_mean'], m['prec_0'], m['prec_1'], m['rec_mean'], m['rec_0'], m['rec_1'],
                                m['iou_mean'], m['iou_0'], m['iou_1']])

        arr = np.array([
            [m['acc'], m['kappa'],
             m['f1_mean'], m['f1_0'], m['f1_1'],
             m['prec_mean'], m['prec_0'], m['prec_1'],
             m['rec_mean'], m['rec_0'], m['rec_1'],
             m['iou_mean'], m['iou_0'], m['iou_1']]
            for _, m in metrics_list
        ], dtype=np.float64)

        acc, kappa, f1m, f10, f11, precm, prec0, prec1, recm, rec0, rec1, ioum, iou0, iou1 = arr.mean(axis=0)
        print('== Mean over {} images ({}) =='.format(len(metrics_list), args.split))
        print(
            'acc={:.4f} kappa={:.4f} '
            'F1(mean/0/1)={:.4f}/{:.4f}/{:.4f} '
            'Rec(mean/0/1)={:.4f}/{:.4f}/{:.4f} '
            'Prec(mean/0/1)={:.4f}/{:.4f}/{:.4f} '
            'IoU(mean/0/1)={:.4f}/{:.4f}/{:.4f}'.format(
                acc, kappa, f1m, f10, f11, recm, rec0, rec1, precm, prec0, prec1, ioum, iou0, iou1
            )
        )
    else:
        print("No labels found; metrics were not computed.")

if __name__ == '__main__':
    main()