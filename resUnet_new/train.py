# train.py
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import rasterio

# ===== Model builders =====
try:
    from resunet import Config as ForestCfg, build_res_unet, build_resunet_forest
    HAS_FOREST = True
except Exception:
    HAS_FOREST = False

# ---- mixed precision ----
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ---- GPU memory growth ----
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[Warn] set_memory_growth failed:", e)


def list_from_txt(path):
    with open(path, 'r') as f:
        items = [ln.strip().split()[0] for ln in f if ln.strip()]
    return items

def read_tiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
    return arr

def _ensure_chw(arr):
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4, 6, 8, 12):
        arr = np.transpose(arr, (2, 0, 1))
    return arr

def _resize_chw(arr, size_hw, interp=cv2.INTER_LINEAR):
    H, W = size_hw
    c, h, w = arr.shape
    if (h, w) == (H, W):
        return arr
    out = [cv2.resize(arr[i], (W, H), interpolation=interp) for i in range(c)]
    return np.stack(out, axis=0)

def load_pair_and_label(root, name, use_rgb_only=False, patch_size=256, strict=False, siamese_input=False):
    pre  = read_tiff(os.path.join(root, '2017', f'{name}'))
    post = read_tiff(os.path.join(root, '2023', f'{name}'))
    lab_path = os.path.join(root, 'label', f'{name}')
    with rasterio.open(lab_path) as src:
        label = src.read(1)

    pre  = _ensure_chw(pre)
    post = _ensure_chw(post)

    if use_rgb_only:
        pre  = pre[:3, ...]
        post = post[:3, ...]

    hp, wp = pre.shape[1], pre.shape[2]
    hq, wq = post.shape[1], post.shape[2]
    hl, wl = label.shape

    if strict and not (hp == hq == hl and wp == wq == wl):
        raise ValueError(f"Shape mismatch {name}: pre={pre.shape}, post={post.shape}, lab={label.shape}")

    if patch_size is None:
        H = min(hp, hq, hl); W = min(wp, wq, wl)
        pre  = pre[:, :H, :W]
        post = post[:, :H, :W]
        label = label[:H, :W]
    else:
        pre  = _resize_chw(pre,  (patch_size, patch_size), interp=cv2.INTER_LINEAR)
        post = _resize_chw(post, (patch_size, patch_size), interp=cv2.INTER_LINEAR)
        if (hl, wl) != (patch_size, patch_size):
            label = cv2.resize(label, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    if siamese_input:
        pre = pre.transpose(1, 2, 0).astype(np.float32)
        post = post.transpose(1, 2, 0).astype(np.float32)
        x = [pre, post]
    else:
        x = np.concatenate([pre, post], axis=0).transpose(1, 2, 0).astype(np.float32)

    if label.max() > 1:
        label = (label > 0).astype(np.uint8)
    y = label.astype(np.uint8)
    return x, y


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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def make_fixed_imagenet_stats(in_ch: int, use_rgb_only: bool, siamese_input=False):
    if siamese_input:
        mean = np.zeros((3,), dtype=np.float32)
        std  = np.ones((3,), dtype=np.float32)
        mean[0:3] = IMAGENET_MEAN; std[0:3] = IMAGENET_STD
        return mean, std
    else:
        mean = np.zeros((in_ch,), dtype=np.float32)
        std  = np.ones((in_ch,), dtype=np.float32)
        if use_rgb_only:
            assert in_ch >= 6, f"Expect >=6 channels for RGBx2, got {in_ch}"
            mean[0:3] = IMAGENET_MEAN; std[0:3] = IMAGENET_STD
            mean[3:6] = IMAGENET_MEAN; std[3:6] = IMAGENET_STD
        else:
            half = in_ch // 2
            if half >= 3:
                mean[0:3] = IMAGENET_MEAN; std[0:3] = IMAGENET_STD
                mean[half:half+3] = IMAGENET_MEAN; std[half:half+3] = IMAGENET_STD
            else:
                if in_ch >= 3:
                    mean[0:3] = IMAGENET_MEAN; std[0:3] = IMAGENET_STD
                if in_ch >= 6:
                    mean[3:6] = IMAGENET_MEAN; std[3:6] = IMAGENET_STD
        return mean, std

class OnlineStandardizer:
    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)
    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-7)


def _swap_pre_post(x):
    if isinstance(x, list):
        return [x[1], x[0]]
    else:
        C2 = x.shape[-1]
        C = C2 // 2
        return np.concatenate([x[..., C:], x[..., :C]], axis=-1)

def _hflip(x):
    if isinstance(x, list):
        return [np.fliplr(x[0]), np.fliplr(x[1])]
    else:
        return np.fliplr(x)

def _vflip(x):
    if isinstance(x, list):
        return [np.flipud(x[0]), np.flipud(x[1])]
    else:
        return np.flipud(x)

def _rot90k(x, k):
    if isinstance(x, list):
        return [np.rot90(x[0], k), np.rot90(x[1], k)]
    else:
        return np.rot90(x, k)

def _sample_resized_crop_params(H, W, scale=(0.333, 1.0), ratio=(0.75, 1.333), trials=10):
    area = H * W
    for _ in range(trials):
        target_area = np.random.uniform(*scale) * area
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect = np.exp(np.random.uniform(*log_ratio))
        h = int(round(np.sqrt(target_area / aspect)))
        w = int(round(np.sqrt(target_area * aspect)))
        if 0 < h <= H and 0 < w <= W:
            i = np.random.randint(0, H - h + 1)
            j = np.random.randint(0, W - w + 1)
            return i, j, h, w
    in_short = min(H, W)
    h = w = in_short
    i = (H - h) // 2
    j = (W - w) // 2
    return i, j, h, w

def _resized_crop_pair(x, y, size_hw=(256, 256)):
    if isinstance(x, list):
        H, W = x[0].shape[:2]
        i, j, h, w = _sample_resized_crop_params(H, W, scale=(0.333, 1.0), ratio=(0.75, 1.333))
        x0_crop = x[0][i:i+h, j:j+w, :]
        x1_crop = x[1][i:i+h, j:j+w, :]
        y_crop  = y[i:i+h, j:j+w]

        tgtH, tgtW = size_hw
        C = x[0].shape[-1]

        x0_res = np.stack(
            [cv2.resize(x0_crop[..., c], (tgtW, tgtH), interpolation=cv2.INTER_LINEAR) for c in range(C)],
            axis=-1
        )
        x1_res = np.stack(
            [cv2.resize(x1_crop[..., c], (tgtW, tgtH), interpolation=cv2.INTER_LINEAR) for c in range(C)],
            axis=-1
        )
        y_res = cv2.resize(y_crop, (tgtW, tgtH), interpolation=cv2.INTER_NEAREST)
        return [x0_res, x1_res], y_res
    else:
        H, W = x.shape[:2]
        i, j, h, w = _sample_resized_crop_params(H, W, scale=(0.333, 1.0), ratio=(0.75, 1.333))
        x_crop = x[i:i+h, j:j+w, :]
        y_crop = y[i:i+h, j:j+w]
        tgtH, tgtW = size_hw
        C = x.shape[-1]
        x_res = np.stack(
            [cv2.resize(x_crop[..., c], (tgtW, tgtH), interpolation=cv2.INTER_LINEAR) for c in range(C)],
            axis=-1
        )
        y_res = cv2.resize(y_crop, (tgtW, tgtH), interpolation=cv2.INTER_NEAREST)
        return x_res, y_res

# ----------------------------
# Data Generator
# ----------------------------
class PatchDataset(Sequence):
    def __init__(self, root, names, batch_size=8, patch_size=256,
                 shuffle=True, augment=False, normalizer=None, use_rgb_only=False,
                 p_low=1.0, p_high=99.0, siamese_input=False):
        self.root = root
        self.names = names
        self.batch_size = batch_size
        self.patch = patch_size
        self.shuffle = shuffle
        self.augment = augment
        self.normalizer = normalizer
        self.use_rgb_only = use_rgb_only
        self.p_low = p_low
        self.p_high = p_high
        self.siamese_input = siamese_input
        self.indexes = np.arange(len(self.names))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.names) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.siamese_input:
            Xs_t1, Xs_t2, Ys = [], [], []
        else:
            Xs, Ys = [], []

        for bi in batch_idx:
            name = self.names[bi]
            read_patch = None if self.augment else self.patch
            x, y = load_pair_and_label(self.root, name, use_rgb_only=self.use_rgb_only,
                                       patch_size=read_patch, siamese_input=self.siamese_input)

            if self.augment:
                if np.random.rand() < 0.5:
                    x = _hflip(x)
                    y = np.fliplr(y)
                if np.random.rand() < 0.5:
                    x = _vflip(x)
                    y = np.flipud(y)
                if np.random.rand() < 0.5:
                    k = np.random.choice([1, 2, 3])
                    x = _rot90k(x, k)
                    y = np.rot90(y, k)
                if read_patch is None:
                    x, y = _resized_crop_pair(x, y, size_hw=(self.patch, self.patch))

            if self.siamese_input:
                x_t1, x_t2 = x
                x_t1 = percentile_stretch01(x_t1, p_low=self.p_low, p_high=self.p_high).astype(np.float32)
                x_t2 = percentile_stretch01(x_t2, p_low=self.p_low, p_high=self.p_high).astype(np.float32)
                if self.normalizer is not None:
                    x_t1 = self.normalizer.transform(x_t1)
                    x_t2 = self.normalizer.transform(x_t2)
                Xs_t1.append(x_t1.astype(np.float32))
                Xs_t2.append(x_t2.astype(np.float32))
            else:
                x = percentile_stretch01(x, p_low=self.p_low, p_high=self.p_high).astype(np.float32)
                if self.normalizer is not None:
                    x = self.normalizer.transform(x)
                Xs.append(x.astype(np.float32))

            y_oh = tf.one_hot(y, 2, dtype=tf.float32).numpy()
            Ys.append(y_oh.astype(np.float32))

        if self.siamese_input:
            X_t1 = np.stack(Xs_t1, axis=0)
            X_t2 = np.stack(Xs_t2, axis=0)
            Y = np.stack(Ys, axis=0)
            return [X_t1, X_t2], Y
        else:
            X = np.stack(Xs, axis=0)
            Y = np.stack(Ys, axis=0)
            return X, Y

# ----------------------------
# Losses / Metrics / Logger
# ----------------------------
def weighted_focal_loss(alpha=(0.25, 0.75), gamma=2.0):
    a0, a1 = alpha
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = K.sum(y_true * y_pred, axis=-1)
        alpha_map = y_true[..., 0] * a0 + y_true[..., 1] * a1
        focal = -alpha_map * K.pow(1.0 - pt, gamma) * K.log(pt)
        return K.mean(focal)
    return loss

def dice_loss(smooth=1.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_t = y_true[..., 1]; y_p = y_pred[..., 1]
        inter = K.sum(y_t * y_p, axis=(1, 2))
        denom = K.sum(y_t + y_p, axis=(1, 2))
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return 1.0 - K.mean(dice)
    return loss

def combined_loss(alpha=(0.25, 0.75), gamma=2.0, lambda_focal=1.0, lambda_dice=1.0):
    fl = weighted_focal_loss(alpha=alpha, gamma=gamma)
    dl = dice_loss()
    def loss(y_true, y_pred):
        return lambda_focal * fl(y_true, y_pred) + lambda_dice * dl(y_true, y_pred)
    loss.focal = fl; loss.dice = dl
    return loss

def per_class_confusion(y_true, y_pred):
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def compute_all_metrics(y_true_oh, y_pred_prob):
    y_true = np.argmax(y_true_oh, axis=-1)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    tn, fp, fn, tp = per_class_confusion(y_true, y_pred)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec1 = tp / max(1, tp + fp); rec1 = tp / max(1, tp + fn)
    f11 = 2 * prec1 * rec1 / max(1e-7, (prec1 + rec1)); iou1 = tp / max(1, tp + fp + fn)
    prec0 = tn / max(1, tn + fn); rec0 = tn / max(1, tn + fp)
    f10 = 2 * prec0 * rec0 / max(1e-7, (prec0 + rec0)); iou0 = tn / max(1, tn + fp + fn)
    kappa = cohen_kappa_score(y_true.reshape(-1), y_pred.reshape(-1), labels=[0, 1])
    return {'acc': acc,
            'f1_mean': (f10 + f11) / 2, 'f1_0': f10, 'f1_1': f11,
            'prec_mean': (prec0 + prec1) / 2, 'prec_0': prec0, 'prec_1': prec1,
            'rec_mean': (rec0 + rec1) / 2, 'rec_0': rec0, 'rec_1': rec1,
            'iou_mean': (iou0 + iou1) / 2, 'iou_0': iou0, 'iou_1': iou1,
            'kappa': kappa}

class MetricsLogger(Callback):
    def __init__(self, val_data, loss_components, siamese_input=False):
        super().__init__()
        self.val_data = val_data
        self.fl = loss_components['focal']
        self.dl = loss_components['dice']
        self.siamese_input = siamese_input

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true_all = []
        y_pred_all = []
        focal_vals = []
        dice_vals = []
        for xb, yb in self.val_data:
            if self.siamese_input and not isinstance(xb, list):
                xb = [xb[..., :3], xb[..., 3:]]
            yp = self.model.predict(xb, verbose=0)
            focal_vals.append(K.get_value(self.fl(yb, yp)))
            dice_vals.append(K.get_value(self.dl(yb, yp)))
            y_true_all.append(yb)
            y_pred_all.append(yp)

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        met = compute_all_metrics(y_true_all, y_pred_all)

        logs['val_f1_1']    = float(met['f1_1'])
        logs['val_f1_0']    = float(met['f1_0'])
        logs['val_f1_mean'] = float(met['f1_mean'])
        logs['val_iou_1']   = float(met['iou_1'])
        logs['val_iou_0']   = float(met['iou_0'])
        logs['val_iou_mean']= float(met['iou_mean'])
        logs['val_acc']     = float(met['acc'])
        logs['val_kappa']   = float(met['kappa'])
        logs['val_focal']   = float(np.mean(focal_vals))
        logs['val_dice']    = float(np.mean(dice_vals))

        print(
            f"[Val@{epoch:03d}] "
            f"Acc={met['acc']*100:.2f}%  Kappa={met['kappa']:.4f}  "
            f"F1(m/0/1)={met['f1_mean']:.4f}/{met['f1_0']:.4f}/{met['f1_1']:.4f}  "
            f"IoU(m/0/1)={met['iou_mean']:.4f}/{met['iou_0']:.4f}/{met['iou_1']:.4f}  "
            f"FL={logs['val_focal']:.4f}  DL={logs['val_dice']:.4f}"
        )

def evaluate_on_dataset(model, ds, desc="Eval"):
    y_true_all, y_pred_all = [], []
    for xb, yb in ds:
        yp = model.predict(xb, verbose=0)
        y_true_all.append(yb)
        y_pred_all.append(yp)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    met = compute_all_metrics(y_true_all, y_pred_all)
    print(
        f"[{desc}] "
        f"Acc={met['acc']*100:.2f}%  Kappa={met['kappa']:.4f}  "
        f"F1(mean/0/1)={met['f1_mean']:.4f}/{met['f1_0']:.4f}/{met['f1_1']:.4f}  "
        f"IoU(mean/0/1)={met['iou_mean']:.4f}/{met['iou_0']:.4f}/{met['iou_1']:.4f}  "
        f"Prec(mean/0/1)={met['prec_mean']:.4f}/{met['prec_0']:.4f}/{met['prec_1']:.4f}  "
        f"Rec(mean/0/1)={met['rec_mean']:.4f}/{met['rec_0']:.4f}/{met['rec_1']:.4f}"
    )
    return met

# ----------------------------
# Main
# ----------------------------
def str2bool(v): return str(v).lower() in ['1', 'true', 'yes', 'y']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./runs_resunet_redd')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gamma', type=float, default=4.0)
    ap.add_argument('--class_weight_0', type=float, default=0.4)
    ap.add_argument('--class_weight_1', type=float, default=2.0)
    ap.add_argument('--lambda_focal', type=float, default=1.0)
    ap.add_argument('--lambda_dice', type=float, default=1.0)
    ap.add_argument('--use_rgb_only', type=str2bool, default=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=10)

    ap.add_argument('--model_type', type=str, default='concat_siamese',
                    choices=['original', 'concat_siamese'])

    ap.add_argument('--model', type=str, default='resunet',
                    choices=['resunet', 'resunet_forest'])
    ap.add_argument('--depth', type=int, default=4, choices=[3, 4])
    ap.add_argument('--stem_filters', type=int, default=64)
    ap.add_argument('--up_mode', type=str, default='nearest', choices=['nearest', 'transposed'])
    ap.add_argument('--preact', type=str2bool, default=True)
    ap.add_argument('--se_ratio', type=int, default=0)
    ap.add_argument('--dropout_rate', type=float, default=0.0)
    ap.add_argument('--act', type=str, default='gelu', choices=['relu', 'gelu'])

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tf.random.set_seed(args.seed); np.random.seed(args.seed)

    train_ids = list_from_txt(os.path.join(args.data_root, 'splits', 'train.txt'))
    val_ids   = list_from_txt(os.path.join(args.data_root, 'splits', 'val.txt'))
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise FileNotFoundError("Empty train/val split. Check splits/train.txt and splits/val.txt")

    siamese_input = (args.model_type == 'concat_siamese')
    if siamese_input:
        in_ch = 3
    else:
        in_ch = 6 if args.use_rgb_only else 8

    mean_im, std_im = make_fixed_imagenet_stats(in_ch, args.use_rgb_only, siamese_input=siamese_input)
    mean_path = os.path.join(args.save_dir, 'train_mean.npy')
    std_path  = os.path.join(args.save_dir, 'train_std.npy')
    np.save(mean_path, mean_im); np.save(std_path, std_im)
    print(f"[Norm] Percentile stretch to [0,1] + ImageNet RGB normalization. Saved mean/std to {mean_path}, {std_path}")

    normalizer = OnlineStandardizer(mean=mean_im, std=std_im)

    train_ds = PatchDataset(args.data_root, train_ids,
                            batch_size=args.batch_size, patch_size=args.patch_size,
                            shuffle=True, augment=True,
                            normalizer=normalizer, use_rgb_only=args.use_rgb_only,
                            siamese_input=siamese_input)
    val_ds = PatchDataset(args.data_root, val_ids,
                          batch_size=args.batch_size, patch_size=args.patch_size,
                          shuffle=False, augment=False,
                          normalizer=normalizer, use_rgb_only=args.use_rgb_only,
                          siamese_input=siamese_input)

    if siamese_input:
        input_shape = (args.patch_size, args.patch_size, 3)
    else:
        input_shape = (args.patch_size, args.patch_size, in_ch)

    if siamese_input:
        model = build_res_unet(
            input_shape=input_shape, nClasses=2,
            stem_filters=args.stem_filters,
            depth=args.depth, up_mode=args.up_mode,
            preact=args.preact, se_ratio=args.se_ratio,
            dropout_rate=args.dropout_rate, act=args.act
        )
        model_name = 'concat_siamese'
    else:
        if args.model == 'resunet_forest' and HAS_FOREST:
            cfg = ForestCfg()
            cfg.input_shape = input_shape
            cfg.num_classes = 2
            cfg.stem_filters = args.stem_filters
            if args.depth == 4:
                cfg.encoder_filters = [64, 128, 256, 512]; cfg.decoder_filters = [256, 128, 64]
            else:
                cfg.encoder_filters = [64, 128, 256]; cfg.decoder_filters = [128, 64]
            cfg.up_mode = args.up_mode; cfg.preact = args.preact
            cfg.se_ratio = args.se_ratio; cfg.dropout_rate = args.dropout_rate
            cfg.act = args.act; cfg.final_activation = "softmax"
            model = build_resunet_forest(cfg)
            model_name = 'resunet_forest'
        else:
            try:
                model = build_res_unet(input_shape=input_shape, nClasses=2,
                                       stem_filters=args.stem_filters,
                                       depth=args.depth, up_mode=args.up_mode,
                                       preact=args.preact, se_ratio=args.se_ratio,
                                       dropout_rate=args.dropout_rate, act=args.act)
            except TypeError:
                H, W, C = input_shape
                model = build_res_unet(nClasses=2, input_height=H, input_width=W, nChannels=C)
            model_name = 'resunet'

    alpha_pair = (args.class_weight_0, args.class_weight_1)
    loss_fn = combined_loss(alpha=alpha_pair, gamma=args.gamma,
                            lambda_focal=args.lambda_focal, lambda_dice=args.lambda_dice)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=loss_fn, metrics=['accuracy'])

    ckpt_path = os.path.join(args.save_dir, f'best_{model_name}.h5')

    logger = MetricsLogger(val_data=val_ds,
                           loss_components={'focal': loss_fn.focal, 'dice': loss_fn.dice},
                           siamese_input=siamese_input)
    ckpt  = ModelCheckpoint(
        ckpt_path, monitor='val_f1_1', mode='max',
        save_best_only=True, save_weights_only=False, verbose=1
    )
    rlrop = ReduceLROnPlateau(
        monitor='val_f1_1', mode='max',
        factor=0.5, patience=max(3, args.patience // 4), min_lr=1e-6, verbose=1
    )
    estop = EarlyStopping(
        monitor='val_f1_1', mode='max',
        patience=args.patience, restore_best_weights=True, verbose=1
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[logger, ckpt, rlrop, estop],
        verbose=1
    )

    print("Training done. Best (by val_f1_1) model saved to:", ckpt_path)
    print("Mean/std saved:", mean_path, std_path)

    try:
        model = tf.keras.models.load_model(ckpt_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=args.lr), loss=loss_fn, metrics=['accuracy'])
        print("[Info] Reloaded best model from:", ckpt_path)
    except Exception as e:
        print("[Warn] Reload best model failed:", e)

    test_txt = os.path.join(args.data_root, 'splits', 'test.txt')
    if os.path.isfile(test_txt):
        test_ids = list_from_txt(test_txt)
        if len(test_ids) > 0:
            test_ds = PatchDataset(args.data_root, test_ids,
                                   batch_size=args.batch_size, patch_size=args.patch_size,
                                   shuffle=False, augment=False,
                                   normalizer=normalizer, use_rgb_only=args.use_rgb_only,
                                   siamese_input=siamese_input)
            _ = evaluate_on_dataset(model, test_ds, desc="Test")
        else:
            print("[Test] splits/test.txt found but empty; skip test evaluation.")
    else:
        print("[Test] splits/test.txt not found; skip test evaluation.")

if __name__ == '__main__':
    main()
