import os
import csv
import json
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision
from train import OnlineStandardizer, PatchDataset, combined_loss, list_from_txt, IMAGENET_MEAN, IMAGENET_STD
from resunet import build_res_unet

import optuna
from optuna.integration import TFKerasPruningCallback

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Mixed precision + memory growth
mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("[Warn] set_memory_growth failed:", e)
        
# ----------------------------
# Utils
# ----------------------------
def str2bool(v):
    return str(v).lower() in ['1', 'true', 'yes', 'y']

def make_imagenet_stats(in_ch: int, siamese_input: bool, use_rgb_only: bool):
    """
    Returns (mean, std) arrays that match the input layout expected by PatchDataset.
    For siamese_input=True, we need per-branch 3-channel stats.
    """
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


# ----------------------------
# Dataset builders
# ----------------------------
def make_datasets_siamese(data_root, batch_size, patch_size, use_rgb_only, seed=42):
    """
    Siamese: dataset yields [X_t1, X_t2], Y with each branch (H, W, 3).
    """
    train_ids = list_from_txt(os.path.join(data_root, 'splits', 'train.txt'))
    val_ids   = list_from_txt(os.path.join(data_root, 'splits', 'val.txt'))
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise FileNotFoundError("Empty train/val split. Check splits/train.txt and splits/val.txt")

    # Each branch is 3 channels (RGB), so in_ch=3 for normalizer statistics
    mean_im, std_im = make_imagenet_stats(in_ch=3, siamese_input=True, use_rgb_only=use_rgb_only)
    normalizer = OnlineStandardizer(mean=mean_im, std=std_im)

    train_ds = PatchDataset(
        data_root, train_ids,
        batch_size=batch_size, patch_size=patch_size,
        shuffle=True, augment=True,
        normalizer=normalizer, use_rgb_only=use_rgb_only,
        siamese_input=True
    )
    val_ds = PatchDataset(
        data_root, val_ids,
        batch_size=batch_size, patch_size=patch_size,
        shuffle=False, augment=False,
        normalizer=normalizer, use_rgb_only=use_rgb_only,
        siamese_input=True
    )
    return train_ds, val_ds, normalizer


# ----------------------------
# Metrics for binary segmentation (streaming over dataset)
# ----------------------------
def _ensure_int_labels(y):
    y = np.asarray(y)
    if y.ndim == 4 and y.shape[-1] == 1:
        y = np.squeeze(y, axis=-1)
    elif y.ndim == 4 and y.shape[-1] == 2:
        y = np.argmax(y, axis=-1)
    return y.astype(np.int64)

def _pred_to_labels(pred):
    pred = np.asarray(pred)
    if pred.ndim == 4 and pred.shape[-1] == 2:
        pred = np.argmax(pred, axis=-1)
    return pred.astype(np.int64)

def evaluate_binary_metrics_stream(model, dataset, verbose=0):
    tp1 = fp1 = fn1 = tn1 = 0
    total = correct = 0
    for batch in dataset:
        x, y_true = batch   # x is [X_t1, X_t2] for siamese
        pred = model.predict_on_batch(x)
        y_hat = _pred_to_labels(pred)
        y     = _ensure_int_labels(y_true)
        y = y.reshape(-1); y_hat = y_hat.reshape(-1)
        tp1 += int(np.sum((y_hat == 1) & (y == 1)))
        fp1 += int(np.sum((y_hat == 1) & (y != 1)))
        fn1 += int(np.sum((y_hat != 1) & (y == 1)))
        tn1 += int(np.sum((y_hat == 0) & (y == 0)))
        total   += y.size
        correct += int(np.sum(y_hat == y))
    acc = correct / max(1, total)
    prec1 = tp1 / max(1, (tp1 + fp1))
    rec1  = tp1 / max(1, (tp1 + fn1))
    f1_1  = 2 * prec1 * rec1 / max(1e-7, (prec1 + rec1))
    iou1  = tp1 / max(1, (tp1 + fp1 + fn1))
    tp0 = tn1
    fp0 = fn1
    fn0 = fp1
    prec0 = tp0 / max(1, (tp0 + fp0))
    rec0  = tp0 / max(1, (tp0 + fn0))
    f1_0  = 2 * prec0 * rec0 / max(1e-7, (prec0 + rec0))
    iou0  = tp0 / max(1, (tp0 + fp0 + fn0))
    miou = (iou0 + iou1) / 2.0
    return {
        "accuracy": acc,
        "precision_class0": prec0,
        "recall_class0": rec0,
        "f1_class0": f1_0,
        "iou_class0": iou0,
        "precision_class1": prec1,
        "recall_class1": rec1,
        "f1_class1": f1_1,
        "iou_class1": iou1,
        "miou": miou,
        "tp1": tp1, "fp1": fp1, "fn1": fn1, "tn1": tn1,
        "total_pixels": total
    }


# ----------------------------
# Model builder from trial params (Siamese)
# ----------------------------
def build_siamese_model_from_params(trial, input_shape):
    """
    input_shape = (H, W, 3) per branch
    """
    act          = trial.suggest_categorical('act', ['gelu'])
    preact       = trial.suggest_categorical('preact', [True])
    up_mode      = trial.suggest_categorical('up_mode', ['nearest'])
    se_ratio     = trial.suggest_categorical('se_ratio', [0, 8])
    depth        = trial.suggest_categorical('depth', [4])
    stem         = trial.suggest_categorical('stem_filters', [64])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.4, step=0.05)

    model = build_res_unet(
        input_shape=input_shape, nClasses=2,
        stem_filters=stem,
        depth=depth, up_mode=up_mode,
        preact=preact, se_ratio=se_ratio,
        dropout_rate=dropout_rate, act=act
    )
    model._model_name = 'concat_siamese'
    return model


# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial, args):
    # Datasets (siamese mode)
    input_shape = (args.patch_size, args.patch_size, 3)
    train_ds, val_ds, _ = make_datasets_siamese(
        args.data_root, args.batch_size, args.patch_size, args.use_rgb_only, seed=args.seed
    )

    # Build model
    model = build_siamese_model_from_params(trial, input_shape=input_shape)

    # Optional warm-start
    init_loaded = False
    if args.init_weights and os.path.exists(args.init_weights):
        try:
            model.load_weights(args.init_weights, by_name=True, skip_mismatch=True)
            init_loaded = True
            print(f"[Info] Loaded init weights: {args.init_weights}")
        except Exception as e:
            print(f"[Warn] Failed to load init weights ({args.init_weights}): {e}")

    # Loss + optim params
    class_w0 = trial.suggest_float('class_weight_0', 0.1, 0.6)
    class_w1 = trial.suggest_float('class_weight_1', 1.2, 3.0)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    lam_focal = trial.suggest_float('lambda_focal', 0.5, 1.5)
    lam_dice  = trial.suggest_float('lambda_dice', 0.5, 1.8)
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)

    loss_fn = combined_loss(alpha=(class_w0, class_w1), gamma=gamma,
                            lambda_focal=lam_focal, lambda_dice=lam_dice)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=['accuracy'])

    # Trial directory
    trial_dir = os.path.join(args.save_dir, f'optuna_t{trial.number:04d}')
    os.makedirs(trial_dir, exist_ok=True)

    # Callbacks
    ckpt_path = os.path.join(trial_dir, f'best_{getattr(model, "_model_name", "concat_siamese")}.h5')
    ckpt = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=0)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, args.patience // 4),
                              min_lr=1e-6, verbose=0)
    estop = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=0)
    pruner_cb = TFKerasPruningCallback(trial, monitor='val_loss')

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, rlrop, estop, pruner_cb],
        verbose=0
    )

    val_loss_best = float(np.min(history.history['val_loss']))
    metrics = evaluate_binary_metrics_stream(model, val_ds, verbose=0)

    # Save per-trial metrics
    save_trial_metrics(trial_dir, metrics, extra_info={
        "val_loss_best": val_loss_best,
        "trial_number": trial.number,
        "params": trial.params,
        "init_weights_used": init_loaded,
        "init_weights_path": args.init_weights if init_loaded else ""
    })

    # Append to global summary
    row = {
        "trial_number": trial.number,
        "value_objective": -metrics["iou_class1"],
        "iou_class1": metrics["iou_class1"],
        "iou_class0": metrics["iou_class0"],
        "miou": metrics["miou"],
        "f1_class1": metrics["f1_class1"],
        "f1_class0": metrics["f1_class0"],
        "precision_class1": metrics["precision_class1"],
        "recall_class1": metrics["recall_class1"],
        "precision_class0": metrics["precision_class0"],
        "recall_class0": metrics["recall_class0"],
        "accuracy": metrics["accuracy"],
        "val_loss_best": val_loss_best,
        "params_json": json.dumps(trial.params, ensure_ascii=False),
        "trial_dir": trial_dir
    }
    append_global_metrics(args.save_dir, row)

    # Minimize objective = -IoU_class1
    return -metrics["iou_class1"]


# ----------------------------
# Metrics saving helpers
# ----------------------------
def save_trial_metrics(trial_dir, metrics_dict, extra_info=None):
    os.makedirs(trial_dir, exist_ok=True)
    csv_path = os.path.join(trial_dir, "metrics.csv")
    keys = list(metrics_dict.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerow([metrics_dict[k] for k in keys])
    js = {"metrics": metrics_dict}
    if extra_info is not None:
        js.update(extra_info)
    with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
        json.dump(js, f, indent=2)

def append_global_metrics(save_dir, row_dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "trials_metrics.csv")
    write_header = not os.path.exists(path)
    fieldnames = [
        "trial_number", "value_objective",
        "iou_class1", "iou_class0", "miou",
        "f1_class1", "f1_class0",
        "precision_class1", "recall_class1",
        "precision_class0", "recall_class0",
        "accuracy", "val_loss_best",
        "params_json", "trial_dir"
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row_dict.get(k, "") for k in fieldnames})


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./optuna_runs_siamese')
    ap.add_argument('--study_name', type=str, default='resunet_siamese_redd')
    ap.add_argument('--storage', type=str, default='sqlite:///optuna_resunet_siamese.db')
    ap.add_argument('--n_trials', type=int, default=30)
    ap.add_argument('--timeout', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--use_rgb_only', type=str2bool, default=True)  # must be True for (H,W,3) branches
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=8)

    # Optional warm-start
    ap.add_argument('--init_weights', type=str, default='', help='Optional .h5 weights to warm-start each trial.')
    ap.add_argument('--baseline_params', type=str, default='', help='JSON file of hyperparams for baseline trial.')
    ap.add_argument('--enqueue_baseline', action='store_true', help='Enqueue baseline params as the first trial.')

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    )

    if args.enqueue_baseline and args.baseline_params and os.path.exists(args.baseline_params):
        with open(args.baseline_params, 'r') as f:
            base_params = json.load(f)
        print("[Info] Enqueue baseline params as first trial:", base_params)
        study.enqueue_trial(base_params)

    study.optimize(
        lambda tr: objective(tr, args),
        n_trials=args.n_trials,
        timeout=None if args.timeout <= 0 else args.timeout,
        gc_after_trial=True
    )

    print("== Best trial ==")
    bt = study.best_trial
    print("objective (=-IoU_class1):", bt.value)
    print("=> IoU_class1:", -bt.value)
    print("params:")
    for k, v in bt.params.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
