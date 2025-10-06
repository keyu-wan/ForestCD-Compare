import sys
sys.path.append('/mnt/project-data/kwan/AI_Forestry/model')

import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import rasterio
from tqdm import tqdm

from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset
from MambaCD.changedetection.models.ChangeMambaBCD import ChangeMambaBCD
from MambaCD.changedetection.utils_func.metrics import Evaluator

# ========= 与训练保持一致的读入 =========
def _percentile_stretch_to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        vmin, vmax = np.percentile(arr, [1, 99])
        vmax = max(vmax, vmin + 1.0)
        out = (arr - vmin) / (vmax - vmin + 1e-6)
        out = np.where(out < 0, 0, np.where(out > 1, 1, out))
        return (out * 255.0).round().astype(np.uint8)
    else:
        out = np.empty_like(arr, dtype=np.uint8)
        for c in range(arr.shape[-1]):
            vmin, vmax = np.percentile(arr[..., c], [1, 99])
            vmax = max(vmax, vmin + 1.0)
            ch = (arr[..., c] - vmin) / (vmax - vmin + 1e-6)
            ch = np.where(ch < 0, 0, np.where(ch > 1, 1, ch))
            out[..., c] = (ch * 255.0).round().astype(np.uint8)
        return out

def redd_img_loader_percentile_uint8(path, bands=(1,2,3)):
    with rasterio.open(path) as src:
        arr = src.read(bands)
    arr = np.transpose(arr, (1,2,0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = _percentile_stretch_to_uint8(arr)
    return arr.astype(np.float32)

def redd_img_loader_no_stretch(path, bands=(1,2,3)):
    with rasterio.open(path) as src:
        arr = src.read(bands)
    arr = np.transpose(arr, (1,2,0)).astype(np.float32)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = np.where(arr < 0, 0, np.where(arr > 255.0, 255.0, arr))
    return arr

def redd_label_loader_binary(path):
    with rasterio.open(path) as src:
        lab = src.read(1).astype(np.float32)
    if lab.max() > 1.5:
        lab = lab / 255.0
    return (lab > 0.5).astype(np.float32)

def build_redd_loader(use_stretch=True, bands=(1,2,3)):
    def _loader(path):
        if os.sep + 'GT' + os.sep in path or os.sep + 'label' + os.sep in path:
            return redd_label_loader_binary(path)
        return redd_img_loader_percentile_uint8(path, bands=bands) if use_stretch \
               else redd_img_loader_no_stretch(path, bands=bands)
    return _loader

# ========= 保存工具 =========
def save_mask_geotiff(mask01, ref_tif_path, out_tif_path):
    mask_uint8 = (mask01.astype(np.uint8) * 255)
    with rasterio.open(ref_tif_path) as ref:
        profile = ref.profile
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        if mask_uint8.shape != (ref.height, ref.width):
            raise ValueError(f"Size mismatch for {out_tif_path}: mask {mask_uint8.shape} vs ref {(ref.height, ref.width)}")
    with rasterio.open(out_tif_path, 'w', **profile) as dst:
        dst.write(mask_uint8, 1)

def save_prob_geotiff(prob1, ref_tif_path, out_tif_path):
    prob = prob1.astype(np.float32)
    with rasterio.open(ref_tif_path) as ref:
        profile = ref.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(out_tif_path, 'w', **profile) as dst:
        dst.write(prob, 1)

def save_mask_png(mask01, out_png_path):
    import numpy as np
    import imageio.v2 as iio

    # 保证是 numpy 数组
    arr = np.asarray(mask01)

    # 若是布尔/对象/其他非常规类型，统一转为 uint8 的 0/255
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    else:
        # 先确保是数值型（有些环境里可能意外成 object）
        if arr.dtype == object:
            arr = arr.astype(np.float32, copy=False)
        # 掩膜 0/1 -> 0/255，再转成 uint8
        # 如果本来就是 0/255 也没关系，再 cast 一次不会出问题
        arr = (arr * (255 if arr.max() <= 1 else 1)).astype(np.uint8, copy=False)

    iio.imwrite(out_png_path, arr)


# ========= 推理主流程 =========
def main():
    ap = argparse.ArgumentParser("Inference & Export on Test Set (ChangeMambaBCD)")
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--model_ckpt', type=str, required=True)
    ap.add_argument('--dataset', type=str, default='REDD')
    ap.add_argument('--test_dataset_path', type=str, required=True)
    ap.add_argument('--test_data_list_path', type=str, required=True)
    ap.add_argument('--crop_size', type=int, default=256)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--save_prob', type=int, default=0)
    ap.add_argument('--png', type=int, default=1)
    ap.add_argument('--bands', type=str, default='1,2,3')
    ap.add_argument('--no_stretch', action='store_true')
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'tif'), exist_ok=True)
    if args.png:
        os.makedirs(os.path.join(args.out_dir, 'png'), exist_ok=True)
    if args.save_prob:
        os.makedirs(os.path.join(args.out_dir, 'prob_tif'), exist_ok=True)

    with open(args.test_data_list_path, 'r') as f:
        test_names = [line.strip() for line in f if line.strip()]
    print(f"[INFO] test samples: {len(test_names)}")

    class Dummy: pass
    dummy = Dummy()
    dummy.cfg = args.cfg
    dummy.opts = None
    config = get_config(dummy)

    model = ChangeMambaBCD(
        pretrained=None,
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=False,
    ).cuda().eval()

    ckpt = torch.load(args.model_ckpt, map_location='cuda')
    model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] loaded weights from {args.model_ckpt}")

    bands = tuple(int(x) for x in args.bands.split(',') if x.strip())
    loader_fn = build_redd_loader(use_stretch=(not args.no_stretch), bands=bands)

    dataset = ChangeDetectionDatset(
        args.test_dataset_path,
        test_names,
        args.crop_size,
        None,
        'test',
        data_loader=loader_fn
    )
    dl = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, drop_last=False)

    evaluator = Evaluator(num_class=2)
    cm = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dl, desc="Infer(Test)")):
            pre_imgs, post_imgs, labels, names = batch
            name = names[0]
            pre_imgs  = pre_imgs.cuda().float()
            post_imgs = post_imgs.cuda().float()
            labels_t  = labels.cuda().long()  # [1,H,W]

            logits = model(pre_imgs, post_imgs)          # [1,2,H,W]
            probs  = torch.softmax(logits, dim=1)[:,1]   # [1,H,W]
            pred   = torch.argmax(logits, dim=1)         # [1,H,W]

            # === 指标/Evaluator（转 numpy）===
            pred_np = pred.squeeze(0).to(torch.uint8).cpu().numpy()
            labels_np = labels_t.squeeze(0).detach().cpu().numpy().astype(np.int64)
            # evaluator.add_batch(labels_np[None, ...], pred_np[None, ...])
            evaluator.add_batch(
                labels_t.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)[None, ...],
                pred_np.astype(np.int64, copy=False)[None, ...]
            )

            # === 手工混淆矩阵（全用 torch，避免 numpy 归约）===
            gt_t = labels_t.squeeze(0)       # [H,W]
            pr_t = pred.squeeze(0)           # [H,W]
            valid_t = gt_t != 255
            if valid_t.any():
                gt_m = torch.clamp(gt_t, 0, 1).to(torch.int64)[valid_t]  # [N]
                pr_m = torch.clamp(pr_t, 0, 1).to(torch.int64)[valid_t]  # [N]
                k = (gt_m << 1) + pr_m                                    # 0..3
                binc = torch.bincount(k, minlength=4)[:4].to(torch.int64).cpu().tolist()  # 改成 Python list
                # 逐元素相加，避免 numpy 的广播/ufunc
                cm[0, 0] += int(binc[0])  # TN
                cm[0, 1] += int(binc[1])  # FP
                cm[1, 0] += int(binc[2])  # FN
                cm[1, 1] += int(binc[3])  # TP

            # === 保存输出 ===
            gt_path_gt = os.path.join(args.test_dataset_path, 'GT', name)
            gt_path_label = os.path.join(args.test_dataset_path, 'label', name)
            ref_path = gt_path_gt if os.path.exists(gt_path_gt) else gt_path_label
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Cannot find GT/label for {name}: tried {gt_path_gt} and {gt_path_label}")

            out_tif = os.path.join(args.out_dir, 'tif', name)
            save_mask_geotiff(pred_np, ref_path, out_tif)

            if args.save_prob:
                prob_np = probs.squeeze(0).detach().cpu().numpy().astype(np.float32)
                out_prob_tif = os.path.join(args.out_dir, 'prob_tif', name.replace('.tif', '_prob.tif'))
                save_prob_geotiff(prob_np, ref_path, out_prob_tif)

            if args.png:
                out_png = os.path.join(args.out_dir, 'png', name.replace('.tif', '.png'))
                save_mask_png(pred_np, out_png)

    # 汇总指标
    f1 = evaluator.Pixel_F1_score()
    oa = evaluator.Pixel_Accuracy()
    rec = evaluator.Pixel_Recall_Rate()
    pre_ = evaluator.Pixel_Precision_Rate()
    iou = evaluator.Intersection_over_Union()
    kc = evaluator.Kappa_coefficient()

    print("========= TEST METRICS =========")
    print(f"Recall={rec:.6f}, Precision={pre_:.6f}, OA={oa:.6f}, "
          f"F1={f1:.6f}, IoU={iou:.6f}, Kappa={kc:.6f}")
    print("Confusion Matrix (rows=GT, cols=Pred, classes=[0,1]):")
    print(cm)

    # === 分类别指标 ===
    eps = 1e-7
    metrics_per_class = {}
    for cls in [0, 1]:
        TP = int(cm[cls, cls])
        FP = int(cm[:, cls].sum() - TP)
        FN = int(cm[cls, :].sum() - TP)
        precision_c = TP / (TP + FP + eps)
        recall_c    = TP / (TP + FN + eps)
        f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)
        iou_c       = TP / (TP + FP + FN + eps)
        metrics_per_class[cls] = {
            "Precision": precision_c,
            "Recall": recall_c,
            "F1": f1_c,
            "IoU": iou_c
        }
        print(f"[Class {cls}] Precision={precision_c:.6f} "
              f"Recall={recall_c:.6f} F1={f1_c:.6f} IoU={iou_c:.6f}")

    # === 宏平均 (macro average) ===
    macro_metrics = {
        "Precision": np.mean([metrics_per_class[c]["Precision"] for c in metrics_per_class]),
        "Recall":    np.mean([metrics_per_class[c]["Recall"] for c in metrics_per_class]),
        "F1":        np.mean([metrics_per_class[c]["F1"] for c in metrics_per_class]),
        "IoU":       np.mean([metrics_per_class[c]["IoU"] for c in metrics_per_class]),
    }
    print("=== Macro Average over classes ===")
    print(f"Precision={macro_metrics['Precision']:.6f} "
          f"Recall={macro_metrics['Recall']:.6f} "
          f"F1={macro_metrics['F1']:.6f} "
          f"IoU={macro_metrics['IoU']:.6f}")

    # 保存到 json
    out_report = {
        "Overall": {
            "Recall": float(rec),
            "Precision": float(pre_),
            "OA": float(oa),
            "F1": float(f1),
            "IoU": float(iou),
            "Kappa": float(kc),
        },
        "ConfusionMatrix": cm.tolist(),
        "PerClass": metrics_per_class,
        "MacroAverage": macro_metrics,
        "num_test": len(test_names)
    }
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"[INFO] Saved detailed report to {os.path.join(args.out_dir, 'test_metrics.json')}")

if __name__ == "__main__":
    main()
