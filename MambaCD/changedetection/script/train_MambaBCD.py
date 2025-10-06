# import sys
# sys.path.append('/mnt/project-data/kwan/AI_Forestry/model')

# import argparse
# import os
# import time

# import numpy as np

# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch.nn as nn

# from MambaCD.changedetection.configs.config import get_config
# from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
# from MambaCD.changedetection.utils_func.metrics import Evaluator
# from MambaCD.changedetection.models.ChangeMambaBCD import ChangeMambaBCD
# import MambaCD.changedetection.utils_func.lovasz_loss as L

# # 粗估全数据的前景/背景像素数（可以随便抽100~500张）
# import rasterio
# import math

# from torch.cuda.amp import GradScaler, autocast


# def estimate_class_weights(gt_dir, name_list, sample_k=300):
#     """粗估前景/背景像素占比，给 CE 设权重用。"""
#     from random import sample
#     pick = sample(name_list, min(sample_k, len(name_list)))
#     pos = neg = 0
#     for n in pick:
#         with rasterio.open(os.path.join(gt_dir, n)) as src:
#             lab = src.read(1)
#         lab = (lab > 0).astype(np.int64)
#         pos += int(lab.sum())
#         neg += int(lab.size - lab.sum())
#     pos = max(pos, 1); neg = max(neg, 1)
#     w_bg = 1.0
#     w_fg = float(np.clip(neg / pos, 1.0, 20.0))  # 反频率，截断到[1,20]
#     return w_bg, w_fg

# def init_fg_bias_2class(final_conv, p_fg=0.02):
#     """2类最后一层 bias 以先验 p_fg 初始化：bias[1]-bias[0] = logit(p_fg)。"""
#     b = math.log(p_fg / (1 - p_fg))
#     with torch.no_grad():
#         if final_conv.bias is None:
#             final_conv.bias = torch.nn.Parameter(torch.zeros(final_conv.out_channels, device=final_conv.weight.device))
#         final_conv.bias.zero_()
#         if final_conv.out_channels >= 2:
#             final_conv.bias[1].add_(b)

# @torch.no_grad()
# def diag_print_logits_ratio(logits, labels, step, every=50):
#     """诊断：预测与GT的前景比例（2类 argmax）。"""
#     if step % every != 0: return
#     pred = logits.argmax(dim=1)
#     fg_pred = (pred == 1).float().mean().item()
#     fg_gt   = (labels == 1).float().mean().item()
#     print(f"[diag] step {step} | fg_ratio_pred={fg_pred:.5f} fg_ratio_gt={fg_gt:.5f}")

# def freeze_all_bn(model):
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#             m.eval()
#             for p in m.parameters():
#                 p.requires_grad_(False)

# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         config = get_config(args)

#         self.train_data_loader = make_data_loader(args)

#         self.evaluator = Evaluator(num_class=2)

#         self.deep_model = ChangeMambaBCD(
#             pretrained=args.pretrained_weight_path,
#             patch_size=config.MODEL.VSSM.PATCH_SIZE,
#             in_chans=config.MODEL.VSSM.IN_CHANS,
#             num_classes=config.MODEL.NUM_CLASSES,
#             depths=config.MODEL.VSSM.DEPTHS,
#             dims=config.MODEL.VSSM.EMBED_DIM,
#             # ===================
#             ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#             ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#             ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#             ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#             ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#             ssm_conv=config.MODEL.VSSM.SSM_CONV,
#             ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#             ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#             ssm_init=config.MODEL.VSSM.SSM_INIT,
#             forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#             # ===================
#             mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#             mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#             mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#             # ===================
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             patch_norm=config.MODEL.VSSM.PATCH_NORM,
#             norm_layer=config.MODEL.VSSM.NORM_LAYER,
#             downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#             patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#             gmlp=config.MODEL.VSSM.GMLP,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#         )
#         self.deep_model = self.deep_model.cuda()
#         freeze_all_bn(self.deep_model)
#         self.model_save_path = os.path.join(args.model_param_path, args.dataset,
#                                             args.model_type + '_' + str(time.time()))
#         self.lr = args.learning_rate
#         self.epoch = args.max_iters // args.batch_size

#                 # ---- 保证 val 与 train 完全一致的 REDD 读取方式 ----
#         def _percentile_stretch_to_uint8(arr):
#             import numpy as np
#             if arr.dtype == np.uint8:
#                 return arr
#             arr = arr.astype(np.float32)
#             if arr.ndim == 2:
#                 vmin, vmax = np.percentile(arr, [1, 99])
#                 vmax = max(vmax, vmin + 1.0)
#                 out = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
#                 return (out * 255.0).round().astype(np.uint8)
#             else:
#                 out = np.empty_like(arr, dtype=np.uint8)
#                 for c in range(arr.shape[-1]):
#                     vmin, vmax = np.percentile(arr[..., c], [1, 99])
#                     vmax = max(vmax, vmin + 1.0)
#                     ch = np.clip((arr[..., c] - vmin) / (vmax - vmin + 1e-6), 0, 1)
#                     out[..., c] = (ch * 255.0).round().astype(np.uint8)
#                 return out

#         def redd_img_loader_percentile_uint8(path, bands=(1, 2, 3)):
#             import rasterio, numpy as np
#             with rasterio.open(path) as src:
#                 arr = src.read(bands)  # (C,H,W)
#             arr = np.transpose(arr, (1, 2, 0))  # HWC
#             if arr.shape[-1] == 1:
#                 arr = np.repeat(arr, 3, axis=-1)
#             elif arr.shape[-1] > 3:
#                 arr = arr[..., :3]
#             arr = _percentile_stretch_to_uint8(arr)   # uint8
#             return arr.astype(np.float32)             # 0–255 float32

#         def redd_img_loader_no_stretch(path, bands=(1, 2, 3)):
#             import rasterio, numpy as np
#             with rasterio.open(path) as src:
#                 arr = src.read(bands)
#             arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
#             if arr.shape[-1] == 1:
#                 arr = np.repeat(arr, 3, axis=-1)
#             elif arr.shape[-1] > 3:
#                 arr = arr[..., :3]
#             arr = np.clip(arr, 0, 255.0)
#             return arr  # 0–255 float32

#         def redd_label_loader_binary(path):
#             import rasterio, numpy as np
#             with rasterio.open(path) as src:
#                 lab = src.read(1)
#             lab = lab.astype(np.float32)
#             if lab.max() > 1.5:
#                 lab = lab / 255.0
#             lab = (lab > 0.5).astype(np.float32)
#             return lab  # (H,W) in {0,1}

#         def build_redd_loader():
#             # 与 make_data_loader 一致：默认使用 1–99% 分位拉伸
#             use_stretch = getattr(self.args, 'redd_percentile_stretch', True)
#             bands = getattr(self.args, 'redd_bands', (1, 2, 3))

#             def _loader(path):
#                 # 标签：路径包含 GT 或 label
#                 if os.sep + 'GT' + os.sep in path or os.sep + 'label' + os.sep in path:
#                     return redd_label_loader_binary(path)
#                 # 影像
#                 if use_stretch:
#                     return redd_img_loader_percentile_uint8(path, bands=bands)
#                 else:
#                     return redd_img_loader_no_stretch(path, bands=bands)
#             return _loader

#         # 暴露给 validation() 使用
#         self._redd_loader_for_val = build_redd_loader() if 'REDD' in self.args.dataset else None

#         if not os.path.exists(self.model_save_path):
#             os.makedirs(self.model_save_path)

#         if args.resume is not None:
#             if not os.path.isfile(args.resume):
#                 raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             model_dict = {}
#             state_dict = self.deep_model.state_dict()
#             for k, v in checkpoint.items():
#                 if k in state_dict:
#                     model_dict[k] = v
#             state_dict.update(model_dict)
#             self.deep_model.load_state_dict(state_dict)

#         self.optim = optim.AdamW(self.deep_model.parameters(),
#                                  lr=args.learning_rate,
#                                  weight_decay=args.weight_decay)
#         self.scaler = GradScaler()# to allow mixed precision training


#     def training(self):
#         """
#         冻结BN + 复合损失(Weighted CE + λ_lovasz*Lovasz + λ_focal*Focal) + CosineWarmup
#         - 每个 epoch：打印平均 TrainLoss(含各分量)；跑一次 validation() 打印全指标
#         - 打印 fg_ratio 心跳；不做阈值搜索；不改 batch size；不引入 TTA
#         - w_fg 采用 --w_fg_cap 上限裁剪
#         - 按验证 F1 保存最优权重，并生成 best.pth 软链接/拷贝
#         """
#         import math
#         import os
#         import numpy as np
#         import torch
#         from torch.cuda.amp import autocast

#         print("steps_per_epoch =", len(self.train_data_loader))


#         # ========= 类权重 =========
#         gt_dir = os.path.join(self.args.train_dataset_path, 'GT')
#         w_bg, w_fg = estimate_class_weights(gt_dir, self.args.train_data_name_list, sample_k=300)
#         w_fg = float(np.clip(w_fg, 1.0, float(self.args.w_fg_cap)))
#         weight = torch.tensor([1.0, w_fg], dtype=torch.float32, device='cuda')
#         criterion_ce = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=255)

#         # ========= Focal Loss (softmax 版) =========
#         def focal_loss_softmax(logits, labels, gamma=2.0, ignore_index=255, alpha=None):
#             probs = torch.softmax(logits, dim=1)  # [B,2,H,W]
#             tgt = labels.clamp_min(0).clamp_max(1)
#             oh = torch.nn.functional.one_hot(tgt, num_classes=2).permute(0,3,1,2).contiguous()
#             valid = (labels != ignore_index)
#             if valid.sum() == 0:
#                 return logits.new_zeros(())
#             probs_v = probs.permute(0,2,3,1)[valid]
#             oh_v    = oh.permute(0,2,3,1)[valid]
#             pt = (probs_v * oh_v).sum(dim=1).clamp_min(1e-7)
#             loss = ((1.0 - pt) ** gamma) * (-pt.log())
#             if alpha is not None:
#                 alpha_v = (oh_v * alpha).sum(dim=1)
#                 loss = alpha_v * loss
#             return loss.mean()

#         alpha_for_focal = weight
#         lambda_lovasz = float(self.args.lambda_lovasz)
#         lambda_focal  = float(self.args.lambda_focal)
#         gamma_focal   = float(self.args.gamma_focal)

#         # ========= 先验初始化最后一层 bias =========
#         p_fg = 1.0 / (1.0 + float(w_fg))
#         init_fg_bias_2class(self.deep_model.main_clf, p_fg=p_fg)

#         # ========= 训练日程：支持 epochs_per_trial（最小改动点）=========
#         elem_num = len(self.train_data_loader)  # 每个 epoch 的 iter 数
#         if elem_num == 0:
#             raise RuntimeError("Empty train_data_loader.")

#         if getattr(self.args, 'epochs_per_trial', 0) and self.args.epochs_per_trial > 0:
#             num_epochs = int(self.args.epochs_per_trial)
#             total_iters = num_epochs * elem_num
#         else:
#             total_iters = int(self.args.max_iters)
#             num_epochs = max(1, total_iters // elem_num)

#         warmup_ratio = float(getattr(self.args, 'warmup_ratio', 0.05))
#         warmup_iters = max(1, int(warmup_ratio * total_iters))

#         def lr_lambda(last_epoch: int):
#             cur_iter = last_epoch  # LambdaLR 用 last_epoch 记录 step 次数-1，这里等价看成 iter 索引
#             if cur_iter < warmup_iters:
#                 return float(cur_iter + 1) / float(warmup_iters)  # 线性 warmup
#             denom = max(1, (total_iters - warmup_iters))
#             progress = (cur_iter - warmup_iters) / denom
#             return 0.5 * (1.0 + math.cos(math.pi * progress))

#         # —— 用 last_epoch 对齐 scheduler（最小改动）——
#         global_iter = int(getattr(self, 'resume_global_iter', 0))
#         scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=global_iter-1)

#         best_score = -1.0
#         best_round = []

#         for epoch in range(num_epochs):
#             self.deep_model.train()
#             loss_epoch = ce_epoch = lovasz_epoch = focal_epoch = 0.0

#             for itera, data in enumerate(tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
#                 pre_change_imgs, post_change_imgs, labels, _ = data
#                 pre_change_imgs  = pre_change_imgs.cuda().float()
#                 post_change_imgs = post_change_imgs.cuda().float()
#                 labels           = labels.cuda().long()

#                 self.optim.zero_grad(set_to_none=True)
#                 with autocast():
#                     logits = self.deep_model(pre_change_imgs, post_change_imgs)
#                     ce_loss = criterion_ce(logits, labels)
#                     lovasz  = L.lovasz_softmax(torch.softmax(logits, dim=1), labels, ignore=255)
#                     focal   = focal_loss_softmax(logits, labels, gamma=gamma_focal, alpha=alpha_for_focal)
#                     final_loss = ce_loss + lambda_lovasz * lovasz + lambda_focal * focal

#                 # AMP + 只在成功更新时推进 scheduler（保留你之前的安全写法）
#                 self.scaler.scale(final_loss).backward()
#                 prev_scale = self.scaler.get_scale()
#                 self.scaler.step(self.optim)
#                 self.scaler.update()
#                 if self.scaler.get_scale() >= prev_scale:
#                     scheduler.step()

#                 loss_epoch   += float(final_loss)
#                 ce_epoch     += float(ce_loss)
#                 lovasz_epoch += float(lovasz)
#                 focal_epoch  += float(focal)
#                 global_iter  += 1

#                 # 心跳
#                 if (itera + 1) % 50 == 0:
#                     with torch.no_grad():
#                         pred_argmax = logits.argmax(dim=1)
#                         probs1 = torch.softmax(logits, dim=1)[:, 1]
#                         fg_argmax = (pred_argmax == 1).float().mean().item()
#                         fg_p05    = (probs1 > 0.5).float().mean().item()
#                         fg_gt     = (labels == 1).float().mean().item()
#                         print(f"[diag] iter {itera+1} | fg_argmax={fg_argmax:.5f} fg_prob>0.5={fg_p05:.5f} fg_gt={fg_gt:.5f}")

#                 if global_iter >= total_iters:
#                     break

#             n_batches = itera + 1
#             try:
#                 cur_lr = scheduler.get_last_lr()[0]
#             except Exception:
#                 cur_lr = self.optim.param_groups[0]['lr']

#             print(f"[Epoch {epoch+1}] TrainLoss={loss_epoch/n_batches:.4f} "
#                 f"(CE={ce_epoch/n_batches:.4f}, Lovasz={lovasz_epoch/n_batches:.4f}, Focal={focal_epoch/n_batches:.4f}) "
#                 f"| lr={cur_lr:.2e}")

#             # 验证 + 保存最优（按 F1）
#             self.deep_model.eval()
#             rec, pre, oa, f1_score, iou, kc = self.validation()
#             self.deep_model.train()

#             score = float(f1_score)
#             if score > best_score:
#                 os.makedirs(self.model_save_path, exist_ok=True)
#                 ckpt_name = f'best_F1{f1_score:.4f}_IoU{iou:.4f}_epoch{epoch+1}.pth'
#                 best_path = os.path.join(self.model_save_path, ckpt_name)
#                 torch.save(self.deep_model.state_dict(), best_path)
#                 link_path = os.path.join(self.model_save_path, 'best.pth')
#                 try:
#                     if os.path.islink(link_path) or os.path.exists(link_path):
#                         os.remove(link_path)
#                     os.symlink(best_path, link_path)
#                 except Exception:
#                     import shutil; shutil.copy2(best_path, link_path)
#                 best_score = score
#                 best_round = [rec, pre, oa, f1_score, iou, kc]
#                 print(f"[best@epoch{epoch+1}] F1={f1_score:.4f} IoU={iou:.4f} (model saved)")

#             if global_iter >= total_iters:
#                 break

#         print("Best (rec, pre, oa, f1, iou, kappa):", best_round)



#     def validation(self):
#         import numpy as np
#         from torch.utils.data import DataLoader
#         print('---------starting evaluation-----------')

#         # 进入 eval 模式（退出时恢复）
#         was_training = self.deep_model.training
#         self.deep_model.eval()

#         self.evaluator.reset()
#         cm = np.zeros((2, 2), dtype=np.int64)

#         # === 与训练完全一致的读取与归一化 ===
#         if 'REDD' in self.args.dataset and self._redd_loader_for_val is not None:
#             loader_for_val = self._redd_loader_for_val
#         else:
#             # 其它数据集沿用默认 img_loader + imutils.normalize_img
#             loader_for_val = None  # 让 Dataset 走默认

#         dataset = ChangeDetectionDatset(
#             self.args.test_dataset_path,
#             self.args.test_data_name_list,
#             self.args.crop_size,
#             None,
#             'test',
#             data_loader=loader_for_val if loader_for_val is not None else None
#         )
#         val_workers = getattr(self.args, 'num_workers', 4)
#         val_data_loader = DataLoader(dataset, batch_size=1, num_workers=val_workers, drop_last=False)

#         torch.cuda.empty_cache()

#         with torch.no_grad():
#             for _, data in enumerate(val_data_loader):
#                 pre_change_imgs, post_change_imgs, labels, _ = data
#                 pre_change_imgs  = pre_change_imgs.cuda().float()
#                 post_change_imgs = post_change_imgs.cuda().float()
#                 labels           = labels.cuda().long()

#                 logits = self.deep_model(pre_change_imgs, post_change_imgs)  # [B,2,H,W]

#                 # Evaluator 总体统计
#                 pred_np   = np.asarray(torch.argmax(logits, dim=1).cpu().numpy(), dtype=np.int64)
#                 labels_np = np.asarray(labels.cpu().numpy(),                    dtype=np.int64)
#                 self.evaluator.add_batch(labels_np, pred_np)

#                 # 手工 2x2 混淆矩阵（忽略255）
#                 gt = labels_np.reshape(-1).astype(np.int64, copy=False)
#                 pr = pred_np.reshape(-1).astype(np.int64,  copy=False)
#                 valid = (gt != 255)
#                 if np.any(valid):
#                     gt_m = np.clip(gt[valid], 0, 1).astype(np.int64, copy=False)
#                     pr_m = np.clip(pr[valid], 0, 1).astype(np.int64, copy=False)
#                     k = (gt_m << 1) + pr_m  # [0..3]
#                     k = np.asarray(k, dtype=np.intp)
#                     cm += np.bincount(k, minlength=4).reshape(2, 2)

#         # —— 计算并打印指标 —— 
#         f1_score = self.evaluator.Pixel_F1_score()
#         oa       = self.evaluator.Pixel_Accuracy()
#         rec      = self.evaluator.Pixel_Recall_Rate()
#         pre      = self.evaluator.Pixel_Precision_Rate()
#         iou      = self.evaluator.Intersection_over_Union()
#         kc       = self.evaluator.Kappa_coefficient()
#         print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
#               f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')

#         print("Confusion Matrix (rows=GT, cols=Pred, classes=[0,1]):")
#         print(cm)

#         eps = 1e-7
#         for cls in [0, 1]:
#             TP = int(cm[cls, cls])
#             FP = int(cm[:, cls].sum() - TP)
#             FN = int(cm[cls, :].sum() - TP)
#             precision_c = TP / (TP + FP + eps)
#             recall_c    = TP / (TP + FN + eps)
#             f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)
#             iou_c       = TP / (TP + FP + FN + eps)
#             print(f"[Class {cls}] Precision={precision_c:.6f} "
#                   f"Recall={recall_c:.6f} F1={f1_c:.6f} IoU={iou_c:.6f}")

#         if was_training:
#             self.deep_model.train()

#         return rec, pre, oa, f1_score, iou, kc




# def main():
#     parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
#     parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
#     parser.add_argument("--opts", default=None, nargs='+')
#     parser.add_argument('--pretrained_weight_path', type=str)
#     parser.add_argument('--dataset', type=str, default='SYSU')
#     parser.add_argument('--type', type=str, default='train')
#     parser.add_argument('--train_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/train')
#     parser.add_argument('--train_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/train_list.txt')
#     parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
#     parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
#     parser.add_argument('--shuffle', type=bool, default=True)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--crop_size', type=int, default=256)
#     parser.add_argument('--train_data_name_list', type=list)
#     parser.add_argument('--test_data_name_list', type=list)
#     parser.add_argument('--start_iter', type=int, default=0)
#     parser.add_argument('--cuda', type=bool, default=True)
#     parser.add_argument('--max_iters', type=int, default=240000)
#     parser.add_argument('--model_type', type=str, default='ChangeMambaBCD')
#     parser.add_argument('--model_param_path', type=str, default='../saved_models')

#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--learning_rate', type=float, default=1e-4)
#     parser.add_argument('--momentum', type=float, default=0.9)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)

#     # === 复合损失 & 调度 ===
#     parser.add_argument('--lambda_lovasz', type=float, default=0.75)
#     parser.add_argument('--lambda_focal', type=float, default=0.5)
#     parser.add_argument('--gamma_focal', type=float, default=2.0)
#     parser.add_argument('--w_fg_cap', type=float, default=20.0)
#     parser.add_argument('--warmup_ratio', type=float, default=0.05)

#     # === 新增：每个 trial 训练多少个 epoch；0 表示按 max_iters 旧逻辑 ===
#     parser.add_argument('--epochs_per_trial', type=int, default=0)

#     args = parser.parse_args()

#     with open(args.train_data_list_path, "r") as f:
#         data_name_list = [data_name.strip() for data_name in f]
#     args.train_data_name_list = data_name_list

#     with open(args.test_data_list_path, "r") as f:
#         test_data_name_list = [data_name.strip() for data_name in f]
#     args.test_data_name_list = test_data_name_list

#     trainer = Trainer(args)
    
#     trainer.training()


# if __name__ == "__main__":
#     main()


import sys
sys.path.append('/mnt/project-data/kwan/AI_Forestry/model')

import argparse
import os
import time
import math
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.ChangeMambaBCD import ChangeMambaBCD
import MambaCD.changedetection.utils_func.lovasz_loss as L

import rasterio
from torch.cuda.amp import GradScaler, autocast


def estimate_class_weights(gt_dir, name_list, sample_k=300):
    """粗估前景/背景像素占比，给 CE 设权重用。"""
    from random import sample
    pick = sample(name_list, min(sample_k, len(name_list)))
    pos = neg = 0
    for n in pick:
        with rasterio.open(os.path.join(gt_dir, n)) as src:
            lab = src.read(1)
        lab = (lab > 0).astype(np.int64)
        pos += int(lab.sum())
        neg += int(lab.size - lab.sum())
    pos = max(pos, 1); neg = max(neg, 1)
    w_bg = 1.0
    w_fg = float(np.clip(neg / pos, 1.0, 20.0))
    return w_bg, w_fg


def init_fg_bias_2class(final_conv, p_fg=0.02):
    """2类最后一层 bias 以先验 p_fg 初始化：bias[1]-bias[0] = logit(p_fg)。"""
    b = math.log(p_fg / (1 - p_fg))
    with torch.no_grad():
        if final_conv.bias is None:
            final_conv.bias = torch.nn.Parameter(torch.zeros(final_conv.out_channels, device=final_conv.weight.device))
        final_conv.bias.zero_()
        if final_conv.out_channels >= 2:
            final_conv.bias[1].add_(b)


def freeze_all_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        # ====== Data & Metrics ======
        self.train_data_loader = make_data_loader(args)
        self.evaluator = Evaluator(num_class=2)

        # ====== Model ======
        self.deep_model = ChangeMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # SSM
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
            # MLP
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # misc
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ).cuda()
        freeze_all_bn(self.deep_model)

        # ====== Save path ======
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        os.makedirs(self.model_save_path, exist_ok=True)

        # ====== Optim & AMP ======
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.scaler = GradScaler()

        # ====== REDD 统一读入（供 eval/val 用）======
        def _percentile_stretch_to_uint8(arr):
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

        def redd_img_loader_percentile_uint8(path, bands=(1, 2, 3)):
            with rasterio.open(path) as src:
                arr = src.read(bands)  # (C,H,W)
            arr = np.transpose(arr, (1, 2, 0))  # HWC
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[..., :3]
            arr = _percentile_stretch_to_uint8(arr)
            return arr.astype(np.float32)

        def redd_img_loader_no_stretch(path, bands=(1, 2, 3)):
            with rasterio.open(path) as src:
                arr = src.read(bands)
            arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[..., :3]
            arr = np.clip(arr, 0, 255.0)
            return arr

        def redd_label_loader_binary(path):
            with rasterio.open(path) as src:
                lab = src.read(1).astype(np.float32)
            if lab.max() > 1.5:
                lab = lab / 255.0
            return (lab > 0.5).astype(np.float32)

        def build_redd_loader():
            use_stretch = getattr(self.args, 'redd_percentile_stretch', True)
            bands = getattr(self.args, 'redd_bands', (1, 2, 3))
            def _loader(path):
                if os.sep + 'GT' + os.sep in path or os.sep + 'label' + os.sep in path:
                    return redd_label_loader_binary(path)
                return (redd_img_loader_percentile_uint8 if use_stretch else redd_img_loader_no_stretch)(path, bands=bands)
            return _loader

        self._redd_loader_for_eval = build_redd_loader() if 'REDD' in self.args.dataset else None

        # ====== Resume ======
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            state_dict = self.deep_model.state_dict()
            loadable = {k: v for k, v in checkpoint.items() if k in state_dict}
            state_dict.update(loadable)
            self.deep_model.load_state_dict(state_dict)

    # -------- focal (softmax) --------
    @staticmethod
    def focal_loss_softmax(logits, labels, gamma=2.0, ignore_index=255, alpha=None):
        probs = torch.softmax(logits, dim=1)  # [B,2,H,W]
        tgt = labels.clamp_min(0).clamp_max(1)
        oh = torch.nn.functional.one_hot(tgt, num_classes=2).permute(0,3,1,2).contiguous()
        valid = (labels != ignore_index)
        if valid.sum() == 0:
            return logits.new_zeros(())
        probs_v = probs.permute(0,2,3,1)[valid]
        oh_v    = oh.permute(0,2,3,1)[valid]
        pt = (probs_v * oh_v).sum(dim=1).clamp_min(1e-7)
        loss = ((1.0 - pt) ** gamma) * (-pt.log())
        if alpha is not None:
            alpha_v = (oh_v * alpha).sum(dim=1)
            loss = alpha_v * loss
        return loss.mean()

    def _build_eval_loader(self, split):
        """构建评估 DataLoader。split: 'val' 或 'test'"""
        assert split in ['val', 'test']
        if split == 'val':
            data_root = self.args.val_dataset_path if self.args.val_dataset_path else None
            name_list = self.args.val_data_name_list if self.args.val_data_name_list else None
            if (data_root is None) or (name_list is None):
                print("[WARN] No validation split provided. Falling back to TEST set for per-epoch validation.")
                data_root = self.args.test_dataset_path
                name_list = self.args.test_data_name_list
        else:
            data_root = self.args.test_dataset_path
            name_list = self.args.test_data_name_list

        if 'REDD' in self.args.dataset and self._redd_loader_for_eval is not None:
            loader_for_eval = self._redd_loader_for_eval
        else:
            loader_for_eval = None

        dataset = ChangeDetectionDatset(
            data_root,
            name_list,
            self.args.crop_size,
            None,
            'test',
            data_loader=loader_for_eval if loader_for_eval is not None else None
        )
        workers = getattr(self.args, 'num_workers', 4)
        dl = DataLoader(dataset, batch_size=1, num_workers=workers, drop_last=False)
        return dl

    @torch.no_grad()
    def evaluate(self, split, criterion_ce, lambda_lovasz, lambda_focal, gamma_focal, alpha_for_focal):
        """在指定 split 上计算指标 + 验证损失并打印细节。"""
        print(f'---------starting evaluation on {split.upper()}-----------')

        was_training = self.deep_model.training
        self.deep_model.eval()

        self.evaluator.reset()
        cm = np.zeros((2, 2), dtype=np.int64)

        data_loader = self._build_eval_loader(split)

        # 累计验证损失
        val_loss_sum = 0.0
        n_samples = 0

        torch.cuda.empty_cache()
        for _, data in enumerate(data_loader):
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs  = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda().float()
            labels           = labels.cuda().long()

            logits = self.deep_model(pre_change_imgs, post_change_imgs)  # [B,2,H,W]

            # === 验证损失（与训练一致）===
            ce_loss = criterion_ce(logits, labels)
            lovasz  = L.lovasz_softmax(torch.softmax(logits, dim=1), labels, ignore=255)
            focal   = self.focal_loss_softmax(logits, labels, gamma=gamma_focal, alpha=alpha_for_focal)
            final_loss = ce_loss + lambda_lovasz * lovasz + lambda_focal * focal
            val_loss_sum += float(final_loss)
            n_samples += 1

            # === 统计指标 ===
            pred_np   = np.asarray(torch.argmax(logits, dim=1).cpu().numpy(), dtype=np.int64)
            labels_np = np.asarray(labels.cpu().numpy(),                    dtype=np.int64)
            self.evaluator.add_batch(labels_np, pred_np)

            gt = labels_np.reshape(-1).astype(np.int64, copy=False)
            pr = pred_np.reshape(-1).astype(np.int64,  copy=False)
            valid = (gt != 255)
            if np.any(valid):
                gt_m = np.clip(gt[valid], 0, 1).astype(np.int64, copy=False)
                pr_m = np.clip(pr[valid], 0, 1).astype(np.int64, copy=False)
                k = (gt_m << 1) + pr_m
                k = np.asarray(k, dtype=np.intp)
                cm += np.bincount(k, minlength=4).reshape(2, 2)

        f1_score = self.evaluator.Pixel_F1_score()
        oa       = self.evaluator.Pixel_Accuracy()
        rec      = self.evaluator.Pixel_Recall_Rate()
        pre      = self.evaluator.Pixel_Precision_Rate()
        iou      = self.evaluator.Intersection_over_Union()
        kc       = self.evaluator.Kappa_coefficient()
        val_loss = val_loss_sum / max(1, n_samples)

        print(f'(on {split}) ValLoss={val_loss:.4f}, Recall {rec}, Precision {pre}, '
              f'OA {oa}, F1 {f1_score}, IoU {iou}, Kappa {kc}')
        print("Confusion Matrix (rows=GT, cols=Pred, classes=[0,1]):")
        print(cm)

        # 每类指标
        eps = 1e-7
        for cls in [0, 1]:
            TP = int(cm[cls, cls])
            FP = int(cm[:, cls].sum() - TP)
            FN = int(cm[cls, :].sum() - TP)
            precision_c = TP / (TP + FP + eps)
            recall_c    = TP / (TP + FN + eps)
            f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)
            iou_c       = TP / (TP + FP + FN + eps)
            print(f"[{split} | Class {cls}] Precision={precision_c:.6f} "
                  f"Recall={recall_c:.6f} F1={f1_c:.6f} IoU={iou_c:.6f}")

        if was_training:
            self.deep_model.train()

        return rec, pre, oa, f1_score, iou, kc, val_loss

    def training(self):
        """
        训练仅使用 Train；每个 epoch 在 Validation 上评估并基于 F1 保存最优；
        打印 Validation Loss；支持 Early Stopping（Val F1）。
        """
        print("steps_per_epoch =", len(self.train_data_loader))

        # ====== 类权重 & 损失 ======
        gt_dir = os.path.join(self.args.train_dataset_path, 'GT')
        w_bg, w_fg = estimate_class_weights(gt_dir, self.args.train_data_name_list, sample_k=300)
        w_fg = float(np.clip(w_fg, 1.0, float(self.args.w_fg_cap)))
        weight = torch.tensor([1.0, w_fg], dtype=torch.float32, device='cuda')
        criterion_ce = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=255)

        alpha_for_focal = weight
        lambda_lovasz = float(self.args.lambda_lovasz)
        lambda_focal  = float(self.args.lambda_focal)
        gamma_focal   = float(self.args.gamma_focal)

        # 先验初始化最后一层 bias
        p_fg = 1.0 / (1.0 + float(w_fg))
        init_fg_bias_2class(self.deep_model.main_clf, p_fg=p_fg)

        # ====== 训练日程 ======
        elem_num = len(self.train_data_loader)
        if elem_num == 0:
            raise RuntimeError("Empty train_data_loader.")

        if getattr(self.args, 'epochs_per_trial', 0) and self.args.epochs_per_trial > 0:
            num_epochs = int(self.args.epochs_per_trial)
            total_iters = num_epochs * elem_num
        else:
            total_iters = int(self.args.max_iters)
            num_epochs = max(1, total_iters // elem_num)

        warmup_ratio = float(getattr(self.args, 'warmup_ratio', 0.05))
        warmup_iters = max(1, int(warmup_ratio * total_iters))

        def lr_lambda(last_epoch: int):
            cur_iter = last_epoch
            if cur_iter < warmup_iters:
                return float(cur_iter + 1) / float(warmup_iters)
            denom = max(1, (total_iters - warmup_iters))
            progress = (cur_iter - warmup_iters) / denom
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        global_iter = int(getattr(self, 'resume_global_iter', 0))
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=global_iter-1)

        best_score = -1.0
        best_round = []
        best_val_loss = None  # 仅记录，不用于早停

        # Early Stopping
        patience = int(getattr(self.args, "early_stop_patience", 0))  # 0=disabled
        min_delta = float(getattr(self.args, "early_stop_min_delta", 0.0))
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            # ====== Train ======
            self.deep_model.train()
            loss_epoch = ce_epoch = lovasz_epoch = focal_epoch = 0.0

            for itera, data in enumerate(tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs  = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda().float()
                labels           = labels.cuda().long()

                self.optim.zero_grad(set_to_none=True)
                with autocast():
                    logits = self.deep_model(pre_change_imgs, post_change_imgs)
                    ce_loss = criterion_ce(logits, labels)
                    lovasz  = L.lovasz_softmax(torch.softmax(logits, dim=1), labels, ignore=255)
                    focal   = self.focal_loss_softmax(logits, labels, gamma=gamma_focal, alpha=alpha_for_focal)
                    final_loss = ce_loss + lambda_lovasz * lovasz + lambda_focal * focal

                self.scaler.scale(final_loss).backward()
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optim)
                self.scaler.update()
                if self.scaler.get_scale() >= prev_scale:
                    scheduler.step()

                loss_epoch   += float(final_loss)
                ce_epoch     += float(ce_loss)
                lovasz_epoch += float(lovasz)
                focal_epoch  += float(focal)
                global_iter  += 1

                if global_iter >= total_iters:
                    break

            n_batches = itera + 1
            try:
                cur_lr = scheduler.get_last_lr()[0]
            except Exception:
                cur_lr = self.optim.param_groups[0]['lr']

            print(f"[Epoch {epoch+1}] TrainLoss={loss_epoch/n_batches:.4f} "
                  f"(CE={ce_epoch/n_batches:.4f}, Lovasz={lovasz_epoch/n_batches:.4f}, Focal={focal_epoch/n_batches:.4f}) "
                  f"| lr={cur_lr:.2e}")

            # ====== Validation（含 ValLoss）======
            rec, pre, oa, f1_score, iou, kc, val_loss = self.evaluate(
                split='val',
                criterion_ce=criterion_ce,
                lambda_lovasz=lambda_lovasz,
                lambda_focal=lambda_focal,
                gamma_focal=gamma_focal,
                alpha_for_focal=alpha_for_focal
            )
            print(f"[Epoch {epoch+1}] ValLoss={val_loss:.4f} | F1={f1_score:.4f} IoU={iou:.4f}")

            # ====== Save best by F1 + Early Stop ======
            improved = (f1_score - best_score) > min_delta
            if improved:
                os.makedirs(self.model_save_path, exist_ok=True)
                ckpt_name = f'best_F1{f1_score:.4f}_IoU{iou:.4f}_epoch{epoch+1}.pth'
                best_path = os.path.join(self.model_save_path, ckpt_name)
                torch.save(self.deep_model.state_dict(), best_path)
                link_path = os.path.join(self.model_save_path, 'best.pth')
                try:
                    if os.path.islink(link_path) or os.path.exists(link_path):
                        os.remove(link_path)
                    os.symlink(best_path, link_path)
                except Exception:
                    import shutil; shutil.copy2(best_path, link_path)

                best_score = float(f1_score)
                best_round = [rec, pre, oa, f1_score, iou, kc]
                best_val_loss = val_loss
                no_improve_epochs = 0
                print(f"[best@epoch{epoch+1}] (VAL) F1={f1_score:.4f} IoU={iou:.4f} | ValLoss={val_loss:.4f} (model saved)")
            else:
                no_improve_epochs += 1
                print(f"[early-stop] no improvement on VAL F1 for {no_improve_epochs}/{patience} epoch(s) "
                      f"(best={best_score:.4f}, current={f1_score:.4f}, min_delta={min_delta:.4f})")

            if patience > 0 and no_improve_epochs >= patience:
                print(f"[early-stop] Stop training at epoch {epoch+1} due to no improvement on VAL F1.")
                break

            if global_iter >= total_iters:
                break

        print("Best on VAL (rec, pre, oa, f1, iou, kappa):", best_round)
        if best_val_loss is not None:
            print(f"Best VAL Loss (at best-F1 checkpoint): {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD/REDD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')

    # --- Train ---
    parser.add_argument('--train_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/train')
    parser.add_argument('--train_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/train_list.txt')

    # --- Validation ---
    parser.add_argument('--val_dataset_path', type=str, default=None)
    parser.add_argument('--val_data_list_path', type=str, default=None)

    # --- Test (最终报告用，训练期不使用) ---
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test')

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='ChangeMambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # === 复合损失 & 调度 ===
    parser.add_argument('--lambda_lovasz', type=float, default=0.75)
    parser.add_argument('--lambda_focal', type=float, default=0.5)
    parser.add_argument('--gamma_focal', type=float, default=2.0)
    parser.add_argument('--w_fg_cap', type=float, default=20.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)

    # === Optuna / Epoch 控制 ===
    parser.add_argument('--epochs_per_trial', type=int, default=0)

    # === Dataloader Workers ===
    parser.add_argument('--num_workers', type=int, default=4)

    # === Early Stopping ===
    parser.add_argument('--early_stop_patience', type=int, default=0, help="0 disables early stopping")
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help="minimum F1 improvement to reset patience")

    args = parser.parse_args()

    # 读取 train 名单
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    # 读取 val 名单（可选）
    if args.val_data_list_path is not None and os.path.isfile(args.val_data_list_path):
        with open(args.val_data_list_path, "r") as f:
            val_list = [n.strip() for n in f]
        args.val_data_name_list = val_list
    else:
        args.val_data_name_list = None  # 会在 Trainer 内回退到 test 做临时验证

    # 读取 test 名单
    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()

    # （可选）训练结束后再单独跑一次 TEST：
    # rec, pre, oa, f1, iou, kc, val_loss_dummy = trainer.evaluate(
    #     split='test',
    #     criterion_ce=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0],device='cuda'), ignore_index=255),
    #     lambda_lovasz=args.lambda_lovasz,
    #     lambda_focal=0.0,
    #     gamma_focal=args.gamma_focal,
    #     alpha_for_focal=None
    # )


if __name__ == "__main__":
    main()
