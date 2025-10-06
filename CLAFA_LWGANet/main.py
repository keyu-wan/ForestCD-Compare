import torch
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter
import os
import numpy as np
import random
import argparse
from thop import profile
import torch.nn as nn  # 新增：用于识别/操作 BN


def freeze_backbone_bn(model):

    backbone = getattr(model.detector, "backbone", None)
    if backbone is None:
        print("model.detector.backbone not found")
        return
    frozen = 0
    for _, m in backbone.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.affine:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
            m.train(False)
            frozen += 1
    print(f"frozen backbone BN: {frozen}")

@torch.no_grad()
def snapshot_bn_buffers(model):
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.track_running_stats:
            snap[name] = (m.running_mean.clone(), m.running_var.clone())
    return snap

def diff_bn_buffers(before, after, atol=0.0):
    changed = []
    for name, (bm, bv) in before.items():
        am, av = after.get(name, (None, None))
        if am is None:
            continue
        if not torch.allclose(bm, am, atol=atol, rtol=0.0) or not torch.allclose(bv, av, atol=atol, rtol=0.0):
            changed.append(name)
    return changed


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='lwganet_l0_e200')
        self.parser.add_argument('--backbone', type=str, default='lwganet_l0',
                                 help='lwganet_l0 | lwganet_l2 | lwganet_l1 | mobilenetv2 | resnet18d')
        self.parser.add_argument('--dataroot', type=str, default='/dataset/CD')
        self.parser.add_argument('--dataset', type=str, default='SYSU_256',
                                 help='LEVIR_256_split | WHU_256 | CDD_256 | SYSU_256 | REDDCD | BrazilCD')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
        self.parser.add_argument('--load_pretrain', type=bool, default=True)

        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--input_size', type=int, default=256)
        self.parser.add_argument('--fpn', type=str, default='fpn')
        self.parser.add_argument('--fpn_channels', type=int, default=128)
        self.parser.add_argument('--deform_groups', type=int, default=4)
        self.parser.add_argument('--gamma_mode', type=str, default='SE')
        self.parser.add_argument('--beta_mode', type=str, default='contextgatedconv')
        self.parser.add_argument('--num_heads', type=int, default=1)
        self.parser.add_argument('--num_points', type=int, default=8)
        self.parser.add_argument('--kernel_layers', type=int, default=1)
        self.parser.add_argument('--init_type', type=str, default='kaiming_normal')
        self.parser.add_argument('--alpha', type=float, default=0.25)
        self.parser.add_argument('--gamma', type=int, default=4, help='gamma for Focal loss')
        self.parser.add_argument('--dropout_rate', type=float, default=0.1)

        self.parser.add_argument('--focal_weight', type=float, default=1.0)
        self.parser.add_argument('--dice_weight', type=float, default=1.0)

        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--num_epochs', type=int, default=200)
        self.parser.add_argument('--warmup_epochs', type=int, default=20)
        self.parser.add_argument('--num_workers', type=int, default=4, help='#threads for loading data')
        self.parser.add_argument('--lr', type=float, default=5e-4)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)

        self.parser.add_argument('--freeze_backbone_bn', action='store_true', help='freeze BatchNorm in backbone')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        print(f"Trainval init with: alpha={self.opt.alpha}, focal={self.opt.focal_weight}, dice={self.opt.dice_weight}, lr={self.opt.lr}")

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = 'val'
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = 'train'

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular
        if self.opt.freeze_backbone_bn:
            freeze_backbone_bn(self.model)

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.best_epoch = 0

    def train(self):
        tbar = tqdm(self.train_data, ncols=80)
        self.opt.phase = 'train'
        _loss = _focal_loss = _dice_loss = 0.0
        _p2_loss = _p3_loss = _p4_loss = _p5_loss = 0.0

        bn_before = snapshot_bn_buffers(self.model) if self.opt.freeze_backbone_bn else None

        for i, data in enumerate(tbar):
            self.model.detector.train()

            focal, dice, p2_loss, p3_loss, p4_loss, p5_loss = self.model(
                data['img1'].cuda(), data['img2'].cuda(), data['cd_label'].cuda()
            )
            loss = focal + dice + p3_loss + p4_loss + p5_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.schedular.step()

            if i == 0 and bn_before is not None:
                bn_after = snapshot_bn_buffers(self.model)
                changed = diff_bn_buffers(bn_before, bn_after)
                if len(changed) == 0:
                    print("freeze success")
                else:
                    print("BN running stats changed：", changed[:10])
                bn_before = None 

            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            _p2_loss += p2_loss.item()
            _p3_loss += p3_loss.item()
            _p4_loss += p4_loss.item()
            _p5_loss += p5_loss.item()

            tbar.set_description(
                "Loss: %.3f, Focal: %.3f, Dice: %.3f, LR: %.6f" %
                (_loss / (i + 1), _focal_loss / (i + 1), _dice_loss / (i + 1),
                 self.optimizer.param_groups[0]['lr'])
            )

        n = len(tbar)
        return (_loss / n, _focal_loss / n, _dice_loss / n,
                _p2_loss / n, _p3_loss / n, _p4_loss / n, _p5_loss / n)

    
    def val(self):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        self.opt.phase = 'val'
        self.model.eval()

        _v_loss = _v_focal = _v_dice = _v_p2 = _v_p3 = _v_p4 = _v_p5 = 0.0

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                vfocal, vdice, vp2, vp3, vp4, vp5 = self.model(
                    _data['img1'].cuda(), _data['img2'].cuda(), _data['cd_label'].cuda()
                )
                vloss = vfocal + vdice + vp3 + vp4 + vp5
                _v_loss += vloss.item()
                _v_focal += vfocal.item()
                _v_dice += vdice.item()
                _v_p2 += vp2.item()
                _v_p3 += vp3.item()
                _v_p4 += vp4.item()
                _v_p5 += vp5.item()

                # metric 路径
                val_pred = self.model.inference(_data['img1'].cuda(), _data['img2'].cuda())
                val_target = _data['cd_label'].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())

            n = len(tbar)
            val_scores = self.running_metric.get_scores()
            val_losses = {
                "val_loss": _v_loss / n,
                "val_focal": _v_focal / n,
                "val_dice": _v_dice / n,
                "val_p2": _v_p2 / n,
                "val_p3": _v_p3 / n,
                "val_p4": _v_p4 / n,
                "val_p5": _v_p5 / n,
            }

            message = f"(phase: {self.opt.phase}) "
            for k, v in val_scores.items():
                message += f"{k}: {v*100:.3f} "
            message += " | " + ", ".join([f"{k}={v:.3f}" for k, v in val_losses.items()])
            print(message)

        if val_scores['mf1'] >= self.previous_best:
            name = self.opt.dataset + '_' + self.opt.name
            print("name", name)
            self.model.save(name, self.opt.backbone)
            self.previous_best = val_scores['mf1']
            self.best_epoch = self.current_epoch

        return val_scores, val_losses



if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=1)

    for epoch in range(1, opt.num_epochs + 1):
        name = opt.dataset + '_' + opt.name
        print("name", name)
        print("\n==> Name %s, Epoch %i, previous best = %.3f" % (name, epoch, trainval.previous_best * 100))
        trainval.current_epoch = epoch

        train_loss, focal_loss, dice_loss, p2_loss, p3_loss, p4_loss, p5_loss = trainval.train()
        val_scores, val_losses = trainval.val()

        print(
            f"[Epoch {epoch}] "
            f"Train -> total={train_loss:.3f}, focal={focal_loss:.3f}, dice={dice_loss:.3f}, "
            f"p2={p2_loss:.3f}, p3={p3_loss:.3f}, p4={p4_loss:.3f}, p5={p5_loss:.3f} | "
            f"Val -> total={val_losses['val_loss']:.3f}, focal={val_losses['val_focal']:.3f}, "
            f"dice={val_losses['val_dice']:.3f}, p2={val_losses['val_p2']:.3f}, "
            f"p3={val_losses['val_p3']:.3f}, p4={val_losses['val_p4']:.3f}, p5={val_losses['val_p5']:.3f} | "
            f"mf1={val_scores['mf1']*100:.2f}, miou={val_scores['miou']*100:.2f}, "
            f"lr={trainval.optimizer.param_groups[0]['lr']:.6f}"
        )

    print(f"\n✅ Best epoch = {trainval.best_epoch}, best mf1 = {trainval.previous_best * 100:.3f}%")

    opt.phase = 'test'
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data, ncols=80)
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
            val_target = _data['cd_label'].detach()
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            _ = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
        val_scores = running_metric.get_scores()
        message = '(phase: %s) ' % (opt.phase)
        for k, v in val_scores.items():
            message += '%s: %.3f ' % (k, v * 100)
        print('test: \n')
        print('model_name: {},\n dataset: {},\n message: {}\n'.format(opt.name, opt.dataset, message))
