from .network import Detector
import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from .block.schedular import get_cosine_schedule_with_warmup
from .loss.focal import FocalLoss
from .loss.dice import WeightedDiceLoss

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if torch.cuda.is_available() and opt.gpu_ids else "cpu")
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.detector = Detector(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            num_heads=opt.num_heads,
            num_points=opt.num_points,
            kernel_layers=opt.kernel_layers,
            dropout_rate=opt.dropout_rate,
            init_type=opt.init_type
        )

        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = WeightedDiceLoss(class_weights=[0.2, 0.8])

        self.optimizer = optim.AdamW(self.detector.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.schedular = get_cosine_schedule_with_warmup(
            self.optimizer,
            # Here, 625 = #training images // batch_size.
            num_warmup_steps=43 * opt.warmup_epochs,
            num_training_steps=43 * opt.num_epochs
        )

        if opt.load_pretrain:
            ckpt_path = os.path.join(self.save_dir, f"{opt.dataset}_{opt.name}_{opt.backbone}_best.pth")
            if os.path.exists(ckpt_path):
                self.load_ckpt(self.detector, self.optimizer, ckpt_path)
            else:
                print(f"⚠️ No full checkpoint found at {ckpt_path}, using backbone pretrained weights instead.")
        else:
            print("Training from scratch.")

        self.detector.to(self.device)
        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label):
        pred, pred_p2, pred_p3, pred_p4, pred_p5 = self.detector(x1, x2)
        label = label.long()

        # Loss weights
        focal_weight = self.opt.focal_weight
        dice_weight = self.opt.dice_weight

        focal = self.focal(pred, label)
        dice = self.dice(pred, label)

        # Downsample label for multi-scale loss
        label_ds = F.interpolate(label.unsqueeze(1).float(), size=pred_p2.shape[2:], mode='nearest').squeeze(1).long()
        p2_loss = self.focal(pred_p2, label_ds) * 0.5 + self.dice(pred_p2, label_ds)
        p3_loss = self.focal(pred_p3, label_ds) * 0.5 + self.dice(pred_p3, label_ds)
        p4_loss = self.focal(pred_p4, label_ds) * 0.5 + self.dice(pred_p4, label_ds)
        p5_loss = self.focal(pred_p5, label_ds) * 0.5 + self.dice(pred_p5, label_ds)

        return focal * focal_weight, dice * dice_weight, p2_loss, p3_loss, p4_loss, p5_loss

    def inference(self, x1, x2):
        self.detector.eval()
        with torch.no_grad():
            pred, _, _, _, _ = self.detector(x1, x2)
        return pred

    def load_ckpt(self, network, optimizer, ckpt_path):
        print(f"📥 Loading model checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        missing_keys, unexpected_keys = network.load_state_dict(checkpoint['network'], strict=False)
        print("✅ Weights loaded.")
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_path = os.path.join(self.save_dir, f"{model_name}_{backbone}_best.pth")
        torch.save({'network': network.cpu().state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
        print(f"💾 Model saved at {save_path}")
        network.to(self.device)

    def save(self, model_name, backbone):
        self.save_ckpt(self.detector, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print(f"model [{model.name()}] was created")
    return model.to(model.device)
