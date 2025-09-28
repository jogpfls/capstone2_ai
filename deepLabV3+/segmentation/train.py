import os
import time
import math
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset_coco import CocoSegmentationDataset, split_image_ids
from pycocotools.coco import COCO


def create_model(num_classes: int, backbone: str = "resnet50", pretrained: bool = True, aux_loss: bool = False) -> nn.Module:
    if backbone == "resnet50":
        model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            aux_loss=aux_loss,
        )
    else:
        model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            aux_loss=aux_loss,
        )
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model


def compute_miou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        ious = []
        for cls in range(1, num_classes):  # exclude background idx 0
            pred_i = preds == cls
            targ_i = targets == cls
            inter = (pred_i & targ_i).sum().item()
            union = (pred_i | targ_i).sum().item()
            if union == 0:
                continue
            ious.append(inter / union)
        if not ious:
            return 0.0
        return float(sum(ious) / len(ious))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_miou = 0.0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_miou += compute_miou(outputs.detach(), masks, model.classifier[-1].out_channels)
    return running_loss / len(loader.dataset), running_miou / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            total_miou += compute_miou(outputs, masks, model.classifier[-1].out_channels)
    return total_loss / len(loader.dataset), total_miou / len(loader)


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    images_dir = cfg['dataset']['images_dir']
    ann_path = cfg['dataset']['annotations']
    train_ratio = float(cfg['dataset']['train_split'])
    seed = int(cfg['dataset']['random_seed'])
    cat2train = {k: int(v) for k, v in cfg['classes']['category_id_to_train_id'].items()}

    num_classes = int(cfg['training']['num_classes'])
    img_size = int(cfg['training']['image_size'])
    batch_size = int(cfg['training']['batch_size'])
    num_workers = int(cfg['training']['num_workers'])
    epochs = int(cfg['training']['epochs'])
    lr = float(cfg['training']['lr'])
    weight_decay = float(cfg['training']['weight_decay'])
    amp = bool(cfg['training']['amp'])

    checkpoint_dir = cfg['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    coco = COCO(ann_path)
    train_ids, val_ids = split_image_ids(coco, train_ratio=train_ratio, seed=seed)

    train_ds = CocoSegmentationDataset(images_dir, ann_path, category_id_to_train_id=cat2train, image_ids=train_ids, image_size=img_size)
    val_ds = CocoSegmentationDataset(images_dir, ann_path, category_id_to_train_id=cat2train, image_ids=val_ids, image_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = create_model(num_classes=num_classes,
                         backbone=cfg['model']['backbone'],
                         pretrained=bool(cfg['model']['pretrained']),
                         aux_loss=bool(cfg['model']['aux_loss']))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == 'cuda')) if hasattr(torch.cuda, 'amp') else None

    best_miou = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_miou = evaluate(model, val_loader, criterion, device)

        ckpt_path = os.path.join(checkpoint_dir, f"deeplabv3p_epoch{epoch:03d}_miou{val_miou:.4f}.pt")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_miou': val_miou,
        }, ckpt_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} miou: {train_miou:.4f} | val_loss: {val_loss:.4f} miou: {val_miou:.4f}")

    if bool(cfg['training'].get('run_infer_after_train', False)):
        from infer import main as infer_main
        infer_main()


if __name__ == "__main__":
    main()
