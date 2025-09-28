import os
import yaml
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import models

from dataset_coco import CocoSegmentationDataset


PALETTE = {
    0: (0, 0, 0),            # background - black
    1: (0, 255, 0),          # multicopter_body - green
    2: (0, 0, 255),          # propeller - red (BGR order for cv2)
    3: (255, 255, 0),        # fixed_wing_body - cyan-ish
}


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = np.zeros_like(image)
    for k, bgr in PALETTE.items():
        color[mask == k] = bgr
    return cv2.addWeighted(image, 1 - alpha, color, alpha, 0)


def load_model(num_classes: int, backbone: str, ckpt_path: str) -> torch.nn.Module:
    if backbone == "resnet50":
        model = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
    else:
        model = models.segmentation.deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=True)
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    images_dir = cfg['dataset']['images_dir']
    ann_path = cfg['dataset']['annotations']
    cat2train = {int(k): int(v) for k, v in cfg['classes']['category_id_to_train_id'].items()}

    num_classes = int(cfg['training']['num_classes'])
    img_size = int(cfg['training']['image_size'])

    out_dir = cfg['training']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg['training']['checkpoint_dir'], 'best.pt')

    model = load_model(num_classes=num_classes, backbone=cfg['model']['backbone'], ckpt_path=ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    ds = CocoSegmentationDataset(images_dir, ann_path, category_id_to_train_id=cfg['classes']['category_id_to_train_id'], image_ids=None, image_size=img_size)

    max_samples = int(cfg['eval']['max_samples'])

    with torch.no_grad():
        for i in range(min(len(ds), max_samples)):
            img_t, mask_t, meta = ds[i]
            img = (img_t * 255).byte().permute(1, 2, 0).numpy()[:, :, ::-1]  # to BGR for cv2
            x = img_t.unsqueeze(0).to(device)
            logits = model(x)['out']
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            over = overlay_mask(img, pred, alpha=0.5)

            base = os.path.splitext(os.path.basename(meta['file_name']))[0]
            cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), over)
            cv2.imwrite(os.path.join(out_dir, f"{base}_pred.png"), pred)


if __name__ == "__main__":
    main()
