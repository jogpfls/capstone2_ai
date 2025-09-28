import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


class CocoSegmentationDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 annotation_file: str,
                 category_id_to_train_id: Dict[str, int],
                 image_ids: List[int] | None = None,
                 image_size: int | None = None) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.category_id_to_train_id = {int(k): v for k, v in category_id_to_train_id.items()}
        self.image_size = image_size

        if image_ids is None:
            self.img_ids = list(self.coco.imgs.keys())
        else:
            self.img_ids = image_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_image_original(self, img_info: Dict[str, Any]) -> Image.Image:
        path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(path).convert("RGB")
        return img

    def _ann_to_mask(self, anns: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.category_id_to_train_id:
                continue
            train_id = self.category_id_to_train_id[cat_id]
            segm = ann["segmentation"]
            if isinstance(segm, list):
                rle = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rle)
            elif isinstance(segm, dict) and "counts" in segm:
                rle = segm
            else:
                continue
            m = maskUtils.decode(rle)
            mask[m.astype(bool)] = train_id
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img = self._load_image_original(img_info)
        orig_w, orig_h = img.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id], iscrowd=False))
        mask = self._ann_to_mask(anns, orig_h, orig_w)

        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = np.array(Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST))

        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.astype(np.int64))

        meta = {"image_id": img_id, "file_name": img_info["file_name"]}
        return img_t, mask_t, meta


def split_image_ids(coco: COCO, train_ratio: float = 0.85, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    ids = np.array(list(coco.imgs.keys()))
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return ids[:n_train].tolist(), ids[n_train:].tolist()
