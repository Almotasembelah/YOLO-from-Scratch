import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
import random

class BusTruckDataset(Dataset):
    """
    Dataset class for Bus and Truck images with bounding box annotations.
    Supports optional Albumentations transforms including Mosaic augmentation.

    Bounding boxes are returned in YOLO format: [cx, cy, w, h], normalized (0 to 1).

    Args:
        df (pd.DataFrame): DataFrame with columns ['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax'].
        index (tuple): Tuple with (start_frac, end_frac) to slice dataset, e.g., (0, 0.8).
        img_dir (str): Directory where images are stored.
        transform (albumentations.Compose, optional): Albumentations transform to apply.
    """
    
    decode_labels = {0: 'Bus', 1: 'Truck'}
    Transform = None  # Optional transform applied to new images during inference or evaluation

    def __init__(self, df, index, img_dir='dataset/images/images', transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.unique_images = self.df['ImageID'].unique()[int(len(df['ImageID'].unique())*index[0]):int(len(df['ImageID'].unique())*index[1])]
        self.df = df[df['ImageID'].isin(self.unique_images)]
        self.labels = {'Bus': 0, 'Truck': 1}
        self.instances = {
            'Bus': len(self.df[self.df['LabelName'] == 'Bus']),
            'Truck': len(self.df[self.df['LabelName'] == 'Truck'])
        }

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        # 1. Read the primary image and its boxes
        img_id = self.unique_images[idx]
        df = self.df[self.df['ImageID'] == img_id]
        image = Image.open(f'{self.img_dir}/{img_id}.jpg').convert('RGB')
        image = np.array(image)  # Convert to NumPy array

        boxes = df['LabelName,XMin,YMin,XMax,YMax'.split(',')].values
        yolo_boxes = []
        labels = []
        for label, xmin, ymin, xmax, ymax in boxes:
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            yolo_boxes.append([cx, cy, w, h])  # Normalized YOLO coordinates
            labels.append(self.labels[label])

        # 2. Prepare for Mosaic if used in transform
        if self.transform and any(isinstance(t, A.Mosaic) for t in self.transform.transforms):
            # Sample up to 3 additional images
            num_additional = min(6, len(self.unique_images) - 1)
            additional_indices = random.sample(range(len(self.unique_images)), num_additional)
            mosaic_metadata = []

            # Load additional images and prepare metadata
            for add_idx in additional_indices:
                add_img_id = self.unique_images[add_idx]
                add_df = self.df[self.df['ImageID'] == add_img_id]
                add_image = Image.open(f'{self.img_dir}/{add_img_id}.jpg').convert('RGB')
                add_image = np.array(add_image)  # Convert to NumPy array
                add_boxes = add_df['LabelName,XMin,YMin,XMax,YMax'.split(',')].values
                add_yolo_boxes = []
                add_labels = []
                for label, xmin, ymin, xmax, ymax in add_boxes:
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    add_yolo_boxes.append([cx, cy, w, h])
                    add_labels.append(self.labels[label])
                # Add to mosaic_metadata
                mosaic_metadata.append({
                    'image': add_image,
                    'bboxes': add_yolo_boxes,
                    'labels': add_labels
                })

            # Pad mosaic_metadata with primary image if fewer than 3 additional images
            while len(mosaic_metadata) < 3:
                mosaic_metadata.append({
                    'image': image,
                    'bboxes': yolo_boxes,
                    'labels': labels
                })

            # Apply transformation with mosaic_metadata
            transformed = self.transform(
                image=image,
                bboxes=yolo_boxes,
                labels=labels,
                mosaic_metadata=mosaic_metadata
            )
        else:
            # Apply regular transformations (including CLAHE)
            transformed = self.transform(
                image=image,
                bboxes=yolo_boxes,
                labels=labels
            ) if self.transform else {'image': image, 'bboxes': yolo_boxes, 'labels': labels}

        # Extract transformed data
        image = transformed['image']
        yolo_boxes = transformed['bboxes']
        labels = transformed['labels']

        # 3. Convert to [img_idx, label, cx, cy, w, h] format
        if yolo_boxes:
            boxes = [[idx * 1.0, label, *box] for box, label in zip(yolo_boxes, labels)]
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 6), dtype=torch.float32)

        # 4. Convert image to torch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Shape: [C, H, W], normalize to [0, 1]

        return image, boxes

    @staticmethod
    def collate_fn(batch):
        imgs, bxs = [], []
        for img, boxes in batch:
            imgs.append(img)
            for bb in boxes:
                bxs.append(bb)
        return torch.stack(imgs), torch.stack(bxs)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"num_images={len(self)}, "
                f"num_bus_instances={self.instances['Bus']}, "
                f"num_truck_instances={self.instances['Truck']}, "
                f"transform={self.transform})")