import torch
from torch.cuda.amp import autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, StepLR
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm.autonotebook import tqdm

from IPython.display import clear_output, display, HTML
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert
from utils import nms, draw_boxes
from dataset import BusTruckDataset
from PIL import Image
import os
from torchvision.transforms.functional import to_pil_image

from utils import speed

class ModelManager:
    """
    A utility class for training, validating, and managing PyTorch models.
    Supports learning rate scheduling, checkpointing, and visualization of training progress.

    Inspired by:
        https://github.com/dvgodoy/PyTorchStepByStep/blob/master/stepbystep/v4.py
    """

    def __init__(self, model, optimizer, loss_fn, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the ModelManager instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            loss_fn (torch.nn.Module or list): The loss function.
            device (str, optional): The device to use ('cuda' or 'cpu'). Defaults to GPU if available.
        """
        self.device = device

        self.model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        self.model.to(self.device)

        self._train_losses = []
        self._val_losses = []
        self.lrs = set()

        self._train_data = None
        self._val_data = None

        self._train_step = self._train_step_fn()
        self._val_step = self._val_step_fn()
        self._scaler = torch.cuda.amp.GradScaler()

        self._total_epochs = 0

        self._lr_scheduler = None
        self._STEP_SCHEDULER = False
        self._BATCH_SCHEDULER = False
        

        self._filename = None
        self.names = ['bbox', 'class', 'obj']

        self._metric = MeanAveragePrecision(box_format='cxcywh')
        self._train_metric = MeanAveragePrecision(box_format='cxcywh')
        self.map_val = []
        self.map50_val = []

        # These values are used by the Ultralytics YOLO implementation
        self.iou_thresh = 0.7
        self.conf_thresh = 0.001

        self._i = 0

        # Number of iterations to accumulate gradients before updating weights.
        # In this case, weights will be updated every 2 iterations.
        self._iter_to_accumulate = 2  

        self.nc = 2  # number of classes

        # Learning rate factor (lrf), as defined by Ultralytics.
        # It is computed based on the number of classes (nc)
        self.lrf = round(0.002 * 5 / (4 + self.nc), 6)

        self._warmup_epochs = 3
        self.epochs = 0

        runs_dir = "runs"
        self.run_num = 0  if not os.path.exists(runs_dir) else max([int(f.path.split('\\')[-1]) for f in os.scandir(runs_dir) if f.is_dir()])+1

    def _train_step_fn(self):
        """
        Defines the training step for a single batch, including warmup procedure.

        Returns:
            function: A function that computes the loss and accuracy for a batch.
        """

        def _step(x, y):
            self.model.train()
            x, y = x.to(self.device), y.to(self.device)

            nb = len(self._train_data)  # Number of batches per epoch
            nw = max(round(self._warmup_epochs * nb), 100)  # Total number of warmup batches across all warmup epochs
            ni = self._i + nb * (self._total_epochs) 

            # Warmup procedure
            # The learning rate increases over the warmup batches,
            # starting from 0.0 and gradually reaching the final value defined by lrf
            if ni <= nw:
                lf = lambda x: max(1 - x/self.epochs , 0) * (1.0 - self.lrf) + self.lrf
                xi = [0, nw]
                for j, x_group in enumerate(self._optimizer.param_groups):
                    x_group["lr"] = np.interp(
                        ni, xi, [0.0, x_group["initial_lr"] * lf(self._total_epochs+1)]
                    )

            with autocast(enabled=True, dtype=torch.float16):
                y_pred = self.model(x)
                loss, loss_items = self._loss_fn(y_pred, y)

            loss = loss.sum()
            self._scaler.scale(loss).backward()

            # Update the weights after every self._iter_to_accumulate iterations.
            # This simulates training with a larger batch size. For example, with bs=16 and iter_to_accumulate=2,
            # the model behaves as if the effective batch size is 32.
            # This is useful when memory limitations prevent using large batch sizes.
            if (self._i+1) % self._iter_to_accumulate == 0 or self._i == len(self._train_data) - 1:
                self._scaler.unscale_(self._optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()

                if self._BATCH_SCHEDULER:
                    self._lr_scheduler.step()

            del x, y, y_pred
            return loss_items
        return _step

    def _val_step_fn(self):
        """
        Defines the validation step for a single batch.

        Returns:
            function: A function that computes the loss and accuracy for a batch.
        """
        def _step(x, y):
            self.model.eval()
            x = x.to(self.device)
            y = y.to(self.device)
            with autocast(enabled=True, dtype=torch.float16):
                y_pred = self.model(x)
                loss, loss_items = self._loss_fn(y_pred, y)
                self._calc_map(x, y, y_pred=y_pred)

            del y, x, y_pred
            return loss_items
        return _step

    def _mini_batch(self, validation=False):
        """
        Processes a mini-batch of data (training or validation).

        Args:
            validation (bool, optional): Whether to use the validation dataset. Defaults to False.

        Returns:
            tuple: Average loss and task-specific metrics (for multi-task learning).
        """
        if validation:
            dataloader = self._val_data
            step_fn = self._val_step
            description = "Validation"
            c = 'green'
        else:
            dataloader = self._train_data
            step_fn = self._train_step
            description = "Training"
            c = 'blue'

        tloss = 0

        # Initialize tqdm progress bar and store it in a variable
        progress_bar = tqdm(dataloader, desc=description, position=0, leave=False, colour=c)

        for i, (images, y) in enumerate(progress_bar):
            self._i = i
            loss_batch = step_fn(images, y)
            tloss = (
                (tloss * i + loss_batch) / (i + 1) if tloss is not None else loss_batch
            )

            # Update the description on the progress bar instance
            progress_bar.set_description(f'{description} | Loss {tloss.sum().item():.4f} | LR {self._optimizer.param_groups[0]["lr"]:.8f}')
        del images, y
        return tloss # the mean of the loss

    def set_lr_scheduler(self, scheduler):
        """
        Sets the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        """
        if scheduler.optimizer != self._optimizer:
            raise ValueError('Optimizer is not used in lr_scheduler')
        self._lr_scheduler = scheduler
        if isinstance(scheduler, StepLR) or \
                isinstance(scheduler, MultiStepLR) or \
                isinstance(scheduler, ReduceLROnPlateau) or \
                isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingLR) or \
                isinstance(scheduler,torch.optim.lr_scheduler.LambdaLR):
            self._STEP_SCHEDULER = True
            self._BATCH_SCHEDULER = False
        elif isinstance(scheduler, CyclicLR) or isinstance(scheduler, OneCycleLR):
            self._BATCH_SCHEDULER = True
            self._STEP_SCHEDULER = False

    def train(self, epochs, seed=42, display_table=True):
        """
        Trains the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            display_table (bool, optional): Whether to display metrics after each epoch. Defaults to True.
        """
        self._set_seed(seed)
        last_loss = None

        if display_table:
            # Initialize a DataFrame to store training metrics
            columns = ['Epoch', "Training Loss", "Validation Loss"]
            columns.extend([f'{task} Loss (Val)' for task in self.names])
            columns.append('val mAP50-95')
            columns.append('val mAP50')
            if self._STEP_SCHEDULER:
                columns.append('Learning Rate')
            metrics_df = pd.DataFrame(columns=columns)

        self.epochs = epochs
        self._optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        for epoch in tqdm(range(epochs), desc="Training Progress", position=1, leave=False, colour='blue'):
            try:
                # Perform training step
                loss = self._mini_batch()
                self._train_losses.append(loss)

                # Perform validation step
                with torch.no_grad():
                    val_loss = self._mini_batch(validation=True)
                    self._val_losses.append(val_loss)

                # Step the learning rate scheduler if applicable
                if self._STEP_SCHEDULER and self._total_epochs >= self._warmup_epochs-1:
                    if isinstance(self._lr_scheduler, ReduceLROnPlateau):
                        self._lr_scheduler.step(val_loss.sum())
                    else:
                        self._lr_scheduler.step()

                # Save the best model checkpoint
                if last_loss is None or last_loss > val_loss.sum():
                    last_loss = val_loss.sum()
                    if self._filename:
                        self.save_checkpoint(f'best_{self._filename}')
                    else:
                        self.save_checkpoint('best')

                self._total_epochs += 1

                # Optionally display table with results after each epoch
                if display_table:
                    # Update the metrics DataFrame
                    new_row = {
                        "Epoch": self._total_epochs,
                        "Training Loss": round(loss.sum().item(), 4),
                        "Validation Loss": round(val_loss.sum().item(), 4),
                    }
                    idx = 2
                    for i in range(len(self.names)):
                        idx += 1
                        new_row[columns[idx]] = f"{val_loss[i]:.4f}"

                    val_map = torch.tensor(self.map_val).mean()
                    self.map_val = []
                    new_row[columns[idx+1]] = f'{val_map.item():.5f}'

                    map50_val = torch.tensor(self.map50_val).mean()
                    self.map50_val = []
                    new_row[columns[idx+2]] = f'{map50_val.item():.5f}'
                    
                    if self._STEP_SCHEDULER:
                        new_row['Learning Rate'] = self._lr_scheduler.optimizer.param_groups[0]['lr']
                        
                    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
                    clear_output(wait=True)
                    display(HTML(metrics_df.to_html()))
            
            except KeyboardInterrupt as e:
                if self._filename:
                    self.save_checkpoint(f'last_{self._filename}')
                else:
                    self.save_checkpoint('last')
                # Free the memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ValueError('Training Interrupted by the User')
            except Exception as e:
                print('Fehler beim Training ', e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e

            # Save the most recent checkpoint
            if self._filename:
                self.save_checkpoint(f'last_{self._filename}')
            else:
                self.save_checkpoint('last')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    @speed
    def predict(self, img, iou_thresh=0.4, conf_thresh=0.5, plot=True):
        if isinstance(img, str):
            timg = Image.open(img).convert('RGB')
        
        if BusTruckDataset.Transform:
            timg = BusTruckDataset.Transform(timg)
        else:
            timg = transforms.ToTensor()(timg)

        self.model.eval()
        preds = self.model(timg.unsqueeze(0).to(self.device))
        preds = nms(preds, iou_thresh, conf_thresh)
        labels = [BusTruckDataset.decode_labels[int(label.item())] for label in preds[0][...,-1]]
        bxs = box_convert(preds[0][..., :4], 'xyxy', 'cxcywh')
        if plot:
            draw_boxes(timg, bxs, labels, scores=preds[0][..., 4], normalized=False)
        return [bxs, labels]
        
    def set_dataloaders(self, train_data, val_data=None):
        """
        Sets the training and validation DataLoaders.
        """
        self._train_data = train_data
        if val_data is not None:
            self._val_data = val_data

    def to(self, device):
        """
        Moves the model and optimizer to the specified device.
        """
        self.device = device

    def _set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed.
        """
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _calc_map(self, x, y, y_pred=None):
            if y_pred is None:
                self.model.eval() 
                y_pred = self.model(x)
            preds_ = nms(y_pred,iou_thresh=self.iou_thresh,conf_thresh=self.conf_thresh)

            bs, _, w, h = x.shape
            # Prepare data for calculating mAP
            mTarget = [{
                'boxes': (y[y[...,0] % bs == i][...,2:]).mul_(torch.tensor([w, h, w, h], device=self.device)),
                'labels': y[y[...,0] % bs == i][...,1].type(torch.int64)
            } for i in range(len(x))]

            mPreds = [{
                'boxes': box_convert(F.relu(preds_[i][..., :4]), 'xyxy', 'cxcywh'),
                'scores': preds_[i][..., 4],
                'labels': preds_[i][..., 5:].type(torch.int64).squeeze(1)
            } for i in range(len(x))]

            self._metric.update(mPreds, mTarget)
            result = self._metric.compute()
            self._metric.reset()
            self.map_val.append(result['map'].item())
            self.map50_val.append(result['map_50'].item())
            
            # print(result)
            self.model.train()
            del result

    def save_checkpoint(self, filename):
        """
        Saves the model checkpoint inside a runs folder with a numbered subfolder (e.g., runs/0, runs/1).

        Args:
            filename (str): The filename for the checkpoint.
        """
        import os

        # Create runs folder if it doesn't exist
        runs_dir = "runs"
        os.makedirs(runs_dir, exist_ok=True)

        # Create the numbered subfolder
        run_dir = os.path.join(runs_dir, str(self.run_num))
        os.makedirs(run_dir, exist_ok=True)

        # Full path for the checkpoint file
        checkpoint_path = os.path.join(run_dir, filename)

        checkpoint = {'epoch': self._total_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': self._train_losses,
                    'val_loss': self._val_losses,
                    'run_num': self.run_num}
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename):
        """
        Loads the model checkpoint.
        """
        checkpoint = torch.load(filename, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._total_epochs = checkpoint['epoch']
        self._train_losses = checkpoint['loss']
        self._val_losses = checkpoint['val_loss']
        self.run_num = checkpoint['run_num']
        self.model.train()

    def set_filename(self, filename):
        """
        Sets the filename prefix for saving checkpoints.
        """
        self._filename = filename

    def plot_losses(self):
        """
        Plots training and validation losses for bbox, class, and dfl separately.
        """
        import matplotlib.pyplot as plt

        # Detach and move to CPU
        train_losses = [loss.detach().cpu().numpy() for loss in self._train_losses]
        val_losses = [loss.detach().cpu().numpy() for loss in self._val_losses]

        # Split into separate loss components
        train_bbox = [l[0] for l in train_losses]
        train_class = [l[1] for l in train_losses]
        train_dfl = [l[2] for l in train_losses]

        val_bbox = [l[0] for l in val_losses]
        val_class = [l[1] for l in val_losses]
        val_dfl = [l[2] for l in val_losses]

        figs = []

        # Plot each loss type
        for title, train, val in zip(
            ['BBox Loss', 'Class Loss', 'DFL Loss'],
            [train_bbox, train_class, train_dfl],
            [val_bbox, val_class, val_dfl]
        ):
            fig = plt.figure(figsize=(8, 4))
            plt.plot(train, label='Training', c='b')
            plt.plot(val, label='Validation', c='r')
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            figs.append(fig)

        return figs  # List of figures
    
    def predict_val_batch(self):
        # initialize once if not already
        if not hasattr(self, "_val_batch_index"):
            self._val_batch_index = 0

        val_data = iter(self._val_data)
        
        def wrapper():
            self.model.eval()
            batch = next(val_data)
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)

            # Inference
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                preds = self.model(images)

            # Apply NMS
            preds_ = nms(preds, iou_thresh=0.7, conf_thresh=0.3, max_det=300)

            # Setup figure
            n_rows = batch_size
            fig, axes = plt.subplots(n_rows, 2, figsize=(8, 5 * n_rows))
            if n_rows == 1:
                axes = [axes]

            for i in range(batch_size):
                img = images[i].detach().cpu()
                img_draw = img.clone()
                img_pred = img.clone()

                # === Ground Truth ===
                global_idx = i + batch_size * self._val_batch_index
                t_mask = targets[..., 0] == global_idx
                true_boxes = targets[t_mask][:, 2:]
                true_labels = targets[t_mask][:, 1]
                true_labels_text = [self._train_data.dataset.decode_labels[int(label.item())] for label in true_labels]
                img_with_gt = draw_boxes(img_draw.cpu(), true_boxes.cpu(), labels=true_labels_text, show=False)

                # === Predictions ===
                if len(preds_[i]) > 0:
                    pred_boxes = box_convert(preds_[i][..., :4], 'xyxy', 'cxcywh')
                    pred_scores = preds_[i][:, 4]
                    pred_labels = preds_[i][:, 5]
                    pred_labels_text = [self._train_data.dataset.decode_labels[int(label.item())] for label in pred_labels]
                    img_with_pred = draw_boxes(img_pred.cpu(), pred_boxes.cpu(), labels=pred_labels_text, scores=pred_scores, normalized=0, show=False)
                else:
                    img_with_pred = img_pred.cpu()

                # Plot
                axes[i][0].imshow(to_pil_image(img_with_gt))
                axes[i][0].set_title("Ground Truth")
                axes[i][0].axis("off")

                axes[i][1].imshow(to_pil_image(img_with_pred))
                axes[i][1].set_title("Prediction")
                axes[i][1].axis("off")

            plt.tight_layout()
            plt.show()

            # Move to next batch
            self._val_batch_index += 1
            self.model.train()
        
        return wrapper
