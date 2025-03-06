import os
import numpy as np
from PIL import Image
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, num_local_conditions=3):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.num_local_conditions = num_local_conditions

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """
        Save images to disk. We handle:
          - local_control as a special case (split into each condition)
          - everything else as default.
        """
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            if k == 'local_control':
                # shape [B, 3*num_local_conditions, H, W]
                _, _, h, w = images[k].shape
                if h == w == 1:
                    # Means no real local conditions, skip
                    continue
                for local_idx in range(self.num_local_conditions):
                    # slice out each condition's 3-ch
                    start_ch = 3 * local_idx
                    end_ch = 3 * (local_idx + 1)
                    grid = torchvision.utils.make_grid(images[k][:, start_ch:end_ch, :, :], nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # [-1,1] -> [0,1]
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                    grid = (grid * 255).astype(np.uint8)
                    filename = f"gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}_{k}_{local_idx}.png"
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)
            else:
                # For all other keys (like "reconstruction", "samples", etc.)
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # [-1,1] -> [0,1]
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = f"gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}_{k}.png"
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """
        Called at the end of a training/validation batch to log images.
        """
        check_idx = batch_idx  # or pl_module.global_step if you prefer logging by global_step
        if (self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(
                pl_module.logger.save_dir, split, images,
                pl_module.global_step, pl_module.current_epoch, batch_idx
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
