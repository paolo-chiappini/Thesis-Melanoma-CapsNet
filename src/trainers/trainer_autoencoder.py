from .trainer_base import BaseTrainer
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class AutoEncoderTrainer(BaseTrainer):

    def prepare_batch(self, batch):
        images, _, _, _ = batch
        images = images.to(self.device)
        return {"inputs": images, "targets": images}

    def compute_loss(self, outputs, batch_data):
        return self.criterion(outputs, batch_data["targets"])

    def compute_metrics(self, outputs, batch_data):
        targets = batch_data["targets"]
        _, preds = outputs

        # MSE
        mse = F.mse_loss(preds, targets, reduction="none").mean(dim=(1, 2, 3))
        mse_mean = mse.mean().item()

        # SSIM & PSNR
        preds_np = preds.detach().permute(0, 2, 3, 1).cpu().numpy()
        targets_np = targets.detach().permute(0, 2, 3, 1).cpu().numpy()

        ssim_scores = []
        psnr_scores = []

        for pred_img, tgt_img in zip(preds_np, targets_np):
            ssim_scores.append(ssim(pred_img, tgt_img, channel_axis=-1, data_range=1.0))
            psnr_scores.append(psnr(pred_img, tgt_img, data_range=1.0))

        return {
            "MSE": mse_mean,
            "SSIM": float(np.mean(ssim_scores)),
            "PSNR": float(np.mean(psnr_scores)),
        }
