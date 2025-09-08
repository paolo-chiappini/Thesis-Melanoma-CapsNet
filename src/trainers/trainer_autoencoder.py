import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .trainer_base import BaseTrainer


class AutoEncoderTrainer(BaseTrainer):
    def compute_custom_metrics(self, outputs, batch_data):
        targets = batch_data["images"]
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
