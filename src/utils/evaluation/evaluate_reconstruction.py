import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from tqdm import tqdm


def evaluate_reconstruction(model, dataloader, device="cuda", prepare_batch=None):
    assert (
        prepare_batch is not None
    ), "Must specify a way to unpack the batch data, prepare_batch cannot be None"

    model.eval()
    mse_losses = []
    ssim_scores = []
    psnr_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating reconstruction", unit="batch"):
            batch_dict = prepare_batch(batch)
            images = batch_dict["inputs"].to(device)
            masked_images = images
            if "masks" in batch_dict.keys():
                masked_images = images * batch_dict["masks"]

            capsules = model.encode(images)
            recon = model.decode(
                capsules[:, :, 1:]
            )  # discard first dimension used for logits

            mse = F.mse_loss(recon, masked_images, reduction="none").mean(dim=(1, 2, 3))
            mse_losses.extend(mse.cpu().numpy())

            images_np = masked_images.permute(0, 2, 3, 1).cpu().numpy()
            recon_np = recon.permute(0, 2, 3, 1).cpu().numpy()
            for img, rec in tqdm(
                zip(images_np, recon_np),
                total=len(images_np),
                desc="Calculating SSIM/PSNR",
                leave=False,
            ):
                ssim_scores.append(ssim(img, rec, channel_axis=-1, data_range=1.0))
                psnr_scores.append(psnr(img, rec, data_range=1.0))

    results = {
        "MSE_mean": np.mean(mse_losses),
        "MSE_std": np.std(mse_losses),
        "SSIM_mean": np.mean(ssim_scores),
        "PSNR_mean": np.mean(psnr_scores),
    }
    return results
