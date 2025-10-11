import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def perturb_all_capsules(
    model,
    input_image,
    device,
    visual_attributes,
    dim_range=(-2.5, 2.5),
    steps=10,
    out_prefix="",
    global_perturbation=False,
):
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_image.unsqueeze(0).to(device)
        )  # capsule_pose: [1, N_caps, D, H, W]

        capsule_pose = outputs.get("attribute_poses")
        capsule_pose = capsule_pose.clone()

        dim_size = capsule_pose.shape[2]

        contribution_vis = (
            visualize_global_capsule_contribution
            if global_perturbation
            else visualize_capsule_contribution
        )

        for capsule_idx in tqdm(
            range(len(visual_attributes)),
            desc="Perturbing capsules",
            unit="capsule",
        ):
            contribution_vis(
                model,
                input_image,
                capsule_pose,
                capsule_idx,
                dim_size,
                dim_range,
                steps,
                visual_attributes,
                out_prefix,
            )


def visualize_capsule_contribution(
    model,
    input_image,
    capsule_pose,
    capsule_idx,
    dim_size,
    dim_range,
    steps,
    visual_attributes,
    out_prefix,
):
    _, axes = plt.subplots(dim_size, steps, figsize=(steps * 2, dim_size * 2))

    for d in tqdm(
        range(dim_size), desc=f"Capsule {capsule_idx} dims", leave=False, unit="dim"
    ):
        for i, delta in enumerate(torch.linspace(*dim_range, steps)):
            perturbed = capsule_pose.clone()
            perturbed[0, capsule_idx, d] += delta

            decoder_input = perturbed.view(1, -1)
            recon = model.decoder(decoder_input)[0].cpu().clamp(0, 1)
            recon = recon.view(*input_image.shape)  # Transform from linear to 3d tensor
            recon = recon.permute(1, 2, 0)  # Transform from CxHxW to HxWxC

            axes[d, i].imshow(recon)
            axes[d, i].axis("off")
            if d == 0:
                axes[d, i].set_title(f"{delta:.2f}")

    plt.suptitle(f"Perturbing Capsule {capsule_idx} Pose Dimensions")
    plt.tight_layout()
    os.makedirs(f"./perturbations/{out_prefix}", exist_ok=True)
    out_file = f"./perturbations/{out_prefix}/caps{capsule_idx}_{visual_attributes[capsule_idx]}.png"
    plt.savefig(out_file)

    tqdm.write(f"[Saved] {out_file}")
    plt.close()


def visualize_global_capsule_contribution(
    model,
    input_image,
    capsule_pose,
    capsule_idx,
    dim_size,
    dim_range,
    steps,
    visual_attributes,
    out_prefix,
):
    _, axes = plt.subplots(1, steps, figsize=(steps * 2, 5))

    for i, delta in enumerate(
        tqdm(
            torch.linspace(*dim_range, steps),
            desc=f"Capsule {capsule_idx} global",
            leave=False,
            unit="step",
        )
    ):
        perturbed = capsule_pose.clone()
        perturbed[0, capsule_idx] += delta * torch.ones_like(perturbed[0, capsule_idx])

        decoder_input = perturbed.view(1, -1)
        recon = model.decoder(decoder_input)[0].cpu().clamp(0, 1)
        recon = recon.view(*input_image.shape)
        recon = recon.permute(1, 2, 0)

        axes[i].imshow(recon)
        axes[i].axis("off")
        axes[i].set_title(f"{delta:.2f}")

    plt.suptitle(f"Perturbing Capsule {capsule_idx} Pose Dimensions (Global)")
    plt.tight_layout()
    os.makedirs(f"./perturbations/{out_prefix}", exist_ok=True)
    out_file = f"./perturbations/{out_prefix}/caps{capsule_idx}_{visual_attributes[capsule_idx]}.png"
    plt.savefig(out_file)

    tqdm.write(f"[Saved] {out_file}")
    plt.close()
