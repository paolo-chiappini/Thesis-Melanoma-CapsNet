import torch
import matplotlib.pyplot as plt


def visualize_capsule_contribution(
    model,
    input_image,
    capsule_idx,
    device,
    dim_range=(-2.5, 2.5),
    steps=10,
):
    model.eval()
    with torch.no_grad():
        _, _, _, capsule_pose = model(
            input_image.unsqueeze(0).to(device)
        )  # capsule_pose: [1, N_caps, D, H, W]
        capsule_pose = capsule_pose.clone()

        dim_size = capsule_pose.shape[2]
        fig, axes = plt.subplots(dim_size, steps, figsize=(steps * 5, dim_size * 5))

        for d in range(dim_size):
            for i, delta in enumerate(torch.linspace(*dim_range, steps)):
                perturbed = capsule_pose.clone()
                perturbed[0, capsule_idx, d] += delta

                decoder_input = perturbed.view(1, -1)

                recon = model.decoder(decoder_input)[0].cpu().clamp(0, 1)
                recon = recon.view(
                    *input_image.shape
                )  # Transform from linear to 3d tensor
                recon = recon.permute(1, 2, 0)  # Transform from CxHxW to HxWxC

                axes[d, i].imshow(recon)
                axes[d, i].axis("off")
                if d == 0:
                    axes[d, i].set_title(f"{delta:.2f}")

        plt.suptitle(f"Perturbing Capsule {capsule_idx} Pose Dimensions")
        plt.tight_layout()
        plt.savefig(f"./reconstructions/perturbation_caps{capsule_idx}.png")
        print(f"Saved: ./reconstructions/perturbation_caps{capsule_idx}.png")
