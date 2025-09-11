import numpy as np
import torch
from tqdm import tqdm


def compute_capsule_activations(model, dataloader, device="cuda"):
    model.eval()
    all_caps = []
    all_attrs = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Extracting capsule activations", unit="batch"
        ):
            images = batch["images"].to(device)
            attrs = batch["visual_attributes_targets"].cpu().numpy()
            caps = model.encode(images)

            all_caps.append(caps.cpu().numpy())
            all_attrs.append(attrs)

    return np.concatenate(all_caps), np.concatenate(all_attrs)
