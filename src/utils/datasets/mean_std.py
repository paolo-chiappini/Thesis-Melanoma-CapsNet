from torch.utils.data import DataLoader


def compute_mean_std(dataset, num_workers):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    mean = 0.0
    std = 0.0
    total = 0
    for images, _, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += batch_samples
    mean /= total
    std /= total
    return mean, std
