import torch
import torch.functional as F


def train_tune_recon_hyperparam(
    model,
    dataloader,
    lambda_recon=1e-3,
    decoder_lr=None,
    epochs=3,
    device="cuda" if torch.cude.is_available() else "cpu",
):
    model = model.to(device)
    model.train()

    decoder_params = list(model.decoder.parameters())
    other_params = [p for p in model.parameters if p not in decoder_params]

    if decoder_lr:
        optimizer = torch.optim.Adam(
            [
                {"params": other_params},
                {"params": decoder_params, "lr": decoder_lr},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        total_loss, total_recon = 0.0, 0.0
        for i, batch in dataloader["train"]:
            images, labels, vas, masks, va_masks = [b.to(device) for b in batch]
            optimizer.zero_grad()

            out_va, recon, out_class, _ = model(images)

            class_loss = F.cross_entropy(out_class, labels)
            va_loss = F.binary_cross_entropy_with_logits(out_va, vas)
            recon_loss = F.mse_loss(recon, images)

            loss = class_loss + va_loss + lambda_recon * recon_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Recon: {total_recon:.4f}")

    model.eval()
    with torch.no_grad():
        images_val, *_ = next(iter(dataloader["test"]))
        images_val = images_val.to(device)
        _, recon_val, _, _ = model(images_val)
        mse_val = F.mse_loss(recon_val, images_val).item()

    return mse_val


def find_best_recon_loss(
    model,
    dataloader,
    lambdas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    device="cuda" if torch.cude.is_available() else "cpu",
):
    losses_val = []
    for lambda_recon in lambdas:
        losses_val.append(
            train_tune_recon_hyperparam(model, dataloader, lambda_recon, device=device)
        )

    best = torch.argmin(losses_val)
    return lambdas[best]
