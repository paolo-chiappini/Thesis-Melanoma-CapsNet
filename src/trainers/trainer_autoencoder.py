from datetime import datetime
import os
import torch
import torch.nn as nn
from tqdm import tqdm


def train_autoencoder(
    model, dataloader, num_epochs=50, lr=1e-4, device="cuda", callback_manager=None
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        for batch in progress_bar:
            images = batch[0].to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if callback_manager is not None:
                callback_manager.on_batch_end(
                    batch, logs={"loss": running_loss, "epoch": epoch, "phase": "train"}
                )

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if callback_manager is not None:
            callback_manager.on_epoch_end(
                epoch=epoch, logs={"loss": avg_loss, "epoch": epoch, "phase": "train"}
            )

            model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader))
            val_images = sample_batch[0].to(device)
            val_outputs = model(val_images)

            callback_manager.on_epoch_end(
                epoch=epoch,
                logs={
                    "inputs": val_images,
                    "outputs": val_outputs,
                    "epoch": epoch,
                    "phase": "test",
                },
            )

            callback_manager.on_reconstruction(
                val_images[:8],
                val_outputs[:8],
                epoch,
                "test",  # <-- Required for the reconstruction callback to work
            )
        model.train()

    print("Training complete")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    torch.save(
        model.state_dict(),
        os.path.join("checkpoints/", f"ae_{now}.pth.tar"),
    )
