import torch
from tqdm import tqdm

from losses import create_combined_loss
from utils.visualization.capsule_contribution import perturb_all_capsules

from .base_runner import BaseRunner


class PerturbationRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=False)

        self.build_model(load_weights=True)
        self.loss_criterion = create_combined_loss(
            config=self.config, device=self.device
        )

    def execute(self):
        sampled_images = self.get_samples_from_classes(
            loader=self.loaders["val"],
            num_samples=self.config["perturbation"]["num_samples"],
            num_classes=self.config["model"]["num_classes"],
        )

        for i, image in enumerate(sampled_images):
            perturb_all_capsules(
                self.model,
                image,
                device=self.device,
                visual_attributes=self.dataset.visual_attributes,
                out_prefix=f"img{i}_global_label{i // self.config['perturbation']['num_samples']}",  # TODO: check assumption: loop in class order
                global_perturbation=self.config["perturbation"]["is_global"],
            )

    def get_samples_from_classes(self, loader, num_samples=3, num_classes=2):

        classes = range(num_classes)
        class_images = {i: [] for i in classes}

        with tqdm(total=len(loader), desc="Collecting samples") as pbar:
            for batch in loader:
                images, labels = batch["images"], batch["malignancy_targets"]
                for img, label in zip(images, labels):
                    label = label.item()
                    if len(class_images[label]) < num_samples:
                        class_images[label].append(img)

                if all((len(class_images[cl]) >= num_samples for cl in classes)):
                    break

                print([len(class_images[cl]) for cl in class_images])
                pbar.update(1)

        print("Finished collecting samples")

        for label in classes:
            class_images[label] = torch.stack(class_images[label])

        all_images = torch.cat(list(class_images.values()), dim=0)
        return all_images
