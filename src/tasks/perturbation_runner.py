import torch
from tqdm import tqdm
from .base_runner import BaseRunner
from utils.losses import get_loss
from trainers import get_trainer
from utils.visualization.capsule_contribution import perturb_all_capsules


class PerturbationRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=False)

        self.build_model(load_weights=True)
        self.loss_criterion = get_loss(config=self.config["trainer"]["loss"])

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

    def get_samples_from_classes(loader, num_samples=3, num_classes=2):
        classes = range(num_classes)
        class_images = {i: [] for i in classes}

        with tqdm(total=len(loader), desc="Collecting samples") as pbar:
            for batch in loader:
                images, labels = batch[0], batch[1]
                for img, label in zip(images, labels):
                    label = label.item()
                    if len(class_images[label]) < num_samples:
                        class_images[label].append(img)

                if all((len(class_images[l]) >= num_samples for l in classes)):
                    break

                print([len(class_images[l]) for l in class_images])
                pbar.update(1)

        print("Finished collecting samples")

        for label in classes:
            class_images[label] = torch.stack(class_images[label])

        all_images = torch.cat(list(class_images.values()), dim=0)
        return all_images
