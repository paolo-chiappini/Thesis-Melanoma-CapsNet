from .train import run_training
from .perturbation import run_perturbation


def get_task(task_name):
    tasks = {"train": run_training, "perturbation": run_perturbation}
    return tasks.get(task_name)
