from .train import run_training
from .perturbation import run_perturbation
from .evaluate import run_evaluation


def get_task(task_name):
    tasks = {
        "train": run_training,
        "perturbation": run_perturbation,
        "evaluate": run_evaluation,
    }
    return tasks.get(task_name)
