from .train import run_training
from .perturbation import run_perturbation
from .evaluate import run_evaluation
from .test import run_testing


tasks = {
    "train": run_training,
    "perturbation": run_perturbation,
    "evaluate": run_evaluation,
    "test": run_testing,
}


def get_task(task_name):
    return tasks.get(task_name)
