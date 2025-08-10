from .train_runner import TrainRunner
from .evaluate_runner import EvaluateRunner
from .test_runner import TestRunner
from .perturbation_runner import PerturbationRunner

tasks = {
    "train": TrainRunner,
    "evaluate": EvaluateRunner,
    "test": TestRunner,
    "perturbation": PerturbationRunner,
}


def get_task(task_name):
    return tasks.get(task_name)
