from .train import run_training
from .perturbation import run_perturbation
from .evaluate import run_evaluation
from .test import run_testing
from .train_runner import TrainRunner
from .evaluate_runner import EvaluateRunner
from .test_runner import TestRunner
from .perturbation_runner import PerturbationRunner

tasks = {
    "train": TrainRunner,
    "perturbation": EvaluateRunner,
    "evaluate": TestRunner,
    "test": PerturbationRunner,
}


def get_task(task_name):
    return tasks.get(task_name)
