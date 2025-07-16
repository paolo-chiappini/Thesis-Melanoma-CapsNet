from .train import run_training


def get_task(task_name):
    tasks = {"train": run_training, "visualize": None}
    return tasks.get(task_name)
