from .callback import Callback
from .callback_early_stopping import EarlyStoppingCallback
from .callback_manager import CallbackManager
from .callback_plot import PlotCallback
from .callback_reconstruction import ReconstructionCallback

callbacks = {cls.__name__: cls for cls in Callback.__subclasses__()}

__all__ = ["Callback", "CallbackManager", *callbacks.keys()]


def get_callbacks(callback_config):
    instances = []
    for config in callback_config:
        name = config["name"]
        kwargs = {k: v for k, v in config.items() if k != "name"}
        callback_class = callbacks.get(name)
        if callback_class is None:
            raise ValueError(f"Unknown callback: {name}")
        instances.append(callback_class(**kwargs))
    return instances
