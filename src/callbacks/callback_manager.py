from .callback import Callback


class CallbackManager:
    def __init__(self, callbacks=None, logger=None):
        self.callbacks = callbacks if callbacks is not None else []
        self.logger = logger
        assert isinstance(
            self.callbacks, list
        ), "Callbacks must be a list of Callback instances"
        for callback in self.callbacks:
            assert isinstance(
                callback, Callback
            ), "Callback must be an instance of Callback"
            callback.set_logger(logger=self.logger)

    def add_callback(self, callback):
        assert isinstance(
            callback, Callback
        ), "Callback must be an instance of Callback"
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        assert isinstance(
            callback, Callback
        ), "Callback must be an instance of Callback"
        self.callbacks.remove(callback)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_reconstruction(self, images, reconstructions, epoch, phase):
        for callback in self.callbacks:
            callback.on_reconstruction(images, reconstructions, epoch, phase)
