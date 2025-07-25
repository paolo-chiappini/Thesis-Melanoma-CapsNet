class Callback:
    """
    Base class for all callbacks.
    """

    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        """
        pass

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.
        """
        pass

    def on_reconstruction(self, images, reconstructions, epoch, phase):
        """
        Called at the end of each reconstruction.
        """
        pass
