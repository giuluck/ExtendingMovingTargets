"""Basic Callback Interface."""

from typing import Optional as Opt

from moving_targets.utils.typing import Dataset, Vector, Matrix, Iteration


class Callback:
    """Basic interface for a Moving Target's callback."""

    def __init__(self):
        super(Callback, self).__init__()

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        """Routine called at the end of the MACS fitting process.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            **kwargs: additional arguments.
        """
        pass

    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        """Routine called at the beginning of the MACS fitting process.

        Args:
            macs: reference to the MACS object encapsulating the master.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            **kwargs: additional arguments.
        """
        pass

    def on_pretraining_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        """Routine called at the beginning of the MACS pretraining phase.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            **kwargs: additional arguments.
        """
        self.on_iteration_start(macs, x, y, val_data, **kwargs)
        self.on_training_start(macs, x, y, val_data, **kwargs)

    def on_pretraining_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        """Routine called at the end of the MACS pretraining phase.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            **kwargs: additional arguments.
        """
        self.on_training_end(macs, x, y, val_data, **kwargs)
        self.on_iteration_end(macs, x, y, val_data, **kwargs)

    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        """Routine called at the beginning of a MACS iteration.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            iteration: the current MACS iteration, usually a number.
            **kwargs: additional arguments.
        """
        pass

    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        """Routine called at the end of a MACS iteration.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            iteration: the current MACS iteration, usually a number.
            **kwargs: additional arguments.
        """
        pass

    def on_training_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        """Routine called at the beginning of a MACS training phase.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            iteration: the current MACS iteration, usually a number.
            **kwargs: additional arguments.
        """
        pass

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        """Routine called at the end of a MACS training phase.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            iteration: the current MACS iteration, usually a number.
            **kwargs: additional arguments.
        """
        pass

    def on_adjustment_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        """Routine called at the beginning of a MACS adjustment phase.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            iteration: the current MACS iteration, usually a number.
            **kwargs: additional arguments.
        """
        pass

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Opt[Dataset],
                          iteration: Iteration, **kwargs):
        """Routine called at the end of a MACS adjustment phase.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            adjusted_y: the vector of adjusted labels.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            iteration: the current MACS iteration, usually a number.
            **kwargs: additional arguments.
        """
        pass
