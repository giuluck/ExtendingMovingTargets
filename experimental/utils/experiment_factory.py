"""Datasets Factory Handlers."""

from typing import Optional, List, Tuple

from experimental.utils.experiment_handler import ExperimentHandler
from moving_targets.callbacks import Callback, FileLogger
from src.datasets import IrisManager, WineManager


class ExperimentFactory:
    """Factory class that returns experiment handlers."""

    def __init__(self, res_folder: Optional[str] = '../../res/', temp_folder: Optional[str] = '../../temp/'):
        """
        :param res_folder:
            The res folder filepath where to retrieve inputs.

        :param temp_folder:
            The temp folder filepath where to place outputs.
        """

        self.res: Optional[str] = res_folder.strip('/')
        """The res folder filepath where to retrieve inputs."""

        self.temp: Optional[str] = temp_folder.strip('/')
        """The temp folder filepath where to place outputs."""

    def _get_shared_callbacks(self, callbacks: Optional[List[str]]) -> Tuple[List[Callback], List[str]]:
        """Checks if the input list contains the name of some shared callbacks and includes them in the output list.

        :param callbacks:
            List of callback names.

        :return:
            A tuple where the first item is a list of `Callback` object which are shared among datasets, and the second
            item is the same input list or an empty list if the input was None.
        """
        callbacks: List[str] = [] if callbacks is None else callbacks
        cb: List[Callback] = []
        if 'logger' in callbacks:
            cb.append(FileLogger(f'{self.temp}/log.txt', routines=['on_pretraining_end', 'on_iteration_end']))
        return cb, callbacks

    def iris(self, callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[ExperimentHandler, List[Callback]]:
        """Builds an experiment handler for the iris dataset.

        :param callbacks:
            List of callbacks aliases.

         :param kwargs:
            Custom arguments passed to a `ExperimentHandler` instance.

        :return:
            A tuple containing the `ExperimentHandler` as first item, and a list of `Callback` objects as second one.
        """
        cb, callbacks = self._get_shared_callbacks(callbacks)
        ds = ExperimentHandler(manager=IrisManager(filepath=f'{self.res}/iris.csv'), n_classes=3, **kwargs)
        return ds, cb

    def redwine(self, callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[ExperimentHandler, List[Callback]]:
        """Builds an experiment handler for the redwine dataset.

        :param callbacks:
            List of callbacks aliases.

         :param kwargs:
            Custom arguments passed to a `ExperimentHandler` instance.

        :return:
            A tuple containing the `ExperimentHandler` as first item, and a list of `Callback` objects as second one.
        """
        cb, callbacks = self._get_shared_callbacks(callbacks)
        ds = ExperimentHandler(manager=WineManager(filepath=f'{self.res}/redwine.csv'), n_classes=6, **kwargs)
        return ds, cb

    def whitewine(self, callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[ExperimentHandler, List[Callback]]:
        """Builds an experiment handler for the whitewine dataset.

        :param callbacks:
            List of callbacks aliases.

         :param kwargs:
            Custom arguments passed to a `ExperimentHandler` instance.

        :return:
            A tuple containing the `ExperimentHandler` as first item, and a list of `Callback` objects as second one.
        """
        cb, callbacks = self._get_shared_callbacks(callbacks)
        ds = ExperimentHandler(manager=WineManager(filepath=f'{self.res}/whitewine.csv'), n_classes=7, **kwargs)
        return ds, cb
