import random
import time
from typing import Optional, List, Union

import numpy as np
import tensorflow as tf
from moving_targets.callbacks import WandBLogger, Callback

from src.datasets import Synthetic, Manager
from src.models import MT


class Handler:
    """Experiment Handler. The inputs are the the dataset alias, the wandb project name, and the configuration."""

    _SEED: int = 0

    def __init__(self,
                 dataset: str,
                 project: Optional[str] = None,
                 init_step: str = 'pretraining',
                 alpha: float = 1.0,
                 y_loss: str = 'mse',
                 p_loss: str = 'mse'):

        # handle dataset
        if dataset == 'synthetic':
            self.dataset: Manager = Synthetic()
        else:
            raise ValueError(f"Unknown dataset '{dataset}'")

        # handle logger
        self.wandb_logger = None if project is None else WandBLogger(
            project=project,
            entity='giuluck',
            run_name=dataset,
            init_step=init_step,
            alpha=alpha,
            y_loss=y_loss,
            p_loss=p_loss
        )

        self.init_step: str = init_step
        self.alpha: float = alpha
        self.y_loss: str = y_loss
        self.p_loss: str = p_loss

    def experiment(self,
                   iterations: int = 15,
                   num_folds: Optional[int] = None,
                   folds_index: Optional[List[int]] = None,
                   fold_verbosity: Union[bool, int] = True,
                   model_verbosity: Union[bool, int] = False,
                   callbacks: Optional[List[Union[str, Callback]]] = None,
                   folder: Optional[str] = None,
                   plot_history: bool = True,
                   plot_summary: bool = True):
        """Builds controllable Moving Targets experiment with custom callbacks and plots.

        - 'callbacks' is a list of either callbacks instances or (string) aliases. If None, datasets callbacks are used.
        - 'folder' is the folder where to export callbacks files.
        """
        fold_verbosity = False if num_folds is None or num_folds == 1 else fold_verbosity
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION STARTED')
        folds = self.dataset.folds(num_folds=num_folds)
        for i, fold in enumerate([folds] if num_folds is None else folds):
            random.seed(self._SEED)
            np.random.seed(self._SEED)
            tf.random.set_seed(self._SEED)
            if folds_index is not None and i not in folds_index:
                continue
            # handle verbosity
            start_time = time.time()
            if fold_verbosity in [2, True]:
                print(f'  > fold {i + 1:0{len(str(num_folds))}}/{num_folds}', end=' ')
            # handle default (dataset) callbacks and wandb callback
            callbacks = list(self.dataset.callbacks.keys() if callbacks is None else callbacks)
            signature = (lambda c: None) if folder is None else (lambda c: f'{folder}/{self.dataset.__name__}_{c}')
            for index, callback in enumerate(callbacks):
                if isinstance(callback, str):
                    callbacks[index] = self.dataset.callbacks[callback](fs=signature(callback))
                elif isinstance(callback, WandBLogger) and num_folds is not None:
                    callback.config['fold'] = i
            # build and fit model
            model = MT(
                dataset=self.dataset,
                init_step=self.init_step,
                alpha=self.alpha,
                y_loss=self.y_loss,
                p_loss=self.p_loss,
                iterations=iterations,
                callbacks=callbacks,
                val_data={split: data for split, data in fold.validation.items() if split != 'train'},
                verbose=model_verbosity
            )
            history = model.fit(x=fold.x, y=fold.y)
            # handle plots
            if plot_history:
                history.plot()
            if plot_summary:
                self.dataset.summary(model=model, plot=True, **fold.validation)
            # handle verbosity
            if fold_verbosity in [2, True]:
                print(f'-- elapsed time: {time.time() - start_time}')
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION ENDED')
