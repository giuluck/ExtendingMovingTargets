"""Data Manager."""
import time
from typing import Dict, Optional, Tuple, Any, List, Union

import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.callbacks import Callback, WandBLogger
from moving_targets.metrics import Metric
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Dataset

from src.util.experiments import setup
from src.util.preprocessing import Scaler, split_dataset, cross_validate
from src.util.typing import Method


class Fold:
    """Data class containing the information of a fold for k-fold cross-validation."""

    def __init__(self,
                 data: pd.DataFrame,
                 label: str,
                 x_scaler: Scaler,
                 y_scaler: Scaler,
                 validation: Dict[str, pd.DataFrame]):
        """
        :param data:
            The training data.

        :param label:
            The target label.

        :param x_scaler:
            The input data scaler.

        :param y_scaler:
            The output data scaler.

        :param validation:
            A shared validation dataset which is common among all the k folds.
        """

        def split_df(df: pd.DataFrame, fit: bool):
            x_df, y_df = df.drop(columns=label), df[label].values
            if fit:
                return x_scaler.fit_transform(x_df), y_scaler.fit_transform(y_df)
            else:
                return x_scaler.transform(x_df), y_scaler.transform(y_df)

        x, y = split_df(df=data, fit=True)

        self.x = x
        """The input data."""

        self.y = y
        """The output data/info."""

        self.x_scaler: Scaler = x_scaler
        """The input data scaler."""

        self.y_scaler: Scaler = y_scaler
        """The output data scaler."""

        self.validation: Dataset = {k: split_df(df=v, fit=False) for k, v in validation.items()}
        """A shared validation dataset which is common among all the k folds."""


class AbstractManager:
    """Abstract dataset manager."""

    @classmethod
    def name(cls) -> str:
        """The dataset name

        :return:
            A string representing the dataset name.
        """
        return cls.__name__.replace('Manager', '').lower()

    @classmethod
    def data(cls, **data_kwargs) -> Dict[str, pd.DataFrame]:
        """Loads the dataset.

        :param data_kwargs:
            Any dataset-dependent argument that may be necessary in the implementation of this method.

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        raise NotImplementedError(not_implemented_message(name='load_data', static=True))

    @classmethod
    def model(cls, **model_kwargs) -> MACS:
        """The model to evaluate.

        :param model_kwargs:
            Any model-dependent argument that may be necessary in the implementation of this method.

        :return:
            A `MACS` instance.
        """
        raise NotImplementedError(not_implemented_message(name='model', static=True))

    @classmethod
    def metrics(cls) -> List[Metric]:
        """The metrics used to evaluate the model.

        :return:
            A list of `Metric` instances.
        """
        raise NotImplementedError(not_implemented_message(name='metrics', static=True))

    def __init__(self,
                 label: str,
                 stratify: bool,
                 x_scaling: Method,
                 y_scaling: Method,
                 seed: int = 0,
                 **kwargs):
        """
        :param label:
            The name of the target feature.

        :param stratify:
            Whether or not to stratify the data when splitting.

        :param x_scaling:
            The input data default scaling method.

        :param y_scaling:
            The output data default scaling method.

        :param seed:
            The random seed.

        :param kwargs:
            Any additional argument to be passed to the static `data()` and 'model()' functions.
        """
        train, test = self.data(**kwargs).values()

        self.train: pd.DataFrame = train
        """The training data."""

        self.test: pd.DataFrame = test
        """The test data."""

        self.kwargs: Dict[str, Any] = kwargs
        """Any additional argument to be passed to the static `data()` and 'model()' functions."""

        self.label: str = label
        """The name of the target feature."""

        self.stratify: bool = stratify
        """The vector used for stratified sampling."""

        self.x_scaling: Method = x_scaling
        """The input data default scaling method."""

        self.y_scaling: Method = y_scaling
        """The output data default scaling method."""

        self.seed: int = seed
        """The random seed."""

    def get_scalers(self) -> Tuple[Scaler, Scaler]:
        """Returns the dataset scalers.

        :return:
            A pair of scalers, one for the input and one for the output data, respectively.
        """
        x_scaler = Scaler() if self.x_scaling is None else Scaler(default_method=self.x_scaling)
        y_scaler = Scaler() if self.y_scaling is None else Scaler(default_method=self.y_scaling)
        return x_scaler, y_scaler

    def get_folds(self, num_folds: Optional[int] = None, **kwargs) -> Union[Fold, List[Fold]]:
        """Gets the data split in folds.

        With num_folds = None directly returns a tuple with train/test splits and scalers.
        With num_folds = 1 returns a list with a single tuple with train/val/test splits and scalers.
        With num_folds > 1 returns a list of tuples with train/val/test splits and their respective scalers.

        :param num_folds:
            The number of folds for k-fold cross-validation.

        :param kwargs:
            Arguments passed either to `split_dataset()` or `cross_validate()` method, depending on the number of folds.

        :return:
            Either a tuple of `Dataset` and `Scalers` or a list of them, depending on the number of folds.
        """
        x_scaler, y_scaler = self.get_scalers()
        fold_kwargs = dict(label=self.label, x_scaler=x_scaler, y_scaler=y_scaler)
        if num_folds is None:
            validation = dict(train=self.train, test=self.test)
            return Fold(data=self.train, validation=validation, **fold_kwargs)
        elif num_folds == 1:
            fold = split_dataset(self.train, test_size=0.2, val_size=0.0, stratify=self.stratify, **kwargs)
            fold['validation'] = fold.pop('test')
            fold['test'] = self.test
            return [Fold(data=fold['train'], validation=fold, **fold_kwargs)]
        else:
            folds = cross_validate(self.train, num_folds=num_folds, stratify=self.stratify, **kwargs)
            return [Fold(data=fold['train'], validation={**fold, 'test': self.test}, **fold_kwargs) for fold in folds]

    def experiment(self,
                   iterations: int,
                   num_folds: Optional[int] = None,
                   folds_index: Optional[List[int]] = None,
                   fold_verbosity: Union[bool, int] = True,
                   model_verbosity: Union[bool, int] = False,
                   callbacks: Optional[List[Callback]] = None,
                   plot_args: Dict = None,
                   summary_args: Dict = None):
        """Builds controllable Moving Targets experiment with custom callbacks and plots.

        :param iterations:
            The number of Moving Targets' iterations.

        :param num_folds:
            The number of folds. If None, train/test split only is performed.

        :param folds_index:
            The list of folds index to be validated. If None, all of them are validated.

        :param fold_verbosity:
            How much information about the k-fold cross-validation process to print.

        :param model_verbosity:
            How much information about the Moving Targets process to print.

        :param callbacks:
            List of `Callback` object for the Moving Targets process.

        :param plot_args:
            Arguments passed to the method `History.plot()`.

        :param summary_args:
            The dictionary of arguments for the evaluation summary.
        """
        fold_verbosity = False if num_folds is None or num_folds == 1 else fold_verbosity
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION STARTED')
        folds = self.get_folds(num_folds=num_folds)
        for i, fold in enumerate([folds] if num_folds is None else folds):
            setup(seed=self.seed)
            if folds_index is not None and i not in folds_index:
                continue
            # handle verbosity
            start_time = time.time()
            if fold_verbosity in [2, True]:
                print(f'  > fold {i + 1:0{len(str(num_folds))}}/{num_folds}', end=' ')
            # handle wandb callback
            for c in callbacks:
                if isinstance(c, WandBLogger) and num_folds is not None:
                    c.config['fold'] = i
            # fit model
            model = self.model(**self.kwargs)
            history = model.fit(
                x=fold.x,
                y=fold.y,
                iterations=iterations,
                val_data=fold.validation,
                callbacks=callbacks,
                verbose=model_verbosity
            )
            # handle plots
            if plot_args is not None:
                history.plot(**plot_args)
            if summary_args is not None:
                self.evaluation_summary(model, **fold.validation, **summary_args)
            # handle verbosity
            if fold_verbosity in [2, True]:
                print(f'-- elapsed time: {time.time() - start_time}')
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION ENDED')

    def evaluation_summary(self, model, do_print: bool = False, **data_splits: Tuple[Any, Any]) -> pd.DataFrame:
        """Computes the metrics over a custom set of validation data, then builds a summary.

        :param model:
            A model object having the 'predict(x)' method.

        :param do_print:
            Whether or not to print the output summary.

        :param data_splits:
            A dictionary of named (x, y) tuples.

        :return:
            Either a dictionary for the metric values or a string representing the evaluation summary.
        """
        summary = {}
        for split_name, (x, y) in data_splits.items():
            p = model.predict(x).astype(np.float64)
            summary[split_name] = {}
            for metric in self.metrics():
                summary[split_name][metric.__name__] = metric(x, y, p)
        summary = pd.DataFrame.from_dict(summary)
        if do_print:
            print(summary)
        return summary
