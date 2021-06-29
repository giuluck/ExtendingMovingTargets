"""Benchmark Models Testing Script."""

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    factory, _ = DatasetFactory().law(data_args=dict(full_features=True, full_grid=False))
    factory.get_sbr(wandb_name=None, verbose=True).validate(num_folds=5, summary_args=dict(do_plot=False))
