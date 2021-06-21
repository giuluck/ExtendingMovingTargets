"""Benchmark Models Testing Script."""

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    factory, _ = DatasetFactory().cars_univariate(data_args=dict(full_features=False, full_grid=False))
    factory.get_mlp(wandb_name=None).validate(num_folds=10, summary_args=dict(do_plot=False))
