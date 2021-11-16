"""Benchmark Models Testing Script."""

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    factory, _ = DatasetFactory().restaurants(data_args=dict(full_grid=True), callbacks=['logger', 'response'])
    factory.get_sbr(wandb_name=None, verbose=True).test(summary_args=dict(do_plot=False))
