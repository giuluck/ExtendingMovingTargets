"""Benchmark Models Testing Script."""

from experimental.utils.factories import DatasetFactory

if __name__ == '__main__':
    factory, _ = DatasetFactory().synthetic(epochs=0, data_args=dict(full_grid=True))
    factory.get_mlp(wandb_name=None, verbose=True).test(summary_args=dict(do_plot=True))
