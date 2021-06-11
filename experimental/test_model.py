"""Benchmark Models Testing Script."""

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    factory, _ = DatasetFactory().get_dataset(name='synthetic')
    factory.get_tfl(wandb_name=None).validate(num_folds=2, summary_args={})
