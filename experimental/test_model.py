"""Benchmark Models Testing Script."""

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    for dataset in ['cars', 'synthetic', 'puzzles', 'restaurants', 'default', 'law']:
        factory, _ = DatasetFactory().get_dataset(name=dataset)
        factory.get_mlp(wandb_name=None, epochs=0).validate(num_folds=1, summary_args={})
