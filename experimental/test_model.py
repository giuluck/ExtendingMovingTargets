"""Benchmark Models Testing Script."""

from experimental.utils import DatasetFactory
from experimental.utils.handlers import HandlersFactory

if __name__ == '__main__':
    outputs = DatasetFactory().get_dataset(name='restaurants')
    factory: HandlersFactory = outputs[0]
    factory.get_mlp(wandb_name=None, epochs=1000).validate(num_folds=1, summary_args={})
    # print('---------------------------------------------------------------------------')
    factory.get_sbr(wandb_name=None, epochs=1000, verbose=True).validate(num_folds=1, summary_args={})
    # print('---------------------------------------------------------------------------')
    factory.get_tfl(wandb_name=None, epochs=1000).validate(num_folds=1, summary_args={})
