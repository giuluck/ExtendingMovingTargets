"""Script to obtain Sample Plots from Augmented Data and Moving Targets' Instances."""
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from experimental.utils import AnalysisCallback, DistanceAnalysis, CarsAdjustments, DatasetFactory
from moving_targets.util.typing import Dataset, Iteration
from src.datasets import PuzzlesManager

ALPHA = 0.6
POINT_WIDTH = 400
LINE_WIDTH = 4
FONT_SCALES = [2.0, 1.2]
FIG_SIZE = (15, 10)


# noinspection PyMissingOrEmptyDocstring
class ExportAnalysis(AnalysisCallback):
    def __init__(self, config_name: str = 'image', **kwargs):
        super(ExportAnalysis, self).__init__()
        self.config_name = config_name

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        self.data = self.data.sort_values('price')
        plt.figure(tight_layout=True, figsize=FIG_SIZE)
        self.plot_function(1)
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(f'../temp/{self.config_name}.png', format='png')


# noinspection PyMissingOrEmptyDocstring
class ExportDistance(DistanceAnalysis, ExportAnalysis):
    def __init__(self, config_name: str):
        super(ExportDistance, self).__init__()
        self.config_name = config_name

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        x = np.arange(len(self.data))
        y, p, j = self.data[self.y].values, self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        for i in x:
            plt.plot([i, i], [p[i], j[i]], c='red', alpha=ALPHA, linewidth=LINE_WIDTH)
            plt.plot([i, i], [y[i], j[i]], c='blue', alpha=ALPHA, linewidth=LINE_WIDTH)
        sns.lineplot(x=x, y=p, color='red', linewidth=LINE_WIDTH, label='predictions').set_xticks([])
        sns.scatterplot(x=x, y=y, marker='o', color='blue', s=POINT_WIDTH, label='labels')
        sns.scatterplot(x=x, y=j, marker='X', color='black', s=POINT_WIDTH, label='adjusted')
        return


# noinspection PyMissingOrEmptyDocstring
class ExportCars(CarsAdjustments, ExportAnalysis):
    def __init__(self, config_name: str):
        super(ExportCars, self).__init__()
        self.config_name = config_name

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        x, y = self.data['price'].values, self.data['sales'].values
        s, m = np.array(self.data['mask']), dict(aug='o', label='X')
        sn, al = (0, POINT_WIDTH), ALPHA
        p, adj, sw = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'], self.data[f'sw {iteration}'].values
        for v, c, k in [('label', 'blue', 'X'), ('aug', 'black', 'o')]:
            v = s == v
            sns.scatterplot(x=x[v], y=adj[v], marker=k, size=sw[v], size_norm=(0, 1), sizes=sn, color=c, alpha=al)
        sns.lineplot(x=x, y=p, color='red', linewidth=LINE_WIDTH)
        plt.legend(['predictions', 'supervised', 'unsupervised'])
        return


if __name__ == '__main__':
    sns.set_context('talk', font_scale=FONT_SCALES[0])
    factory, _ = DatasetFactory().cars(data_args=dict(full_features=False))
    factory.get_mt().experiment(callbacks=[ExportCars('default_config_adj'), ExportDistance('default_config_dst')])
    factory.get_mt(mst_learner_omega=10.0).experiment(callbacks=[ExportCars('learner_omega_config')])
    factory.get_mt(mst_master_omega=100.0).experiment(callbacks=[ExportDistance('master_omega_config')])
    factory.get_mt(mst_learner_weights='infeasible').experiment(callbacks=[ExportCars('learner_weights_config')])

    sns.set_context('talk', font_scale=FONT_SCALES[1])
    manager = PuzzlesManager(filepath='../res/puzzles.csv', full_features=False)
    xtr, ytr = manager.get_folds(num_folds=None)[0]['train']
    xag, yag = manager.get_augmented_data(xtr, ytr)[0]
    xag['Augmented'] = np.isnan(yag['label'])
    _, axes = plt.subplots(1, 3, sharey='all', figsize=(14, 4), tight_layout=True)
    for ax, feature in zip(axes, ['word_count', 'star_rating', 'num_reviews']):
        sns.histplot(data=xag, x=feature, hue='Augmented', ax=ax)
    plt.savefig('../temp/augmentation.png', format='png')
