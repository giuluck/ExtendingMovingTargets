import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.regressions import synthetic_function
from src.util.plot import ColorFader
from analysis_callback import AnalysisCallback, PRETRAINING, GroundCallback, BoundsCallback


class SyntheticAdjustments(AnalysisCallback):
    ground_size = 0.3
    alpha = 0.4

    def __init__(self, scalers, file_signature='../../temp/synthetic_adjustments', **kwargs):
        super(SyntheticAdjustments, self).__init__(scalers=scalers, file_signature=file_signature, **kwargs)

    def on_process_start(self, macs, x, y, val_data):
        super(SyntheticAdjustments, self).on_process_start(macs, x, y, val_data)
        self.data['ground'] = synthetic_function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
        self.data[f'adj err {iteration}'] = self.data[f'adj {iteration}'] - self.data['ground']
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', SyntheticAdjustments.ground_size * np.ones_like(y))

    def plot_function(self, iteration):
        def synthetic_inverse(column):
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        a, sw, pred = self.data['a'], self.data[f'sw {iteration}'], synthetic_inverse(f'pred {iteration}')
        s, p = ['aug' if b else 'label' for b in np.isnan(self.data['label'])], dict(aug='blue', label='black')
        gs, al = SyntheticAdjustments.ground_size, SyntheticAdjustments.alpha
        sns.lineplot(x=self.data['a'], y=synthetic_inverse('ground'), label='ground', color='green')
        sns.scatterplot(x=a, y=pred, label='predictions', color='red', alpha=al, s=gs * 50)
        if iteration == PRETRAINING:
            adj = synthetic_inverse('label')
        else:
            adj = synthetic_inverse(f'adj {iteration}')
        sw[np.array(s) == 'label'] = gs
        sns.scatterplot(x=a, y=adj, hue=s, style=s, size=sw, size_norm=(0, 1), sizes=(0, 100), palette=p, alpha=al)
        plt.legend(['predictions', 'labels' if iteration == PRETRAINING else 'adjusted'])


class SyntheticResponse(AnalysisCallback):
    def __init__(self, scalers, file_signature='../../temp/synthetic_response', res=50, **kwargs):
        super(SyntheticResponse, self).__init__(scalers=scalers, file_signature=file_signature, **kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(-1, 1))

    def on_training_end(self, macs, x, y, val_data, iteration):
        input_grid = self.x_scaler.transform(self.grid[['a', 'b']])
        self.grid[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(input_grid))

    def plot_function(self, iteration):
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)


class SyntheticBounds(BoundsCallback):
    def __init__(self, scalers, file_signature='../../temp/synthetic_bounds', sorting_attributes='price', **kwargs):
        super(SyntheticBounds, self).__init__(scalers=scalers, file_signature=file_signature,
                                              sorting_attributes=sorting_attributes, **kwargs)


class SyntheticGround(GroundCallback):
    def __init__(self, scalers, file_signature='../../temp/synthetic_ground', **kwargs):
        super(SyntheticGround, self).__init__(scalers=scalers, file_signature=file_signature, **kwargs)
