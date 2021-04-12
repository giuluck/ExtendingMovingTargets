import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis_callback import AnalysisCallback, PRETRAINING, GroundCallback, BoundsCallback


class CarsAdjustments(AnalysisCallback):
    labels_size = 0.4
    alpha = 0.4

    def __init__(self, scalers, file_signature='../../temp/cars_adjustments', plot_kind='scatter', **kwargs):
        super(CarsAdjustments, self).__init__(scalers=scalers, file_signature=file_signature, **kwargs)
        assert plot_kind in ['line', 'scatter'], "plot_kind should be either 'line' or 'scatter'"
        self.plot_kind = plot_kind

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', CarsAdjustments.labels_size * np.ones_like(y))

    def plot_function(self, iteration):
        x, y = self.data['price'].values, self.data['sales'].values
        s, p = ['aug' if b else 'labels' for b in np.isnan(y)], dict(aug='blue', labels='black')
        gs, al = CarsAdjustments.labels_size, CarsAdjustments.alpha
        pred, adj = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'],
        if iteration == PRETRAINING:
            sns.scatterplot(x=x, y=y, hue=s, style=s, size=gs, size_norm=(0, 1), sizes=(0, 100), palette=p, alpha=al)
        elif self.plot_kind == 'line':
            sns.lineplot(x=x, y=adj, color='blue')
            for i in range(self.data.shape[0]):
                plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=al)
        elif self.plot_kind == 'scatter':
            sw = self.data[f'sw {iteration}']
            sw[np.array(s) == 'labels'] = gs
            sns.scatterplot(x=x, y=adj, hue=s, style=s, size=sw, size_norm=(0, 1), sizes=(0, 100), palette=p, alpha=al)
        sns.lineplot(x=x, y=pred, color='red')
        plt.legend(['predictions', 'labels' if iteration == PRETRAINING else 'adjusted'])
        return f'{iteration}) adj. mae = {np.abs((adj - y).fillna(0)).mean():.4f}'


class CarsBounds(BoundsCallback):
    def __init__(self, scalers, file_signature='../../temp/cars_bounds', sorting_attributes='price', **kwargs):
        super(CarsBounds, self).__init__(scalers=scalers, file_signature=file_signature,
                                         sorting_attributes=sorting_attributes, **kwargs)


class CarsGround(GroundCallback):
    def __init__(self, scalers, file_signature='../../temp/cars_ground', **kwargs):
        super(CarsGround, self).__init__(scalers=scalers, file_signature=file_signature, **kwargs)
