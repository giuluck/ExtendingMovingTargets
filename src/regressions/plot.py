import seaborn as sns
import matplotlib.pyplot as plt

from src.util.preprocessing import Scaler


def plot_cars(scalers=None, figsize=(14, 4), tight_layout=True, **kwargs):
    if len(kwargs) > 0:
        info = []
        x_scaler, y_scaler = (Scaler.get_default(1), Scaler.get_default(1)) if scalers is None else scalers
        _, axes = plt.subplots(1, len(kwargs), sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
        for idx, (title, (x, y)) in enumerate(kwargs.items()):
            info.append(f'{len(x)} {title} samples')
            x = x_scaler.invert(x['price'])
            y = y_scaler.invert(y)
            sns.scatterplot(x=x, y=y, ax=axes[idx]).set(xlabel='price', ylabel='sales', title=title.capitalize())
        print(', '.join(info))
