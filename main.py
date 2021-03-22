import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.moving_targets.learners import LogisticRegression
from src.moving_targets.macs import MACS
from src.moving_targets.masters import BalancedCounts
from src.moving_targets.metrics import Accuracy, ClassFrequenciesStd

SEED = 0

if __name__ == '__main__':
    df = pd.read_csv('res/iris.csv').sample(frac=1)
    x = df.drop('class', axis=1).values
    y = df['class'].astype('category').cat.codes.values

    scaler = MinMaxScaler()
    xt, xv, yt, yv = train_test_split(x, y)
    xt = scaler.fit_transform(xt)
    xv = scaler.transform(xv)

    configs = [
        dict(initial_step='pretraining', use_prob=False, alpha=1, beta=1),
        dict(initial_step='pretraining', use_prob=True, alpha=1, beta=1),
        dict(initial_step='projection', use_prob=False, alpha=1, beta=1),
        dict(initial_step='projection', use_prob=True, alpha=1, beta=1),
    ]

    for config in configs:
        metrics = [Accuracy(name='Acc'), ClassFrequenciesStd(num_classes=3, name='Std')]
        learner = LogisticRegression()
        master = BalancedCounts(num_classes=3)
        model = MACS(learner, master, metrics=metrics, **config)
        model.fit(xt, yt, iterations=15, val_data=(xv, yv))

        title = ' - '.join([f'{k.upper()}: {v}' for k, v in config.items()])
        print(title)
        train_score = '\n'.join([f'Train {k}: {v}' for k, v in model.evaluate(xt, yt).items()])
        print(train_score)
        val_score = '\n'.join([f'Val   {k}: {v}' for k, v in model.evaluate(xv, yv).items()])
        print(val_score)
        print()
