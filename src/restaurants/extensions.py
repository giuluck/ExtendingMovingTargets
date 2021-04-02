import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from tensorflow.keras.utils import to_categorical

from src.restaurants import data, plot
from src.models import MT, ExtensibleModel as Model


def ctr_estimate(model, avg_rating, num_reviews, dollar_rating):
    df = pd.DataFrame({'avg_rating': avg_rating, 'num_reviews': num_reviews})
    df[['D', 'DD', 'DDD', 'DDDD']] = to_categorical([len(dr) - 1 for dr in dollar_rating], num_classes=4)
    return model.predict(df)


def compute_ground_r2(model):
    pred = model.ctr_estimate(model.avg_ratings, model.num_reviews, model.dollar_ratings)
    return r2_score(model.ground_truths, pred)


def evaluation_summary(model, figsize=(14, 3), **kwargs):
    for title, (x, y) in kwargs.items():
        print(f'{roc_auc_score(y, model.predict(x)):.4} ({title} auc)', end=', ')
    print(f'{model.compute_ground_r2():.4} (ground r2)')
    plot.plot_ctr(model.ctr_estimate, title='Estimated CTR', figsize=figsize)
    plot.plot_ctr(data.ctr_estimate, title='Real CTR', figsize=figsize)


def on_training_end(model, macs, x, y, val_data, iteration):
    model.log(**{'learner/ground_r2': model.compute_ground_r2()})


Model.avg_ratings = np.repeat([np.linspace(1, 5, num=100)] * 100, 4)
Model.num_reviews = np.repeat(np.linspace(0, 200, num=100), 100 * 4)
Model.dollar_ratings = np.array(['D', 'DD', 'DDD', 'DDDD'] * (100 * 100))
Model.ground_truths = data.ctr_estimate(Model.avg_ratings, Model.num_reviews, Model.dollar_ratings)
Model.ctr_estimate = ctr_estimate
Model.compute_ground_r2 = compute_ground_r2
Model.evaluation_summary = evaluation_summary
MT.on_training_end = on_training_end
