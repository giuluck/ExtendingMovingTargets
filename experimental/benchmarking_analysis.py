"""Script to get LaTeX Tables from Benchmarking Results."""
import pandas as pd
import wandb

if __name__ == '__main__':
    def _float_format(fp: float, max_digits: int = 7) -> str:
        # up to 'max_digits' (point included), except for 0
        # still, take at least the number of digits up to the point
        fp_digits = f'{fp:.{max_digits}f}'
        integer_digits = fp_digits.find('.') - (1 if fp_digits.startswith('-') else 0)
        decimal_digits = max(0, max_digits - integer_digits - 1)
        fp = f'{fp:.{decimal_digits}f}'
        return '< 1e-5' if fp.strip('0.') == '' else fp


    COLUMNS = {
        'dataset': 'Dataset',
        'name': 'Model',
        'test_metric': 'Test Metric',
        'test_loss': 'Test Loss',
        'validation_metric': 'Val Metric',
        'validation_loss': 'Val Loss',
        'train_metric': 'Train Metric',
        'train_loss': 'Train Loss',
        'avg_violation': 'Avg. Violation',
        'pct_violation': 'Pct. Violations',
        'elapsed_time': 'Training Time'
    }

    MODELS = {
        'MLP': 'MLP',
        'SBR': 'SBR',
        'TFL': 'TFL',
        'MT 0.01': 'MT 0.01',
        'MT 0.1': 'MT 0.1',
        'MT 1.0': 'MT 1.0',
        'SBR (No Augmentation)': 'SBR (NA)',
        'MT 0.01 (No Augmentation)': 'MT 0.01 (NA)',
        'MT 0.1 (No Augmentation)': 'MT 0.1 (NA)',
        'MT 1.0 (No Augmentation)': 'MT 1.0 (NA)'
    }

    DATASETS = {
        'Synthetic': ('$R^2$', 'MSE'),
        'Cars Slim': ('$R^2$', 'MSE'),
        'Cars Full': ('$R^2$', 'MSE'),
        'Puzzles Slim': ('$R^2$', 'MSE'),
        'Puzzles Full': ('$R^2$', 'MSE'),
        'Restaurants': ('AUC', 'BCE'),
        'Law Slim': ('Acc.', 'BCE'),
        'Law Full': ('Acc.', 'BCE'),
        'Default Slim': ('Acc.', 'BCE'),
        'Default Full': ('Acc.', 'BCE')
    }

    # download results using wandb api
    runs = wandb.Api().runs('giuluck/mt_benchmarking')
    df = pd.DataFrame([{'name': run.name, **run.config, **run.summary} for run in runs])
    df = df[list(COLUMNS.keys())].rename(columns=COLUMNS)
    df['Dataset'] = pd.Categorical(df['Dataset'].map(str.title), categories=list(DATASETS.keys()), ordered=True)
    df['Model'] = pd.Categorical(df['Model'].map(MODELS), categories=list(MODELS.values()), ordered=True)
    df = df.sort_values(['Dataset', 'Model']).reset_index(drop=True)

    # group by dataset and return a latex table for each of them
    for dataset in df['Dataset'].unique():
        metric, loss = DATASETS[dataset]
        df_dataset = df[df['Dataset'] == dataset]
        # compute best values which will be used later to make them bold in the table
        whole_df = df_dataset.groupby('Model').mean().dropna().transpose().reset_index(drop=True)
        best = pd.concat((
            whole_df.iloc[:-3:2].max(axis=1),  # metrics
            whole_df.iloc[1:-3:2].min(axis=1),  # losses
            whole_df.drop(columns='TFL', errors='ignore').iloc[-3:-1].min(axis=1)  # violations
        )).sort_index()
        # divide between standard methods and methods with no augmentation for cars slim (univariate) dataset
        if dataset == 'Cars Slim':
            mask = df_dataset['Model'].map(lambda m: '(NA)' not in m)
            df_datasets = [df_dataset[mask == m] for m in [True, False]]
        else:
            df_datasets = [df_dataset]
        # df_datasets contains a single element except for cars slim
        for k, ddf in enumerate(df_datasets):
            ddf = ddf.groupby('Model').mean().dropna().rename(columns={
                'Test Metric': f'Test {metric}',
                'Test Loss': f'Test {loss}',
                'Val Metric': f'Val {metric}',
                'Val Loss': f'Val {loss}',
                'Train Metric': f'Train {metric}',
                'Train Loss': f'Train {loss}'
            }).transpose()
            # handle particular cases
            if dataset == 'Cars Slim' and k == 1:
                ddf.insert(0, 'MLP', '')
                ddf.insert(2, 'TFL', '')
            elif dataset in ['Law Full', 'Default Full']:
                ddf.insert(2, 'TFL', '/')
            # convert the dataset to latex, the do some manual postprocessing
            latex = ddf.to_latex(
                float_format=_float_format,
                caption=f'Results for {dataset.lower()} dataset',
                bold_rows=True,
                escape=False
            )
            latex = latex.replace('\\toprule', '\\hline')
            latex = latex.replace('\\midrule', '\\hline')
            latex = latex.replace('\\bottomrule', '\\hline')
            latex = latex.replace('\\textbf', '\\textit')
            latex = latex.replace('\\textit{Model}', '')
            for model in MODELS.values():
                latex = latex.replace(f'{model} &', '\\textbf{' + model + '} &')
                latex = latex.replace(f'{model} \\', '\\textbf{' + model + '} \\')
            latex = latex.split('\n')
            for i in range(1, len(latex) - 2):
                latex[i] = ('\t' if i < 4 or i == len(latex) - 3 else '\t\t') + latex[i]
            latex.insert(3, '\t\\small')
            # make best values bold line by line (this must me handled manually as formatters work for columns only)
            for i, v in best.items():
                v = _float_format(v)
                latex[i + 8] = latex[i + 8].replace(f'{v} &', '\\textbf{' + v + '} &')
                latex[i + 8] = latex[i + 8].replace(f'{v} \\', '\\textbf{' + v + '} \\')
            # save the table
            latex = '\n'.join(latex)
            name = dataset if len(df_datasets) == 1 else f'{dataset}_{k}'
            with open(f'../temp/{name}.txt', 'w') as f:
                f.write(latex)
