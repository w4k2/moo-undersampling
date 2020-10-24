import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from itertools import combinations
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon

from metrics import confusion_matrix_scores

pd.set_option('max_colwidth', 40000)

REPORTS_DIR = 'reports'
if not os.path.isdir(REPORTS_DIR):
    os.mkdir(REPORTS_DIR)

RESULTS_DIR = 'results'
if not os.path.isdir(RESULTS_DIR):
    warnings.warn(f"{RESULTS_DIR} not found - please run `experiment.py` to perform experiment.")
    exit(1)

def find_results():
    records = []

    for f_name in os.listdir(RESULTS_DIR):
        print(f_name)
        if not f_name.endswith('.pkl'):
            continue

        ds = f_name.split('.')[0]

        with open(os.path.join(RESULTS_DIR, f_name), 'rb') as fp:
            results = pickle.load(fp)

        for res in results:
            scores = confusion_matrix_scores(res['cm'])

            records.append({
                "Dataset": ds,
                "Fold": res['Fold'],
                "Processing": res['Processing'],
                "Classifier": res['Classifier'],
                **scores
            })

    return records


def main():
    table = find_results()
    df = pd.DataFrame(table)

    # Store DF in csv for excel-style analysis
    # df.to_csv(os.path.join(RESULTS_DIR, ''))
    print(df.columns)
    methods = df["Processing"].unique()
    methods_map = {k: str(i) for i, k in enumerate(methods, 1)}

    print("Methods map:")
    print(methods_map)

    means_df = df.groupby(['Dataset', 'Classifier', 'Processing']).mean().drop(columns=['Fold'])

    melt_df = df.melt(['Dataset', 'Classifier', 'Processing', 'Fold'], var_name='Metric', value_name='Value')
    pivot_df = melt_df.pivot(index=['Dataset', 'Classifier', 'Processing', 'Metric'], columns="Fold")
    pivot_df.columns = pivot_df.columns.get_level_values(0)

    print(pivot_df)

    results = {}
    for key, grp_df in pivot_df.groupby(['Dataset', 'Classifier', 'Metric']):
        folds = grp_df.values
        processors = grp_df.index.get_level_values('Processing').values
        wlcx_mat = {p: [] for p in processors}

        for (p1, v1), (p2, v2) in combinations(zip(processors, folds), 2):
            w, p = wilcoxon(v1, v2, zero_method="zsplit")
            if p < 0.05:
                if np.mean(v1) > np.mean(v2):
                    wlcx_mat[p1].append(p2)
                else:
                    wlcx_mat[p2].append(p1)

        results[key] = wlcx_mat

    wilcoxon_means = pd.DataFrame(results).T.stack().unstack(level=2)
    wilcoxon_means = wilcoxon_means[means_df.columns]
    wilcoxon_means.index = wilcoxon_means.index.rename(['Dataset', 'Classifier', 'Processing'])
    print(wilcoxon_means)

    join_val = '\\makecell{' + np.char.array(np.vectorize(lambda _: f"{_:.3f}")(means_df)) + ' \\\\ \\scriptsize{' + np.char.array(np.vectorize(lambda _: ", ".join(sorted(map(lambda a: methods_map[a], _))))(wilcoxon_means)) + '}}'
    complete_table = pd.DataFrame(join_val, columns=means_df.columns, index=means_df.index)
    complete_table = complete_table.melt(var_name='Metric', value_name='Value', ignore_index=False).set_index('Metric', append=True).unstack(2)
    complete_table.columns = complete_table.columns.get_level_values(1)
    for key, grp_df in complete_table.groupby(['Classifier', 'Metric']):
        grp_df.index = grp_df.index.get_level_values(0)
        grp_df = grp_df.reindex(methods, axis=1)
        with open(os.path.join(REPORTS_DIR, f"results_{'_'.join(key)}.tex"), 'w') as fp:
            fp.write(grp_df.to_latex(escape=False))

    # Ranks
    for c_key, grp_df in means_df.groupby('Classifier'):
        ranks = grp_df.groupby(['Dataset']).rank().groupby(['Processing']).mean()
        print(ranks)

        ds_df = grp_df.melt(var_name='Metric', value_name='Value', ignore_index=False).set_index('Metric', append=True).unstack(0)

        results = {}
        for key, grp_df in ds_df.groupby(['Classifier', 'Metric']):
            folds = grp_df.values
            processors = grp_df.index.get_level_values('Processing').values

            wlcx_mat = {p: [] for p in processors}
            for (p1, v1), (p2, v2) in combinations(zip(processors, folds), 2):
                w, p = wilcoxon(v1, v2, zero_method="zsplit")
                if p < 0.05:
                    if np.mean(v1) > np.mean(v2):
                        wlcx_mat[p1].append(p2)
                    else:
                        wlcx_mat[p2].append(p1)

            results[key] = wlcx_mat

        wilcoxon_means = pd.DataFrame(results)
        wilcoxon_means.columns = wilcoxon_means.columns.droplevel()
        wilcoxon_means = wilcoxon_means[ranks.columns]
        print(wilcoxon_means)

        join_val = '\\makecell{' + np.char.array(np.vectorize(lambda _: f"{_:.3f}")(ranks)) + ' \\\\ \\scriptsize{' + np.char.array(np.vectorize(lambda _: ", ".join(sorted(map(lambda a: methods_map[a], _))))(wilcoxon_means)) + '}}'
        rank_df = pd.DataFrame(join_val, columns=ranks.columns, index=ranks.index).T
        rank_df = rank_df.reindex(methods, axis=1)

        with open(os.path.join(REPORTS_DIR, f'ranks_{c_key}.tex'), 'w') as fp:
            fp.write(rank_df.to_latex(escape=False))

    exit()


if __name__ == '__main__':
    main()
