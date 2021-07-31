import pandas as pd
import scipy.stats
import scikit_posthocs as sp
import datetime
import glob
import os


def create_significance_plot(data, metric):
    """
    Create plot of significant values
    @param data:
    @param metric: string, the metric name to plot significant by
    """
    pc = sp.posthoc_conover(data, val_col=metric, group_col='Algorithm Name', p_adjust='holm')
    print(pc)
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                    'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    sp.sign_plot(pc, **heatmap_args)


def read_results():
    """
    Read all the result file and concat them to one df
    @return: dataframe
    """
    all_results = []
    for root, dirs, files in os.walk("./"):
        for file in files:
            if file.endswith(".csv"):
                data = pd.read_csv(file, index_col=0)
                data.reset_index(inplace=True)
                data.rename(columns={'index': 'Fold'}, inplace=True)
                all_results.append(data)
    df = pd.concat(all_results)
    return df


def calculate_statistics(metric='AUC'):
    """
    Sum all the cv over the given metric values and do a Fridman test
    @param metric: string, the value of metric to check significant by
    @return: list of statistic: the test statistic, correcting for ties, P value
    """
    metrics = read_results()

    metrics = metrics[["Algorithm Name", "Dataset Name", metric]]
    metric_mean = metrics.groupby(["Algorithm Name", "Dataset Name"]).mean().reset_index()
    metrics = metric_mean[["Algorithm Name", metric]].groupby("Algorithm Name")[metric].apply(list)
    values = metrics.values
    baseline = values[0]
    improved = values[1]
    paper = values[2]
    result = scipy.stats.friedmanchisquare(paper, improved, baseline)
    create_significance_plot(metric_mean, metric)
    metric_mean.groupby(["Algorithm Name"]).mean().reset_index().to_csv(f"../data/results/algo_{metric}_mean.csv")
    return result


