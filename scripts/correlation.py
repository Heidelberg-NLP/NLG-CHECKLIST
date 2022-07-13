import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def filter_nans(metric_dict):

    new_metric_dict = {}

    for k, v in metric_dict.items():
        if not [x for x in v if np.isnan(x)]:
            new_metric_dict[k] = v

    return new_metric_dict



def frame(metric_dict, save_png=False):
    """
    :param vectors: a numpy array containing nested numpy arrays (vectors) which represent the personal information of every participant
    :param csv: content of the csv file in a list of lists, each inner list representing a row of the csv file
    :param ind_list: list of the target indices
    """
    # f = ["age", "gender", "born\nregion", "lang\nregion", "score"]
    #print(metric_dict)
    for k, v in metric_dict.items():
    #    print(len(v))
        keys = list(metric_dict.keys())

    keys.sort(key=lambda v: v.upper())
    # feature_dic = {f[i]: [int(r[i]) for r in vectors] for i in list(range(4))}
    # feature_dic = {f[i]: [int(r[i]) for r in vectors] for i in list(range(len()))}

    cor = create_corr_matrix(metric_dict, keys)
    
    if save_png:
        plt.figure(figsize=(12, 10))

        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.savefig(save_png)
    else:
        return cor


def create_corr_matrix(dic, feature_names):
    """
    :param dic: dictionary, in which each key is the name of a feature and the value is a list
                containing all the feature values for all participants for one sentence
    :param feature_names: list of strings, which are the names of the features
    :return: a correlation matrix for the sentence
    """

    df = pd.DataFrame(dic, columns=feature_names)
    #print(df)
    df_rev = df
    #print(df_rev)
    cor = df.corr(method="spearman")
    #print(cor.to_string())
    cor = cor.round(3)
    return cor

