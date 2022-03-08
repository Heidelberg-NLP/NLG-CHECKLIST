import math, sys, os
import numpy as np
from tabulate import tabulate
from basics import *

def compute_mean(values):
	return np.mean(values)


def compute_median(values):
	return np.median(values)


def compute_stan_deviation(values, sample=True):
	mean = compute_mean(values)
	dev_sum = sum([(score - mean)**2 for score in values])
	
	if sample:
		n = len(values) - 1
	else:
		n = len(values)

	return math.sqrt(dev_sum / n)


def compute_stan_error(values):
	sd = compute_stan_deviation(values)

	return sd / math.sqrt(len(values))


def norm_deviation(values, golds, ss):

	# print(golds)
	# golds = [norm(gold, ss) for gold in golds]
	# print(golds)
	# print('-----------------------------------------')
	dev_sum = sum([abs(values[i] - gold) for i, gold in enumerate(golds)])
	
	return dev_sum / len(values)


def compute_av_scores(id_list, metrics, tested, metric_dict, ss):

	compute, average = {}, {}
	wanted = metrics + tested

	for idx in id_list:
		# if not [v for k, v in metric_dict[idx].items() if k in wanted and np.isnan(v)]:
		for key, value in metric_dict[idx].items():
			if key in metrics or key == "Ann. Score" or key in tested:
				try:
					compute[key].append(value)
				except KeyError:
					compute[key] = [value]

	for metric, scores in compute.items():
		average[metric] = (np.mean(np.array([score for score in scores if not isinstance(score, str) and not np.isnan(score)])), norm_deviation(scores, [score for score in compute["Ann. Score"] if not isinstance(score, str) and not np.isnan(score)], ss))

	return compute, average


def norm(value, ss):
	if ss == "sick":
		return (value - 1) / (5 - 1)
	else:
		return value / 5


def balance_len(vals):
	if len(vals[0]) < len(vals[1]):
		vals[0].extend(["\n\n"]*(len(vals[1]) - len(vals[0])))
	elif len(vals[0]) > len(vals[1]):
		vals[1].extend(["\n\n"]*(len(vals[0]) - len(vals[1])))

	return vals

