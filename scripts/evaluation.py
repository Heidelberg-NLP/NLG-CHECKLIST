import math, sys, os
import numpy as np
import pandas as pd
from tabulate import tabulate
from basics import *
from standard import *


def create_table_nums(id_dict):

	columns = ["Phenomenon", "Test Cases (SICK)", "Test Cases (STS)"]
	# id_dict = read_json(id_dict)
	rows, row, count = [], [], 0
	for phen in sorted(id_dict.keys()):
		row.append(phen)
		for s_value in ["sick", "sts"]:
			try:
				for sub_phen in sorted(id_dict[phen][s_value].keys()):
					count += len(id_dict[phen][s_value][sub_phen])
				row.append(count)
				count = 0
			except KeyError:
				row.append(0)
		rows.append(row)
		row = []

	print(tabulate(rows, columns, tablefmt="latex"))


def create_table_subnums(id_dict, phen, s_value):

	columns = ["Phenomenon", "Test Cases"]
	# id_dict = read_json(id_dict)
	rows, row = [], []
	for sub_phen in sorted(id_dict[phen][s_value].keys()):
		row.append(sub_phen)
		row.append(len(id_dict[phen][s_value][sub_phen]))
		rows.append(row)
		row = []

	print(tabulate(rows, columns, tablefmt="latex"))


def create_table_scores(metrics, scores):

	columns = ["Metric", "Average Score"]
	# id_dict = read_json(id_dict)
	rows, row = [], []
	for i, metric in enumerate(metrics):
		row.append(metric)
		row.append(scores[i])
		rows.append(row)
		row = []

	return tabulate(rows, columns, tablefmt="latex")


def create_table_average(phens, av_scores, ss):
	columns = ["Metric"] + phens
	# id_dict = read_json(id_dict)
	rows, row = [], []
	for metric in sorted(av_scores.keys()):
		row.append(metric)
		if metric == "Ann. Score":
			row.extend(['{}'.format(norm(av[0], ss)) for av in av_scores[metric]])
		else:
			row.extend(['{} ± {}'.format(av[0], av[1]) for av in av_scores[metric]])
		rows.append(row)
		row = []

	return tabulate(rows, columns, tablefmt="latex")


def create_cor_table(corr_dict, cols):

	columns = ["Metric"] + cols
	rows, row = [], []
	for k, v in corr_dict.items():
		row = [k] + v
		rows.append(row)
		row = []
	return tabulate(rows, columns, tablefmt="latex")


def create_table_comps(id_dict, val_dict, s_value):

	columns = ["Phenomenon", "SR Mean", "SR Median", "Standard Deviation", "Standard Error"]
	# id_dict = read_json(id_dict)
	# val_dict = read_json(val_dict)
	if s_value == "sts":
		columns[1] = "SS Mean"
		columns[2] = "SS Median"

	rows, row = [], []
	for phen in sorted(id_dict.keys()):
		phen_ids = []
		row.append(phen)
		try:
			for sub_phen in sorted(id_dict[phen][s_value].keys()):
				phen_ids.extend(id_dict[phen][s_value][sub_phen])
			vals = [val_dict[idx][0] for idx in phen_ids]
			row.append(round(compute_mean(np.array(vals)), 3))
			row.append(round(compute_median(np.array(vals)), 3))
			row.append(round(compute_stan_deviation(vals), 3))
			row.append(round(compute_stan_error(vals), 3))
			rows.append(row)
			row = []
		except KeyError:
			row = []
			continue

	print(tabulate(rows, columns, tablefmt="latex"))


def define_ranking(metrics, value_dict):

	score_dict = {}
	val_dict = value_dict.copy()
	value_dict.pop('Ann. Score', None)
	metrics.remove('Ann. Score')

	for metric in metrics:
		count, score = 0, 0
		for i, value in enumerate(value_dict[metric]):
			for j, val in enumerate(value_dict[metric]):
				if val_dict['Ann. Score'][i] == val_dict['Ann. Score'][j] and value == val:
					score += 1
				elif val_dict['Ann. Score'][i] < val_dict['Ann. Score'][j] and value < val:
					score += 1
				elif val_dict['Ann. Score'][i] > val_dict['Ann. Score'][j] and value > val:
					score += 1
				count += 1

		score_dict[metric] = round(score / count, 3)

	return score_dict


def create_ranking_table(phenomena, metrics, rank_list):

	metrics.remove('Ann. Score')
	df = pd.DataFrame()
	df['Metrics'] = sorted(metrics)
	for i, phen in enumerate(phenomena):
		# df[phen] = rank_list[i]
		column = []
		for j, metric in enumerate(metrics):
			column.append(rank_list[i][metric])
		#df[phen] = sorted(rank_list[i].items(), key=lambda x:x[1], reverse=True)
		df[phen] = column

	return tabulate(df, headers=phenomena, tablefmt='latex')