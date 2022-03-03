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
	dev_sum = sum([abs(values[i] - norm(gold, ss)) for i, gold in enumerate(golds)])

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


def create_table_average(phens, av_scores):
	columns = ["Metric"] + phens
	# id_dict = read_json(id_dict)
	rows, row = [], []
	for metric in sorted(av_scores.keys()):
		row.append(metric)
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


# def define_ranking(metrics, value_dict):




if __name__ == "__main__":
	test_cases = read_json("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_ids.json")
	vals = read_json("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_values.json")
	print("\n-----------------------------------------------------------------------------------------------")
	print("                                            OVERVIEW")
	print("-----------------------------------------------------------------------------------------------\n\n")
	print("Number of Test Cases by Phenomenon:\n")
	create_table_nums(test_cases)
	print("\n\nSICK Data Set\n\n")
	print("The Test Cases from the SICK Data Set were annotated with a Semantic Relatedness Score.\nThe score ranges from 1 (completely unrelated) to 5 (very related).\n\nSemantic Relatedness (SR) Statistics:\n")
	create_table_comps(test_cases, vals, "sick")
	print("\n\nSTS Data Set\n\n")
	print("The Test Cases from the SICK Data Set were annotated with a Semantic Similarity Score.\nThe score ranges from 0 (on different topics) to 5 (completely equivalent).\n\nSemantic Similarity (SS) Statistics:\n")
	create_table_comps(test_cases, vals, "sts")
	#create_table_nums("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_ids.json")
	#print("\nSICK:")
	#create_table_comps("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_ids.json", "/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_values.json", "sick")
	#print("\nSTS:")
	#create_table_comps("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_ids.json", "/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_values.json", "sts")
	#print(compute_stan_deviation(np.array([18, 21, 19, 26, 20])))
	#print(compute_stan_error(np.array([18, 21, 19, 26, 20])))
	#js_dict = read_json(sys.argv[1])
	# for root, dirs, files in os.walk("."):
		# for file in files:
			# if file.endswith("ids.json"):
				# js_dict_ids = read_json(file)
				# js_dict_vals = read_json(file[:-8] + "values.json")
				# print(file)
				# print("-------------------------")
				# total = []
				# # for sheet in js_dict_ids:
				# for k,v in js_dict_ids.items():
					# scores = [js_dict_vals[idx][0] for idx in v]
					# total.extend(scores)
					# print("-------------------------")
					# print("Analyzing {}\n{} test cases".format(k, len(scores)))
					# print("\nMean: {:>18}".format(str(round(compute_mean(scores), 2))))
					# print("\nMedian: {:>16}".format(str(round(compute_median(scores), 2))))
					# print("\nStandard deviation: {:>4}".format(str(round(compute_stan_deviation(scores), 2))))
					# print("\nStandard error: {:>8}".format(str(round(compute_stan_error(scores), 2))))
					# print("-------------------------")
# 
				# print("\n\nAnalyzing all instances\n{} test cases".format(len(total)))
				# print("\nMean: {:>18}".format(str(round(compute_mean(total), 2))))
				# print("\nMedian: {:>16}".format(str(round(compute_median(total), 2))))
				# print("\nStandard deviation: {:>4}".format(str(round(compute_stan_deviation(total), 2))))
				# print("\nStandard error: {:>8}".format(str(round(compute_stan_error(total), 2))))
# 				
				# print("\n\n")
