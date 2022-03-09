import json, sys, time, os
import pandas as pd
import numpy as np
from basics import *
from standard import *
from evaluation import *
from correlation import *
import amr_diff
from style import HTML_AMR
from collections import defaultdict

m_lines = read_file("../metrics.txt")
wanted = [line.strip() for line in m_lines if line and not line.startswith("#")]	
y_lines = read_file("../my_metrics.txt")
your_wanted = [line.strip() for line in y_lines if line and not line.startswith("#")]
all_used_ids = {"sick": [], "sts": []}
test_cases = read_json("../data/ids_test_cases.json")
vals = read_json("../data/content_test_cases.json")
# def_metric_dict = read_json("../data/metric_scores_final.json")
def_metric_dict = read_json("../data/metric_scores_merged.json")
both = wanted + your_wanted
try:
	if sys.argv[1] == '-m':
		your_scores = {}
		your_wanted = []
	else:
		try:
			your_scores = read_json(sys.argv[1])
		except FileNotFoundError:
			print("Your file could not be found, please check the path!")
except IndexError:
	print("Please provide the path to your metrics or add the flag '-m' if you don't have a metric to test.")
	exit()
try:
	if sys.argv[2] == "-html":
		html = True
except IndexError:
	html = False
metric_dict = {}
for k,v in def_metric_dict.items():
	metric_dict[k] = v
if your_scores:
	for k,v in your_scores.items():
		for met in v:
			metric_dict[k][met] = your_scores[k][met]	
metric_f = defaultdict(list)
ks = []
for k in list(sorted(metric_dict)):
	ks.append(k)
	for met in metric_dict[k]:
		if met in both:
			metric_f[met].append(metric_dict[k][met])
stdize = lambda x: (np.array(x) - np.mean(x)) / np.std(x)
minmax = lambda x: (np.array(x) - np.min(x)) / (np.max(x) - np.min(x))
for met in metric_f:
	# if met == "Ann. Score":
	# 	print(metric_f[met])
	try:
		scores_stdized = minmax(stdize(metric_f[met]))
	except TypeError:
		print(met)
		print(metric_f[met])			
	for i in range(len(metric_f[met])):
		metric_dict[ks[i]][met] = scores_stdized[i]
all_average_sick, av_columns_sick = {}, []
all_average, av_columns = {}, []
corr_hj_sick, corr_hj = {}, {}
rank_list_sick, rank_list = [], []
for phenomenon in sorted(test_cases.keys()):
	if phenomenon != "Multiple Phenomena":
		for ss, sub_phens in test_cases[phenomenon].items():
			correlation, average = compute_av_scores([idx for k,v in sub_phens.items() for idx in v], wanted, your_wanted, metric_dict, ss)
			matrix = frame(correlation)
			both = wanted + your_wanted
			if ss == "sick":
				s, rel = "Relatedness", True
				av_columns_sick.append(phenomenon)
				for met in both:
					if met in all_average_sick:
						all_average_sick[met].append((round(float(average[met.strip()][0]), 3), round(float(average[met.strip()][1]), 2)))
					else:
						all_average_sick[met] = [(round(float(average[met.strip()][0]), 3), round(float(average[met.strip()][1]), 2))]
				ranking_dict_sick = define_ranking(wanted + your_wanted, correlation)
				rank_list_sick.append(ranking_dict_sick)
				unstack = matrix.unstack()
				for k,v in unstack["Ann. Score"].items():
					try:
						corr_hj_sick[k].append(v)
					except KeyError:
						corr_hj_sick[k] = [v]
			else:
				s, rel = "Similarity", False
				av_columns.append(phenomenon)
				for met in both:
					if met in all_average:
						all_average[met].append((round(float(average[met.strip()][0]), 3), round(float(average[met.strip()][1]), 2)))
					else:
						all_average[met] = [(round(float(average[met.strip()][0]), 3), round(float(average[met.strip()][1]), 2))]
				ranking_dict = define_ranking(wanted + your_wanted, correlation)
				rank_list.append(ranking_dict)
				unstack = matrix.unstack()
				for k,v in unstack["Ann. Score"].items():
					try:
						corr_hj[k].append(v)
					except KeyError:
						corr_hj[k] = [v]
			id_list = [x for phen, ids in sub_phens.items() for x in ids]
			for phen in sorted(sub_phens.keys()):
				all_used_ids[ss].extend([idx for idx in sub_phens[phen]])

all_corr_sick, all_av_sick = compute_av_scores(all_used_ids["sick"], wanted, your_wanted, metric_dict, "sick")
all_matrix_sick = frame(all_corr_sick)
all_corr_sts, all_av_sts = compute_av_scores(all_used_ids["sts"], wanted, your_wanted, metric_dict, "sts")
all_matrix_sts = frame(all_corr_sts)	

for met in both:
	all_average_sick[met].append((round(float(all_av_sick[met.strip()][0]), 3), round(float(all_av_sick[met.strip()][1]), 2)))
	all_average[met].append((round(float(all_av_sts[met.strip()][0]), 3), round(float(all_av_sts[met.strip()][1]), 2)))

ranking_dict = define_ranking(wanted + your_wanted, all_corr_sts)
rank_list.append(ranking_dict)
ranking_dict_sick = define_ranking(wanted + your_wanted, all_corr_sick)
rank_list_sick.append(ranking_dict_sick)

sorted(av_columns_sick)
av_columns_sick.append('Overall')
sorted(av_columns)
av_columns.append('Overall')

print("-----------------\n	 OVERALL\n-----------------\n\n")
print("SICK Data Set\n=============\n")
print("Ranking Table:\n-------------------")
print(create_ranking_table(av_columns_sick, wanted + your_wanted, rank_list_sick))
print("\nAverage Scores:\n-------------------")
print(create_table_average(av_columns_sick, all_average_sick, "sick"))
print("\nCorrelation Matrix:\n-------------------")
print(tabulate(all_matrix_sick, headers='keys', tablefmt='grid'))

si = all_matrix_sick.unstack()
for k,v in si["Ann. Score"].items():
	corr_hj_sick[k].append(v)

print(create_cor_table(corr_hj_sick, av_columns_sick))
for met in your_wanted:
	so = si[met].sort_values(kind="quicksort", ascending=False)
	print("\n\nOverall Correlation with Tested Score ({}):\n".format(met))
	print(so[1:].to_string())

print("\n\nSTS Data Set\n============\n")
print("Ranking Table:\n-------------------")
print(create_ranking_table(av_columns, wanted + your_wanted, rank_list))
print("\nAverage Scores:\n-------------------")
print(create_table_average(av_columns, all_average, "sts"))
print("\nCorrelation Matrix:\n-------------------")
print(tabulate(all_matrix_sts, headers='keys', tablefmt='grid'))

st = all_matrix_sts.unstack()
for k,v in st["Ann. Score"].items():
	corr_hj[k].append(v)

print(create_cor_table(corr_hj, av_columns))
for met in your_wanted:
	so = st[met].sort_values(kind="quicksort", ascending=False)
	print("\n\nOverall Correlation with Tested Score ({}):\n".format(met))
	print(so[1:].to_string())
	print("\n\n")