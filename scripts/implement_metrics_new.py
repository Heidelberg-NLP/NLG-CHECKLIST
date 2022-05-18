import sys, os, tempfile, subprocess, re, math, torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from datasets import load_metric
from sentence_transformers import SentenceTransformer, util
# from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
from correlation import *
from basics import *
import numpy as np
from scipy import spatial

bertscore = load_metric("bertscore")


def compute_bleu(pairs):
	bleus = []
	smoothie = SmoothingFunction().method4
	for i, sent in enumerate(pairs[0]):
		bleus.append((float(sentence_bleu([sent.split()], pairs[1][i].split(), smoothing_function=smoothie)) + float(sentence_bleu([pairs[1][i].split()], sent.split(), smoothing_function=smoothie))) / 2)

	print("BLEU computed")

	return bleus


def compute_chrf(pairs, path):

	tmp1, tmp2 = make_tmp(pairs, nl="\n")

	chrf_score = subprocess.check_output(["python3", path, "-R", tmp1, "-H", tmp2, "-s"], encoding="utf-8")
	chrfs = [float(line.split("\t")[1].strip()) for line in chrf_score.split("\n")[:-1] if line[0].isdigit()]
	os.unlink(tmp1)
	os.unlink(tmp2)

	print("chrF++ computed")

	return chrfs


def compute_meteor(pairs, path):

	tmp1, tmp2 = make_tmp(pairs, nl="\n")

	meteor_score = subprocess.check_output(["java", "-Xmx2G", "-jar",  path, tmp1, tmp2, "-l", "en", "-norm"], encoding="utf-8")
	meteors = [float(line.split()[3].strip()) for line in meteor_score.split("\n") if line.startswith("Segment")]
	os.unlink(tmp1)
	os.unlink(tmp2)

	print("Meteor computed")

	return meteors


def compute_smatch(pairs, path, s2=False):

	first_inp = pairs[0]
	print(len(first_inp))
	sec_inp = pairs[1]
	print(len(sec_inp))

	smatchs = []

	tmp1, tmp2 = make_tmp([["".join(sent) for sent in first_inp], ["".join(sec_inp[i]) for i, sent in enumerate(first_inp)]], nl="\n")
	if s2:
		try:
			smatch_score = subprocess.check_output(["python3", path, "-f", tmp1, tmp2, "-cutoff", "0.9", "-diffsense", "0.95", "-vectors", "vectors/glove.6B.300d.txt", "--ms"]).decode('ascii')
		except Exception as e:
			print(e)
			smatch_score = "nan"
	else:
		try:
			smatch_score = subprocess.check_output(["python3", path, "-f", tmp1, tmp2, "--ms"]).decode('ascii')
		except Exception as e:
			print(e)
	smatch_list = smatch_score.split('\n')

	for score in smatch_list:
		if score:
			try:
				smatchs.append(float(score.split()[3].strip()))
			except IndexError:
				smatchs.append(float(score))

	os.unlink(tmp1)
	os.unlink(tmp2)

	return smatchs


def compute_bert_score(pairs):

	bertscores = bertscore.compute(predictions=pairs[0], references=pairs[1], lang="en")
	bertscores = bertscore.compute(predictions=pairs[1], references=pairs[0], lang="en")

	print("BERTScore computed")

	return bertscores


def compute_wasserstein_weisfelder_leman(pairs, path):

	mf_scores = []

	tmp1, tmp2 = make_tmp([["".join(sent) for sent in pairs[0]], ["".join(pairs[1][i]) for i, sent in enumerate(pairs[0])]], nl="\n")

	wl_scores = subprocess.check_output(['python3', path, '-a', tmp1, '-b', tmp2], encoding="utf-8")

	wls = [float(line.strip()) + 1 for line in wl_scores.split("\n") if line.strip()]
	os.unlink(tmp1)
	os.unlink(tmp2)

	return wls


def compute_weisfelder_leman(pairs, path):

	mf_scores = []

	tmp1, tmp2 = make_tmp([["".join(sent) for sent in pairs[0]], ["".join(pairs[1][i]) for i, sent in enumerate(pairs[0])]], nl="\n")

	wl_scores = subprocess.check_output(['python3', path, '-a', tmp1, '-b', tmp2], encoding="utf-8")

	wls = [float(line.strip()) + 1 for line in wl_scores.split("\n") if line.strip()]
	os.unlink(tmp1)
	os.unlink(tmp2)

	return wls

# -----------------------------------------------------------------------------------------------------------



if __name__ == "__main__":

	# metric_dict = {}
	with open("amr-devsuite/data/metric_scores_final.json", "r") as j:
		metric_dict = json.load(j)

	# metric_dict = read_json("metric_scores_030122.json")
	id_file = read_json(sys.argv[1])
	val_file = read_json(sys.argv[2])

	parse_file = read_json("amr-devsuite/data/parsed_amrs.json")
	print('read jsons')
	with open('outfile2.txt', 'w') as o:
		o.write('read jsons')

	all_sents, all_amrs, all_ids, all_parsed = [[], []], [[], []], [], [[], []]

	for phen, ss in id_file.items():
		for s_value, sub_phens in ss.items():
			for sub_phen, ids in sub_phens.items():
				sents = [[val_file[idx][1][0] for idx in ids], [val_file[idx][1][1] for idx in ids]]
				amrs = [[val_file[idx][2][0] for idx in ids], [val_file[idx][2][1] for idx in ids]]
				parsed_amrs = [[parse_file[idx][0] for idx in ids], [parse_file[idx][1] for idx in ids]]
				# add sys
				all_sents[0].extend(sents[0])
				all_sents[1].extend(sents[1])
				all_amrs[0].extend(amrs[0])
				all_amrs[1].extend(amrs[1])
				all_parsed[0].extend(parsed_amrs[0])
				all_parsed[1].extend(parsed_amrs[1])
				all_ids.extend(ids)
	with open('outfile2.txt', 'w') as o:
		o.write('sents, amrs and ids obtained, starting evaluation')
	print('sents, amrs and ids obtained, starting evaluation')
	# bleus = compute_bleu(all_sents)
	# chrfs = compute_chrf(all_sents, "amr-devsuite/metrics/chrF++.py")
	# meteors = compute_meteor(all_sents, "meteor-1.5/meteor-1.5.jar")
	with open('outfile2.txt', 'w') as o:
		o.write('done with text overlapping metrics')
	mf_scores_sent = compute_smatch(all_parsed, "amr-devsuite/metrics/s2match.py", s2=True)
	mf_scores_amr1 = compute_smatch([all_parsed[0], all_amrs[1]], "amr-devsuite/metrics/s2match.py", s2=True)
	mf_scores_amr2 = compute_smatch([all_parsed[1], all_amrs[0]], "amr-devsuite/metrics/s2match.py", s2=True)
	mf_scores_amr = [(mf + mf_scores_amr2[i])/2 for i, mf in enumerate(mf_scores_amr1)]
	with open('outfile2.txt', 'w') as o:
		o.write('done with MF score')
	# s2matchs = compute_smatch(all_amrs, "amr-devsuite/metrics/s2match.py", s2=True)
	# smatchs = compute_smatch(all_amrs, "amr-devsuite/metrics/smatch.py")
	with open('outfile2.txt', 'w') as o:
		o.write('done with Smatch and S2match')
	#sberts_rl = compute_sbert(all_sents, "stsb-roberta-large")
	#sberts_rb = compute_sbert(all_sents, "stsb-roberta-base-v2")
	#sberts_bl = compute_sbert(all_sents, "stsb-bert-large")
	#sberts_db = compute_sbert(all_sents, "stsb-distilbert-base")
	# bert_scores = compute_bert_score(all_sents)["f1"]
	print('metrics computed')
	with open('outfile2.txt', 'w') as o:
		o.write('done with SBERTs and BLEUScore')

	# mover_scores_uni = compute_mover_score(all_sents, 1)
	# mover_scores_bi = compute_mover_score(all_sents, 2)

	# wasser_weisfelder_score = compute_wasserstein_weisfelder_leman(all_amrs, 'weisfeiler-leman-amr-metrics/src/main_wlk_wasser.py')
	wasser_weisfelder_score_parse = compute_wasserstein_weisfelder_leman(all_parsed, 'weisfeiler-leman-amr-metrics/src/main_wlk_wasser.py')
	wasser_weisfelder_score_parseg1 = compute_wasserstein_weisfelder_leman([all_parsed[0], all_amrs[1]], 'weisfeiler-leman-amr-metrics/src/main_wlk_wasser.py')
	wasser_weisfelder_score_parseg2 = compute_wasserstein_weisfelder_leman([all_parsed[1], all_amrs[0]], 'weisfeiler-leman-amr-metrics/src/main_wlk_wasser.py')
	wasser_weisfelder_score_parseg = [(wwlk + wasser_weisfelder_score_parseg2[i])/2 for i, wwlk in enumerate(wasser_weisfelder_score_parseg1)]
	# weisfelder_score = compute_weisfelder_leman(all_amrs, 'weisfeiler-leman-amr-metrics/src/main_wlk.py')


	for i, idx in enumerate(all_ids):
		# metric_dict[idx] = {}
		# metric_dict[idx]["BLEU"] = bleus[i]
		# metric_dict[idx]["chrF++"] = chrfs[i]
		# metric_dict[idx]["Meteor"] = meteors[i]
		# metric_dict[idx]["S2match"] = s2matchs[i]
		# metric_dict[idx]["Smatch"] = smatchs[i]
		# metric_dict[idx]["BERT Score"] = bert_scores[i]
		# metric_dict[idx]["WLK"] = weisfelder_score[i]
		# metric_dict[idx]["WWLK"] = wasser_weisfelder_score[i]
		metric_dict[idx]["WWLK Sent"] = wasser_weisfelder_score_parse[i]
		metric_dict[idx]["WWLK AMR"] = wasser_weisfelder_score_parseg[i]

	convert_to_json(metric_dict, "amr-devsuite/data/metric_scores_wwlk.json")

