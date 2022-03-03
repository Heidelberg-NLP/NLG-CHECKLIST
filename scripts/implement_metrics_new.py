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
	# pairs = [sents1, sents2]
	bleus = []
	smoothie = SmoothingFunction().method4
	for i, sent in enumerate(pairs[0]):
		bleus.append(float(sentence_bleu([sent.split()], pairs[1][i].split(), smoothing_function=smoothie)))
		# print('BLEU score -> {}'.format(sentence_bleu([sent.split()], pairs[1][i].split(), smoothing_function=smoothie)))

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
	# print(meteor_score)
	os.unlink(tmp1)
	os.unlink(tmp2)

	print("Meteor computed")

	return meteors


def compute_smatch(pairs, path, s2=False):

	smatchs = []

	tmp1, tmp2 = make_tmp([["".join(sent) for sent in pairs[0]], ["".join(pairs[1][i]) for i, sent in enumerate(pairs[0])]], nl="\n")
	if s2:
		try:
			smatch_score = subprocess.check_output(["python3", path, "-f", tmp1, tmp2, "-cutoff", "0.9", "-diffsense", "0.95", "-vectors", "vectors/glove.6B.300d.txt", "--ms"]).decode('ascii')
		except Exception as e:
			print(e)
			# print(sent)
			smatch_score = "nan"
	else:
		try:
			smatch_score = subprocess.check_output(["python3", path, "-f", tmp1, tmp2, "--ms"]).decode('ascii')
		except Exception as e:
			print(e)
			# print(sent)
			smatch_score = "nan"
	smatch_list = smatch_score.split('\n')

	for score in smatch_list:
		try:
			smatchs.append(float(score.split()[3].strip()))
		except IndexError:
			smatchs.append(float(score))

	os.unlink(tmp1)
	os.unlink(tmp2)


	return smatchs


def compute_mf_score(pairs, path, beta="harm", amr=False):

	mf_scores = []

	for i, sent in enumerate(pairs[0]):
		tmp1, tmp2 = make_tmp([[sent], [pairs[1][i]]], nl="\n")

		process = subprocess.Popen([path, tmp1, tmp2],
                         	stdout=subprocess.PIPE,
                         	stderr=subprocess.STDOUT)
		returncode = process.wait()
		with open("MFscore/evaluation-reports/report.txt", "r") as r:
			lines = r.readlines()

		mf_line = [line.split()[1] for line in lines if line.startswith("---->")]
		if beta == "harm":
			mf_scores.append(float(mf_line[0]))
		elif beta == "md":
			mf_scores.append(float(mf_line[1]))
		elif beta == "fd":
			mf_scores.append(float(mf_line[2]))
		elif beta == "mean":
			mf_scores.append(float(mf_line[3]))
		elif beta == "form":
			mf_scores.append(float(mf_line[4]))

	print("MF Score with beta={} computed".format(beta))

	return mf_scores


def compute_sbert(pairs, model):

	model = SentenceTransformer(model)
	sberts = []

	#Compute embedding for both lists
	embeddings1 = model.encode(pairs[0], convert_to_tensor=True)
	embeddings2 = model.encode(pairs[1], convert_to_tensor=True)

	#Compute cosine-similarits
	cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

	#Output the pairs with their score
	for i in range(len(pairs[0])):
	    # print("{} \t\t {} \t\t Score: {:.4f}".format(pairs[0][i], pairs[1][i], cosine_scores[i][i]))
	    sberts.append(cosine_scores[i][i].item())

	print("SBERT computed")

	return sberts


def compute_bert_score(pairs):

	bertscores = bertscore.compute(predictions=pairs[0], references=pairs[1], lang="en")

	print("BERTScore computed")

	return bertscores


def compute_mover_score(pairs, ngram):

	idf_dict_hyp = get_idf_dict(pairs[0]) # idf_dict_hyp = defaultdict(lambda: 1.)
	idf_dict_ref = get_idf_dict(pairs[1]) # idf_dict_ref = defaultdict(lambda: 1.)

	mover_scores = word_mover_score(pairs[1], pairs[0], idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=ngram, remove_subwords=False)

	return mover_scores


def compute_weisfelder_leman(pairs, path):

	mf_scores = []

	tmp1, tmp2 = make_tmp([["".join(sent) for sent in pairs[0]], ["".join(pairs[1][i]) for i, sent in enumerate(pairs[0])]], nl="\n")

	wl_scores = subprocess.check_output(['python3', path, '-a', tmp1, '-b', tmp2], encoding="utf-8")

	wls = [float(line.strip()) * -1 for line in wl_scores.split("\n") if line.strip()]
	# print(meteor_score)
	os.unlink(tmp1)
	os.unlink(tmp2)

	return wls

# -----------------------------------------------------------------------------------------------------------



if __name__ == "__main__":

	# metric_dict = {}
	with open("amr-devsuite/data/metric_scores_030322.json", "r") as j:
		metric_dict = json.load(j)

	# metric_dict = read_json("metric_scores_030122.json")
	id_file = read_json(sys.argv[1])
	val_file = read_json(sys.argv[2])
	print('read jsons')
	with open('outfile2.txt', 'w') as o:
		o.write('read jsons')

	all_sents, all_amrs, all_ids = [[], []], [[], []], []

	for phen, ss in id_file.items():
		for s_value, sub_phens in ss.items():
			for sub_phen, ids in sub_phens.items():
				sents = [[val_file[idx][1][0] for idx in ids], [val_file[idx][1][1] for idx in ids]]
				amrs = [[val_file[idx][2][0] for idx in ids], [val_file[idx][2][1] for idx in ids]]
				# add sys
				all_sents[0].extend(sents[0])
				all_sents[1].extend(sents[1])
				all_amrs[0].extend(amrs[0])
				all_amrs[1].extend(amrs[1])
				all_ids.extend(ids)
	with open('outfile2.txt', 'w') as o:
		o.write('sents, amrs and ids obtained, starting evaluation')
	print('sents, amrs and ids obtained, starting evaluation')
	#bleus = compute_bleu(all_sents)
	#chrfs = compute_chrf(all_sents, "amr-devsuite/metrics/chrF++.py")
	#meteors = compute_meteor(all_sents, "meteor-1.5/meteor-1.5.jar")
	with open('outfile2.txt', 'w') as o:
		o.write('done with text overlapping metrics')
	# mf_scores = compute_mf_score(all_sents, "MFscore/mfscore_for_genSent_vs_refSent.sh")
	# mf_scores_md = compute_mf_score(all_sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="md")
	# mf_scores_fd = compute_mf_score(all_sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="fd")
	# mf_scores_mean = compute_mf_score(all_sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="mean")
	# mf_scores_form = compute_mf_score(all_sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="form")
	with open('outfile2.txt', 'w') as o:
		o.write('done with MF score')
	#s2matchs = compute_smatch(all_amrs, "amr-devsuite/metrics/s2match.py", s2=True)
	#smatchs = compute_smatch(all_amrs, "amr-devsuite/metrics/smatch.py")
	with open('outfile2.txt', 'w') as o:
		o.write('done with Smatch and S2match')
	#sberts_rl = compute_sbert(all_sents, "stsb-roberta-large")
	#sberts_rb = compute_sbert(all_sents, "stsb-roberta-base-v2")
	#sberts_bl = compute_sbert(all_sents, "stsb-bert-large")
	#sberts_db = compute_sbert(all_sents, "stsb-distilbert-base")
	#bert_scores = compute_bert_score(all_sents)["f1"]
	print('metrics computed')
	with open('outfile2.txt', 'w') as o:
		o.write('done with SBERTs and BLEUScore')

	# mover_scores_uni = compute_mover_score(all_sents, 1)
	# mover_scores_bi = compute_mover_score(all_sents, 2)

	weisfelder_score = compute_weisfelder_leman(all_amrs, 'weisfeiler-leman-amr-metrics/src/main_wlk_wasser.py')


	for i, idx in enumerate(all_ids):
		# metric_dict[idx] = {}
		# metric_dict[idx]["BLEU"] = bleus[i]
		# metric_dict[idx]["chrF++"] = chrfs[i]
		# metric_dict[idx]["Meteor"] = meteors[i]
		# metric_dict[idx]["MF Score"] = mf_scores[i]
		# metric_dict[idx]["MF Score (M double)"] = mf_scores_md[i]
		# metric_dict[idx]["MF Score (F double)"] = mf_scores_fd[i]
		# metric_dict[idx]["MF Score (Meaning)"] = mf_scores_mean[i]
		# metric_dict[idx]["MF Score (Form)"] = mf_scores_form[i]
		# metric_dict[idx]["S2match"] = s2matchs[i]
		# metric_dict[idx]["Smatch"] = smatchs[i]
		# metric_dict[idx]["S-BERT (roberta-large)"] = sberts_rl[i]
		# metric_dict[idx]["S-BERT (roberta-base)"] = sberts_rb[i]
		# metric_dict[idx]["S-BERT (bert-large)"] = sberts_bl[i]
		# metric_dict[idx]["S-BERT (distilbert-base)"] = sberts_db[i]
		# metric_dict[idx]["BERT Score"] = bert_scores[i]
		# metric_dict[idx]["MoverScore uni"] = mover_scores_uni[i]
		# metric_dict[idx]["MoverScore bi"] = mover_scores_bi[i]
		metric_dict[idx]["Weisfelder Leman Score"] = weisfelder_score[i]

	convert_to_json(metric_dict, "metric_scores_wl.json")

