import sys, os, tempfile, subprocess, re, math, torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from datasets import load_metric
from sentence_transformers import SentenceTransformer, util
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

	# print("BLEU computed")

	return bleus

# evt. chrf und meteor (evt. auch smatch und s2match zsm fassen, wegen tmp file)
def compute_chrf(pairs, path):

	tmp1, tmp2 = make_tmp(pairs, nl="\n")

	chrf_score = subprocess.check_output(["python3", path, "-R", tmp1, "-H", tmp2, "-s"], encoding="utf-8")
	chrfs = [float(line.split("\t")[1].strip()) for line in chrf_score.split("\n")[:-1] if line[0].isdigit()]
	os.unlink(tmp1)
	os.unlink(tmp2)

	# print("chrF++ computed")

	return chrfs


def compute_meteor(pairs, path):

	tmp1, tmp2 = make_tmp(pairs, nl="\n")

	meteor_score = subprocess.check_output(["java", "-Xmx2G", "-jar",  path, tmp1, tmp2, "-l", "en", "-norm"], encoding="utf-8")
	meteors = [float(line.split()[3].strip()) for line in meteor_score.split("\n") if line.startswith("Segment")]
	# print(meteor_score)
	os.unlink(tmp1)
	os.unlink(tmp2)

	# print("Meteor computed")

	return meteors


def compute_smatch(pairs, path, s2=False):

	smatchs = []
	for i, sent in enumerate(pairs[0]):
		tmp1, tmp2 = make_tmp([["".join(sent)], ["".join(pairs[1][i])]], nl="\n")
		if s2:
			try:
				smatch_score = subprocess.check_output(["python3", path, "-f", tmp1, tmp2, "-vectors", "vectors/glove.6B.100d.txt"])
			except Exception as e:
				print(e)
				print(sent)
				smatch_score = "nan"
		else:
			try:
				smatch_score = subprocess.check_output(["python3", path, "-f", tmp1, tmp2])
			except Exception as e:
				print(e)
				print(sent)
				smatch_score = "nan"
		try:
			smatchs.append(float(smatch_score.split()[3].strip()))
		except IndexError:
			smatchs.append(smatch_score)
		os.unlink(tmp1)
		os.unlink(tmp2)

	# print("Smatch computed")

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

	# print("MF Score with beta={} computed".format(beta))

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

	# print("SBERT computed")

	return sberts


def compute_bert_score(pairs):

	bertscores = bertscore.compute(predictions=pairs[0], references=pairs[1], lang="en")

	# print("BERTScore computed")

	return bertscores

# -----------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
	# pairs = [["A woman peels a potato.", "A man is climbing a rope.", "A woman is putting on sun glasses.", "A man is slicing an onion.", "A man plays the guitar and sings.", "A man practicing boxing"], ["A woman is peeling a potato.", "A man climbs a rope.", "A woman puts on sun glasses.", "A man slices an onion.", "A man is singing and playing a guitar.", "A man practices boxing"]]
	# print("hopefully imported everything")
	# subs = ["Antonym", "Aspect", "Hyponym", "Article"]
	# subs = ["Article", "Near Synonyms", "Negation"]
	# subs = ["Near Synonyms", "Negation", "Omission", "Prepositions", "Passiv"]
	# subs = ["Word Families"]
	# subs = ["Omission", "Prepositions", "Passiv"]
	# subs = ["Semantic Roles", "Related Words", "Subordinate Clauses", "Multiple Phenomena", "Word Families"]
	# subs = ["Subordinate Clauses"]
	metric_dict = {}

	# miss = ["T2957sick", "T692sick", "T3909sick", "T4340sick", "T326sick", "T901sick", "T599sts", "D273sick", "D405sick", "D443sick", "T4170sick"]
	miss = ["T1591sick", "T2314sick", "T3385sick"]

	id_file = read_json(sys.argv[1]) 
	val_file = read_json(sys.argv[2]) 
	# for root, dirs, files in os.walk("./amr-devsuite/Phens_GitHub/"):
		# print(root)
		# print(dirs)
		# for subdir in dirs:
			# print(subdir)
			# if subdir in subs:
				# for subroot, subdirs, subfiles in os.walk(root + subdir):
					# for file in subfiles:
						# print(file)
						# if file.endswith("_ids.json"):
							# id_file = read_json(root + subdir + "/" + file)
							# val_file = read_json(root + subdir + "/" + file[:-8] + "values.json")
	for phen, ss in id_file.items():
		# ???
		print(phen)
		# if phen in subs:
		for s_value, sub_phens in ss.items():
			for sub_phen, ids in sub_phens.items():
				missed = [idx for idx in ids if idx in miss]
				# sents = [[val_file[idx][1][0] for idx in ids], [val_file[idx][1][1] for idx in ids]]
				sents = [[val_file[idx][1][0] for idx in ids if idx in miss], [val_file[idx][1][1] for idx in ids if idx in miss]]
				# amrs = [[val_file[idx][2][0] for idx in ids], [val_file[idx][2][1] for idx in ids]]
				amrs = [[val_file[idx][2][0] for idx in ids if idx in miss], [val_file[idx][2][1] for idx in ids if idx in miss]]
				# add sys
				# mf_scores = compute_mf_score(sents, "MFscore/mfscore_for_genSent_vs_refSent.sh")
				# mf_scores_md = compute_mf_score(sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="md")
				# mf_scores_fd = compute_mf_score(sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="fd")
				# mf_scores_mean = compute_mf_score(sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="mean")
				# mf_scores_form = compute_mf_score(sents, "MFscore/mfscore_for_genSent_vs_refSent.sh", beta="form")
				# bleus = compute_bleu(sents)
				# chrfs = compute_chrf(sents, "amr-devsuite/metrics/chrF++.py")
				# meteors = compute_meteor(sents, "meteor-1.5/meteor-1.5.jar")
				# sberts_rl = compute_sbert(sents, "stsb-roberta-large")
				# sberts_rb = compute_sbert(sents, "stsb-roberta-base-v2")
				# sberts_mpnet = compute_sbert(sents, "sstsb-mpnet-base-v2")
				# sberts_bl = compute_sbert(sents, "stsb-bert-large")
				# sberts_db = compute_sbert(sents, "stsb-distilbert-base")
				# s2matchs = compute_smatch(amrs, "amr-devsuite/metrics/smatch/s2match.py", s2=True)
				# smatchs = compute_smatch(amrs, "amr-devsuite/metrics/smatch/smatch.py")
				# bert_scores = compute_bert_score(sents)["f1"]		
				for i, idx in enumerate(ids):
				# for i, idx in enumerate(missed):
					metric_dict[idx] = {}
					# metric_dict[idx]["MF Score"] = mf_scores[i]
					# metric_dict[idx]["MF Score (M double)"] = mf_scores_md[i]
					# metric_dict[idx]["MF Score (F double)"] = mf_scores_fd[i]
					# metric_dict[idx]["MF Score (Meaning)"] = mf_scores_mean[i]
					# metric_dict[idx]["MF Score (Form)"] = mf_scores_form[i]
					# metric_dict[idx]["BLEU"] = bleus[i]
					# metric_dict[idx]["chrF++"] = chrfs[i]
					# metric_dict[idx]["Meteor"] = meteors[i]
					# metric_dict[idx]["S-BERT (roberta-large)"] = sberts_rl[i]
					# metric_dict[idx]["S-BERT (roberta-base)"] = sberts_rb[i]
					# metric_dict[idx]["S-BERT (mpnet-base)"] = sberts_mpnet[i]
					# metric_dict[idx]["S-BERT (bert-large)"] = sberts_bl[i]
					# metric_dict[idx]["S-BERT (distilbert-base)"] = sberts_db[i]
					metric_dict[idx]["S2match"] = s2matchs[i]
					metric_dict[idx]["Smatch"] = smatchs[i]
					# metric_dict[idx]["BERT Score"] = bert_scores[i]
		
	convert_to_json(metric_dict, "metric_scores_miss.json")
	# convert_to_json(metric_dict, "metric_scores_NP.json")
	# convert_to_json(metric_dict, "metric_scores_OP.json")
	# convert_to_json(metric_dict, "metric_scores_SC.json")


	# final_dict = {}
	# print("Starting evaluation…")
	# print("---------------------------------")
	# print("MF-Score:")
	# final_dict["MF-Score"] = compute_mf_score(pairs, "MFscore/mfscore_for_genSent_vs_refSent.sh")
	# print("Bleu:")
	# final_dict["Bleu"] = compute_bleu(pairs)
	# print("---------------------------------")
	# print("chrF++:")
	# final_dict["chrF++"] = compute_chrf(pairs, "amr-devsuite/metrics/chrF++.py")
	# print("---------------------------------")
	# print("Meteor:")
	# final_dict["Meteor"] = compute_meteor(pairs, "meteor-1.5/meteor-1.5.jar")
	# print("---------------------------------")	
	# print("S-Bert:")
	# final_dict["S-Bert"] = compute_sbert(pairs, "stsb-roberta-large")

	# print("---------------------------------")
	# first = read_amrs("source.amr")
	# print(first)
	# second = read_amrs("target.amr")
	# print("S2match:")
	# final_dict["S2match"] = compute_smatch([first, second], "amr-devsuite/metrics/smatch/s2match.py", s2=True)
	# print("---------------------------------")
	# print("Smatch:")
	# final_dict["Smatch"] = compute_smatch([first, second], "amr-devsuite/metrics/smatch/smatch.py")
	# print("---------------------------------")
	# print("BERTScore:")
	# final_dict["BERTScore"] = compute_bert_score(pairs)["f1"]
	# frame(final_dict)
	
	# print(alignment("aligned.txt"))

