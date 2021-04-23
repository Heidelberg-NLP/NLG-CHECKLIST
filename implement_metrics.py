import sys, os, tempfile, subprocess, re, math
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def make_tmp(pairs):

	with tempfile.NamedTemporaryFile(delete=False) as tmp:
		tmp1 = tmp.name
		for line in pairs[0]:
			tmp.write(line.encode('utf-8'))
	with tempfile.NamedTemporaryFile(delete=False) as tmp:
		tmp2 = tmp.name
		for line in pairs[1]:
			tmp.write(line.encode('utf-8'))

	return tmp1, tmp2


def compute_bleu(pairs):
	# pairs = [sents1, sents2]
	bleus = []
	smoothie = SmoothingFunction().method4
	for i, sent in enumerate(pairs[0]):
		tmp1, tmp2 = make_tmp([[sent], [pair[1][i]]])
		bleus.append(sentence_bleu([pair[0].split()], pair[1].split(), smoothing_function=smoothie))
		print('BLEU score -> {}'.format(sentence_bleu([pair[0].split()], pair[1].split(), smoothing_function=smoothie)))
	print(bleus)
	return bleus

# evt. chrf und meteor (evt. auch smatch und s2match zsm fassen, wegen tmp file)
def compute_chrf(pairs, path):

	tmp1, tmp2 = make_tmp(pairs)
	chrf_score = subprocess.run(["python3", path, "-R", tmp1, "-H", tmp2, "-s"])
	print(chrf_score)
	chrfs = [line.split()[1].strip() for line in chrf_score.split("\n") if line[0].isdigit()]
	os.unlink(tmp1)
	os.unlink(tmp2)
	print(chrfs)
	return chrfs

def compute_meteor(pairs, path):

	tmp1, tmp2 = make_tmp(pairs)
	meteor_score = subprocess.run(["java", "-Xmx2G", "-jar",  path, tmp1, tmp2, "-l", "en", "-norm"])
	meteors = [line.split()[3].strip() for line in meteor_score.split("\n") if line.startswith("Segment")]
	print(meteor_score)
	os.unlink(tmp1)
	os.unlink(tmp2)

	print(meteors)
	return meteors


def compute_smatch(pairs, path, s2=False):

	smatchs = []
	for i, sent in enumerate(pairs[0]):
		tmp1, tmp2 = make_tmp([[sent], [pair[1][i]]])
		if s2:
			smatch_score = subprocess.run(["python3", path, "-f", files[0], files[1], "-vectors", "amr-metric-suite/vectors/glove.6B.100d.txt"])
		else:
			smatch_score = subprocess.run(["python3", path, "-f", files[0], files[1]])			
		smatchs.append(smatch_score.split()[3].strip())
		print(smatch_score)
		os.unlink(tmp1)
		os.unlink(tmp2)

	return smatchs


def compute_sbert(pairs, model):
	from sentence_transformers import SentenceTransformer, util

	model = SentenceTransformer(model)
	sberts = []

	#Compute embedding for both lists
	embeddings1 = model.encode(pairs[0], convert_to_tensor=True)
	embeddings2 = model.encode(pairs[1], convert_to_tensor=True)

	#Compute cosine-similarits
	cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

	#Output the pairs with their score
	for i in range(len(pairs[0])):
	    print("{} \t\t {} \t\t Score: {:.4f}".format(pairs[0][i], pairs[1][i], cosine_scores[i][i]))
	    sberts.append(cosine_scores[i][i])
	print(sberts)

	return sberts


def compute_bert_lcg(files, pairs, model):
	from amr_utils.amr_readers import AMR_Reader

	reader = AMR_Reader()
	amrs1 = reader.load(files[0], remove_wiki=True)
	amrs2 = reader.load(files[1], remove_wiki=True)

	for i, sent in enumerate(pairs[0]):
		inputs1 = tokenizer(sent, return_tensors="pt")
		outputs1 = model(**inputs)

		last_hidden_states1 = outputs1.last_hidden_state

		embeddings1 = get_embeddings(sent, amrs1, [], last_hidden_states1)

		inputs2 = tokenizer(pairs[1][i], return_tensors="pt")
		outputs2 = model(**inputs)

		last_hidden_states2 = outputs2.last_hidden_state

		embeddings2 = get_embeddings(pairs[1][i], amrs2, [], last_hidden_states2)


def get_embeddings(sent, amrs, embeddings, lhs):

	for j, token in enumerate(tokenizer.tokenize(sent)):
			in_amr = []
			if token in [value.split("-")[0] for value in amrs[i].nodes.values()]:
				embeddings.append(lhs[j+1])
				in_amr.append(token)
				# weitermachen
		for node in [value.split("-")[0] for value in amrs[i].nodes.values()]:
			if node not in in_amr:
				tok_node = tokenizer(node, return_tensors="pt")
				output_amr = model(**tok_node)
				hs_amr = output_amr.last_hidden_state
				print(hs_amr)
				embeddings.append(hs_amr[1])

#def compute_connectivity(embeddings):

if __name__ == "__main__":

	pairs = [["A woman peels a potato.", "A man is climbing a rope.", "A woman is putting on sun glasses.", "A man is slicing an onion.", "A man plays the guitar and sings.", "A man practicing boxing"], 
	["A woman is peeling a potato.", "A man climbs a rope.", "A woman puts on sunglasses.", "A man slices an onion.", "A man is singing and playing a guitar.", "A man practices boxing"]]
	
	print("Starting evaluation…")
	print("Bleu:")
	compute_bleu(pairs)
	print("---------------------------------")
	print("chrF++:")
	compute_chrf(pairs, "./metrics/chrF++.py")
	print("---------------------------------")
	print("Meteor:")
	compute_meteor(pairs, "./metrics/meteor-1.5/meteor-1.5.jar")
	print("---------------------------------")
	print("Smatch:")
	compute_smatch(["two.amr", "sec.amr"], "amr-metric-suite/py3-Smatch-and-S2match/smatch/smatch.py")
	print("---------------------------------")
	print("S2match:")
	compute_smatch(["two.amr", "sec.amr"], "amr-metric-suite/py3-Smatch-and-S2match/smatch/smatch.py", s2=True)
	print("---------------------------------")
	print("SBert:")
	compute_sbert(pairs, "stsb-roberta-large")
	#compute_sbert(pairs, "stsb-roberta-base")
	#compute_sbert(pairs, "stsb-bert-large")
	#compute_sbert(pairs, "stsb-distilbert-base")	
	print("---------------------------------")
