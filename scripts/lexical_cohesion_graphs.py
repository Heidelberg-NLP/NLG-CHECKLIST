import sys, os, tempfile, subprocess, re, math, torch, spacy
from correlation import *
from basics import *
import numpy as np
from scipy import spatial
from nltk.stem.porter import *
from transformers import BertTokenizer, BertModel


def compute_glove_lcg(pairs, vectors, al_path, bp=False, bp_amr=False, filtered=False, star=False):

	glove_vecs = load_vecs(vectors)
	# nlp = spacy.load('en_core_web_lg',  disable=["parser", "ner"])
	sp = spacy.load('en_core_web_sm')
	conn_scores, conn_dicts = [], []

	# tmp1, tmp2 = make_tmp([["".join(amr) for amr in pairs[0]], ["".join(amr) for amr in pairs[1]]], nl="\n")

	# with open('/home/students/zeidler/ba/al_glove1', 'w') as a:
		# for x in ["".join(amr) for amr in pairs[0]]:
			# a.write(x)
			# a.write('\n')
# 
	# with open('/home/students/zeidler/ba/al_glove2', 'w') as a:
		# for x in ["".join(amr) for amr in pairs[1]]:
			# a.write(x)
			# a.write('\n')

	# get_jamr(tmp1, al_path + "1.out")
	# get_jamr(tmp2, al_path + "2.out")

	# get_jamr('/home/students/zeidler/ba/al_glove1', al_path + "1.out")
	# get_jamr('/home/students/zeidler/ba/al_glove2', al_path + "2.out")

	if filtered:
		count = 0
		aligned_token1 = alignment_filter(al_path + "1.out", count)
		aligned_token2 = alignment_filter(al_path + "2.out", count)
	else:
		aligned_token1 = alignment(al_path + "1.out")
		print(aligned_token1)
		aligned_token2 = alignment(al_path + "2.out")
		print(aligned_token2)

	for i, sent in enumerate(pairs[0]):
		if aligned_token1[i] == 0 or aligned_token2[i] == 0:
			conn_scores.append("nan")
			conn_dicts.append([{}, {}])
		else:
			vecs1 = get_gloves(aligned_token1[i], glove_vecs, sp)
			vecs2 = get_gloves(aligned_token2[i], glove_vecs, sp)

			if star:
				star1, star2 = find_stars(aligned_token1[i], aligned_token2[i])
				conn_dict1, conn1 = compute_star_connectivity(vecs1, aligned_token1[i], star1)
				conn_dict2, conn2 = compute_star_connectivity(vecs2, aligned_token2[i], star2)
			else:
				conn_dict1, conn1 = compute_connectivity(vecs1, aligned_token1[i])
				conn_dict2, conn2 = compute_connectivity(vecs2, aligned_token2[i])

			conn_scores.append(1 - math.sqrt((conn1 - conn2)**2))
			conn_dicts.append([conn_dict1, conn_dict2])

	return conn_scores


def compute_bert_lcg(sents, pairs, model, al_path, bp=False, bp_amr=False, filtered=False, star=False):

	tokenizer = BertTokenizer.from_pretrained(model)
	model = BertModel.from_pretrained(model, return_dict=True)

	# tmp1, tmp2 = make_tmp([["".join(amr) for amr in pairs[0]], ["".join(amr) for amr in pairs[1]]], nl="\n")

	# get_jamr(tmp1, al_path + "1.out")
	# get_jamr(tmp2, al_path + "2.out")

	if filtered:
		count = 0
		aligned_token1 = alignment_filter(al_path + "1.out", count)
		aligned_token2 = alignment_filter(al_path + "2.out", count)
	else:
		aligned_token1 = alignment(al_path + "1.out")
		aligned_token2 = alignment(al_path + "2.out")

	conn_scores, conn_dicts = [], []
	for i, sent in enumerate(sents[0]):
		if aligned_token1[i] == 0 or aligned_token2[i] == 0:
			conn_scores.append("nan")
			conn_dicts.append([{}, {}])
		else:
			inputs1 = tokenizer(sent, return_tensors="pt")
			outputs1 = model(**inputs1)

			last_hidden_states1 = outputs1.last_hidden_state

			embeddings1, aligned_order1 = get_embeddings(sent, aligned_token1[i], [], last_hidden_states1, tokenizer)

			inputs2 = tokenizer(sents[1][i], return_tensors="pt")
			outputs2 = model(**inputs2)

			last_hidden_states2 = outputs2.last_hidden_state

			embeddings2, aligned_order2 = get_embeddings(sents[1][i], aligned_token2[i], [], last_hidden_states2, tokenizer)

			if star:
				star1, star2 = find_stars(aligned_order1, aligned_order2)
				conn_dict1, conn1 = compute_star_connectivity([emb.detach().numpy() for emb in embeddings1], aligned_order1, star1)
				conn_dict2, conn2 = compute_star_connectivity([emb.detach().numpy() for emb in embeddings2], aligned_order2, star2)
			else:
				conn_dict1, conn1 = compute_connectivity([emb.detach().numpy() for emb in embeddings1], aligned_order1)
				conn_dict2, conn2 = compute_connectivity([emb.detach().numpy() for emb in embeddings2], aligned_order2)

			conn_scores.append(1 - math.sqrt((conn1 - conn2)**2))
			conn_dicts.append([conn_dict1, conn_dict2])

	return conn_scores


def load_vecs(fp):
    dic = {}
    with open(fp, "r") as f:
        for line in f:
            ls = line.split()
            word = ls[0]
            vec = np.array([float(x) for x in ls[1:]])
            dic[word] = vec
    return dic


def get_gloves(tokens, glove_vecs, sp):
	stemmer = PorterStemmer()
	text = sp(" ".join(tokens))
	vecs = []
	for i, token in enumerate(tokens):
		try:
			vecs.append(glove_vecs[token.lower()])
		except KeyError:
			try:
				vecs.append(glove_vecs[text[i].lemma_])
			except KeyError:
				try:
					vecs.append(glove_vecs[stemmer.stem(token.lower())])
				except KeyError:
					vecs.append(np.array([0]*300))

	return vecs


def get_jamr(filename, output_file):

	subprocess.run(["bash", "/home/students/zeidler/ba/amr-devsuite/jamr.sh", filename, output_file])


def alignment(align_file):
	# finds the words in the text that can be aligned to AMR nodes
	with open(align_file, "r") as af:
		lines = af.readlines()

	aligned_all, aligned = [], []
	for line in lines:
		if not line.strip():
			aligned_all.append(aligned)
			aligned = []
		if line.startswith("# ::tok"):
			token = line.split()[2:]
		elif line.startswith("# ::node"):
			try:
				if (int(line.split()[4].split("-")[0]) + 1) == int(line.split()[4].split("-")[1]):
					aligned.append(token[int(line.split()[4].split("-")[0])])
				else:
					for i in range((int(line.split()[4].split("-")[0])), int(line.split()[4].split("-")[1])):
						aligned.append(token[i])
			except IndexError:
				print("no alignment for {}".format(line.split()[3]))

	if aligned:
		print("this")
		aligned_all.append(aligned)

	# subprocess.run(["rm", align_file])

	return aligned_all[1:]


def alignment_filter(align_file, count):
	# finds the words in the text that can be aligned to AMR nodes
	with open(align_file, "r") as af:
		lines = af.readlines()
	aligned_all, aligned = [], []
	for line in lines:
		if not line.strip():
			if "nan" in aligned:
				aligned_all.append(0)
			else:
				aligned_all.append(aligned)
			aligned = []
		if line.startswith("# ::tok"):
			token = line.split()[2:]
		elif line.startswith("# ::node"):
			try:
				if (int(line.split()[4].split("-")[0]) + 1) == int(line.split()[4].split("-")[1]):
					aligned.append(token[int(line.split()[4].split("-")[0])])
				else:
					for i in range((int(line.split()[4].split("-")[0])), int(line.split()[4].split("-")[1])):
						aligned.append(token[i])
			except IndexError:
				print("no alignment for {}".format(line.split()[3]))
				count += 1
				aligned.append("nan")

	if aligned:
		aligned_all.append(aligned)

	# subprocess.run(["rm", align_file])
	print("{} testcases could not be aligned".format(count))
	return aligned_all[1:]


def get_embeddings(sent, amr_aligned, embeddings, lhs, tokenizer):

	aligned_order = []
	for j, token in enumerate(tokenizer.tokenize(sent)):
		if token in amr_aligned:
			embeddings.append(lhs[0][j+1])
			aligned_order.append(token)
		else:
			if token.startswith("##"):
				if tokenizer.tokenize(sent)[j-1] + token[2:] in amr_aligned:
					embeddings.append(lhs[0][j+1])
					aligned_order.append(tokenizer.tokenize(sent)[j-1] + token[2:])
	return embeddings, aligned_order


def compute_connectivity(embeddings, token):

	conn_dict = {}
	for i, embedding in enumerate(embeddings):
		for j, emb in enumerate(embeddings):
			if token[i] not in conn_dict:
				if token[j] in conn_dict:
					if conn_dict[token[j]][1] != token[i]:
						if token[j] != token[i]:
							conn_dict[token[i]] = (compute_cosine(embedding, emb), token[j])
				else:
					if token[j] != token[i]:
						conn_dict[token[i]] = (compute_cosine(embedding, emb), token[j])
			else:

				if token[j] not in conn_dict:
					if conn_dict[token[i]][1] != token[j]:
						if token[j] != token[i]:
							conn_dict[token[j]] = (compute_cosine(embedding, emb), token[i])
					
	conn = np.mean(np.array([sim[0] for k, sim in conn_dict.items()]))

	return conn_dict, conn


def find_stars(ref_token, can_token):

	return [token for token in ref_token if token not in can_token], [token for token in can_token if token not in ref_token]


def compute_star_connectivity(embeddings, token, stars):

	conn_dict = {}
	if stars:
		for star in stars:
			for i, tok in enumerate(token):
				if not star == tok:
					conn_dict[star + str(i)] = (compute_cosine(embeddings[i], embeddings[token.index(star)]), tok)

		conn = np.mean(np.array([sim[0] for k, sim in conn_dict.items()]))
	else:
		conn_dict = {}
		conn = 1

	return conn_dict, conn


def compute_cosine(vector1, vector2):

	return 1 - spatial.distance.cosine(vector1, vector2)


if __name__ == "__main__":
	metric_dict = {}
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
	all_sents, all_amrs, all_ids = [[], []], [[], []], []
	for phen, ss in id_file.items():
		# ???
		print(phen)
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
	lcg_gloves = compute_glove_lcg(all_amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned")	
	lcg_gloves_filter = compute_glove_lcg(all_amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", filtered=True)	
	# lcg_gloves_bp = compute_glove_lcg(amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", bp=True)	
	# lcg_gloves_bpamr = compute_glove_lcg(amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", bp_amr=True)
	lcg_gloves_star = compute_glove_lcg(all_amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", star=True)
	lcg_gloves_starf = compute_glove_lcg(all_amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", star=True, filtered=True)
	# lcg_gloves_starbp = compute_glove_lcg(amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", star=True, bp=True)
	# lcg_gloves_starbpamr = compute_glove_lcg(amrs, "vectors/glove.6B.300d.txt", "/home/students/zeidler/ba/aligned", star=True, bp_amr=True)
	lcgs = compute_bert_lcg(all_sents, all_amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned")
	lcgs_filter = compute_bert_lcg(all_sents, all_amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", filtered=True)
	# lcgs_bp = compute_bert_lcg(sents, amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", bp=True)
	# lcgs_bpamr = compute_bert_lcg(sents, amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", bp_amr=True)					
	lcgs_star = compute_bert_lcg(all_sents, all_amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", star=True)					
	lcgs_starf = compute_bert_lcg(all_sents, all_amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", star=True, filtered=True)					
	# lcgs_starbp = compute_bert_lcg(sents, amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", star=True, bp=True)					
	# lcgs_starbpamr = compute_bert_lcg(sents, amrs, 'bert-large-uncased',"/home/students/zeidler/ba/aligned", star=True, bp_amr=True)					
	for i, idx in enumerate(all_ids):
		metric_dict[idx] = {}
		metric_dict[idx]["Glove LCG"] = lcg_gloves[i]
		metric_dict[idx]["Glove LCG (Filter)"] = lcg_gloves_filter[i]
		# metric_dict[idx]["Glove LCG (BP: Sentence)"] = lcg_gloves_bp[i]
		# metric_dict[idx]["Glove LCG (BP: AMR)"] = lcg_gloves_bpamr[i]
		metric_dict[idx]["Glove LCG (Star)"] = lcg_gloves_star[i]
		metric_dict[idx]["Glove LCG (StarF)"] = lcg_gloves_starf[i]
		# metric_dict[idx]["Glove LCG (Star, BPS)"] = lcg_gloves_starbp[i]
		# metric_dict[idx]["Glove LCG (Star, BPAMR)"] = lcg_gloves_starbpamr[i]
		metric_dict[idx]["LCG"] = lcgs[i]
		metric_dict[idx]["LCG (Filter)"] = lcgs_filter[i]
		# metric_dict[idx]["LCG (BP: Sentence)"] = lcgs_bp[i]
		# metric_dict[idx]["LCG (BP: AMR)"] = lcgs_bpamr[i]
		metric_dict[idx]["LCG (Star)"] = lcgs_star[i]
		metric_dict[idx]["LCG (StarF)"] = lcgs_starf[i]
		# metric_dict[idx]["LCG (Star, BPS)"] = lcgs_starbp[i]
		# metric_dict[idx]["LCG (Star, BPAMR)"] = lcgs_starbpamr[i]

	convert_to_json(metric_dict, "metric_scores_LCG.json")
