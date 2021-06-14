import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os, sys
from basics import *


def prep(path):
	ids, vals = {}, {}
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith("ids.json"):
				ids[file[:-5]] = read_json(os.path.join(path, file))
			elif file.endswith("values.json"):
				vals[file[:-5]] = read_json(os.path.join(path, file))
	return ids, vals


def get_sent_len(sent_list):

	sent_lens = [len(max(sents, key=len).split()) for sents in sent_list]

	return sent_lens


def plot_distr(vals, name, phen, save, ss):
	print("plotting")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x_axis = np.arange(0, 5.1, 0.1)
	y_axis = [vals.count(round(score, 1)) for score in x_axis]
	ind = next(i for i, score in enumerate(y_axis) if score > 0)
	back_ind = next(i for i, score in enumerate(y_axis[::-1]) if score > 0)
	ax.set_ylabel('Test Cases (total: {})'.format(len(vals)))
	if ss == "sick":
		rel = "Relatedness"
	else:
		rel = "Similarity"
	ax.set_xlabel('Semantic {} Score'.format(rel))
	if phen:
		ax.set_title('Score Distribution for {}\n({})'.format(name, phen))
	else:
		ax.set_title('Score Distribution for {}\n'.format(name))
	#bars = ax.bar(x_axis[ind-2:-back_ind+2], y_axis[ind-2:-back_ind+2], width=0.1, color="seagreen")
	if back_ind - 3 > 0:
		y_axis = y_axis[ind-2:-back_ind+2]
		x_axis = x_axis[ind-2:-back_ind+2]
	else:
		y_axis = y_axis[ind-2:]
		x_axis = x_axis[ind-2:]
	bars = ax.bar(x_axis, y_axis, width=0.1, color="seagreen")
	for i, bar in enumerate(bars):
		if y_axis[i] > 0:
			bar.set_edgecolor("black")
			bar.set_linewidth(1.0)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
	if max(y_axis) > 10:
		ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
	else:
		ax.yaxis.set_major_locator(ticker.MultipleLocator(1))		
	#score_scale = np.arange(0, 5.2, 0.2)
	#plt.xticks(score_scale[int(ind/2)-2:-int(back_ind/2)+2])
	#plt.show()
	plt.savefig(save + "_" + name + ".png")
	print("saved in {}".format(save + "_" + name + ".png"))


def score_senlen(vals, sent_lens, name, phen, save):
	print("plotting")

	fig=plt.figure()
	ax=fig.add_subplot(111)
	plt.ylim(min(sent_lens)-2, max(sent_lens)+2) 
	mean = np.mean(np.array(sent_lens))
	ax.scatter(vals, sent_lens, color='navy', alpha=0.35)
	#ax.scatter(grades_range, boys_grades, color='b')
	ax.set_xlabel('Scores')
	ax.set_ylabel('Sentence Length')
	if phen:
		ax.set_title('Sentence Length and Score for {}\n({})'.format(name, phen))
	else:
		ax.set_title('Sentence Length and Score for {}\n'.format(name))

	ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

	plt.axhline(mean)
	plt.savefig(save + "_" + name + ".png")
	print("saved in {}".format(save + "_" + name + ".png"))
	#plt.show()


if __name__ == "__main__":
	# path = sys.argv[1]
	# my_dirs = ['Equivocation', 'Hyponym', 'Aspect', 'Subordinate Clause', 'Antonym', 'Passiv', 'Related Words', 'Article', 'Omission', 'Semantic Role', 'Preposition', 'Near Synonyms']
	# #my_dirs = ['Preposition']
	# for root, dirs, files in os.walk(path):
		# #print(dirs)
		# if root.split("/")[-1] in my_dirs:
			# ids, vals = prep(root + "/")
			#print(prep(root + "/"))
			#print(sents)
			#print(scores)
	final_scores_s, final_lens_s = [], []
	final_scores, final_lens = [], []
	ids, vals = read_json("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_ids_try.json"), read_json("/home/laura/Dokumente/CoLi/BA/Phens_GitHub/all_values_hc.json")
	for phen, ss in ids.items():
		for s_value, sub_phens in ss.items():
			s_ids = []
			for sub_phen, id_list in sub_phens.items():
				s_ids.extend(id_list)
				# print(id_list)
				score_list = [round(vals[idx][0], 1) for idx in id_list]
				print(score_list)
				plot_distr(score_list, phen, sub_phen, "/home/laura/ba_thesis/plots/" + s_value + "_" + sub_phen, s_value)
				sent_lens = get_sent_len([vals[idx][1] for idx in id_list])
				score_senlen(score_list, sent_lens, phen, sub_phen, "/home/laura/ba_thesis/plots/" + s_value + "_sentence_" + sub_phen)
			all_score_list = [round(vals[idx][0], 1) for idx in s_ids]
			if s_value == "sick":
				final_scores_s.extend(all_score_list)
			else:
				final_scores.extend(all_score_list)

			print(all_score_list)
			plot_distr(all_score_list, phen, False, "/home/laura/ba_thesis/plots/" + s_value + "_" + phen, s_value)
			all_sent_lens = get_sent_len([vals[idx][1] for idx in s_ids])
			if s_value == "sick":
				final_lens_s.extend(all_sent_lens)
			else:
				final_lens.extend(all_sent_lens)

			score_senlen(all_score_list, all_sent_lens, phen, False, "/home/laura/ba_thesis/plots/" + s_value + "_sentence_" + phen)

	# score_senlen(final_scores_s, final_lens_s, "All Test Cases (SICK)", False, "/home/laura/ba_thesis/plots/sick_overall_sentence")
	# score_senlen(final_scores, final_lens, "All Test Cases (STS)", False, "/home/laura/ba_thesis/plots/sts_overall_sentence")

	df = pd.DataFrame({'scores': final_scores_s,
                   'lens': final_lens_s})

	print(df['scores'].corr(df['lens'], method='spearman'))

	df = pd.DataFrame({'scores': final_scores,
                   'lens': final_lens})

	print(df['scores'].corr(df['lens'], method='spearman'))


			# for k,v in ids.items():
				# for key,value in vals.items():
					# if k[:3] == key[:3]:
						# # for i, jsn in enumerate(v):
						# for phen, id_list in v.items():
							# print(id_list)
							# score_list = [round(value[idx][0], 1) for idx in id_list]
							# print(score_list)
							# plot_distr(score_list, root.split("/")[-1], phen, root + "/plots/" + key)
							# sent_lens = get_sent_len([value[idx][1] for idx in id_list])
							# score_senlen(score_list, sent_lens, root.split("/")[-1], phen, root + "/plots/" + k)

	#plot_distr([
      #3.8,
      #3.3,
      #3.1,
      #4.0,
      #3.5,
      #3.6,
      #4.0,
      #4.1,
      #3.5,
      #3.915,
      #3.7,
      #3.5,
      #3.6,
      #3.1,
      #3.6,
      #3.585,
      #3.4,
      #3.6,
      #3.9,
      #3.7,
      #4.1,
      #3.915,
      #3.9,
      #4.2,
      #4.1,
      #3.6
    #], "Antonyms")
    #score_senlen([4.0, 3.6, 3.8, 3.3, 3.1, 4.0, 3.5, 3.6, 4.0, 4.1, 3.5], [10, 6, 9, 8, 12, 7, 9, 13, 11, 5, 6], "Beispiel")
