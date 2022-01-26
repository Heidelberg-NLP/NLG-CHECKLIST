import json
from basics import *

vals = read_json(val_file)

your_metric = {}

for idx, values in vals.items():
	# values[0] is the annotated score
	# values[1] is a list containing the two sentences (string) of the sentence pair
	# values[2] is a list containing the two AMR structures (list of strings) of the sentence pair
	# (values[3] is a description of the phenomena in the sentence pair, only for Multiple Phenomena!)

	your_metric[idx] = {}

	your_score = # implement your metric here

	your_metric[idx][your_scores_name] = your_score # substitute your_scores_name

	# maybe multiple metrics?

	# your_other_score = # implement your other metric here

	# your_metric[idx][your_other_scores_name] = your_other_score # substitute your_other_scores_name

convert_to_json(your_metric, your_amazing_filename) # substitute your_amazing_filename
	
