import json, sys

with open(sys.argv[1], 'r') as f:
	first_dict = json.load(f)

with open(sys.argv[2], 'r') as s:
	second_dict = json.load(s)

for k, v in second_dict.items():
	for key, value in v.items():
		first_dict[k][key] = value


with open(sys.argv[3], 'w') as o:
	json.dump(first_dict, o)