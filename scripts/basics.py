import json, os, re, tempfile


def read_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    return lines


def write_file(filename, lines):
    with open(filename, "w") as f:
        for line in lines:
            f.write(line)


def read_json(filename):
    with open(filename, "r") as f:
        js_dict = json.load(f, strict=False)

    return js_dict


def convert_to_json(js_dict, filename):
    with open(filename, "w") as f:
        json.dump(js_dict, f)


def read_amrs(file):
    with open(file, "r") as f:
        lines = f.readlines()
    amrs, amr = [], []
    for i, line in enumerate(lines):
        if line.strip():
            amr.append(lines[i])
        else:
            amrs.append(amr)
            amr = []

    return amrs


def make_tmp(pairs, nl="", tmp_name="temp123"):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.name = tmp_name
        tmp1 = tmp.name
        for line in pairs[0]:
            tmp.write(line+nl)
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.name = tmp_name + "_2"
        tmp2 = tmp.name
        for line in pairs[1]:
            tmp.write(line+nl)

    return tmp1, tmp2


def align_files(path):
    file_dict = {}
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.split(".")[1] == "input":
                file_dict[file_name] = re.sub("input", "gs", file_name)

    return file_dict


def read_smatch_json(filename):
    with open(filename, encoding='utf-8') as fn:
        smatch_dict = json.load(fn)

    cond_dict, idx = {}, "none"
    for k, v in smatch_dict.items():
        for res in v:
            for i, line in enumerate(res):
                print(line)
                if line.startswith("ID"):
                    idx = line.split()[1]
                elif line.startswith("Smatch"):
                    try:
                        cond_dict[idx] = line.strip()[-5:]
                    except KeyError:
                        print("no id for " + res[i+1])

    return cond_dict

