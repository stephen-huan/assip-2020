"""
Loads data from the https://www.kaggle.com/c/asap-aes/overview competition.
Data is unavailable from the original kaggle competition,
so a different repository is used:
https://github.com/pjankiewicz/asap-sas/tree/master/data/descriptions

Exposes the following variables:
train: training set
validation: validation set
test: testing set
corpus: large text corpus
"""
import os, argparse, csv
import requests
import numpy as np
from scipy import stats

uint = np.uint32
ustr = (np.unicode_, 1000)

REPO = "https://raw.githubusercontent.com/pjankiewicz/asap-sas/master/data/raw/" 
FOLDER = "data"
asap_datasets = ["train", "public_leaderboard", "private_leaderboard"]
datatypes = {"train": np.dtype({'names': ['id', 'essay','score1', 'score2', 'text'],
                                'formats': [uint, uint, uint, uint, ustr]}),
             "public_leaderboard": np.dtype({'names': ['id', 'essay', 'text'],
                                             'formats': [uint, uint, ustr]}),
             "private_leaderboard": np.dtype({'names': ['id', 'essay', 'text'],
                                             'formats': [uint, uint, ustr]})
            }
ENCODING = "utf-8"

def parse_url(fname: str) -> None:
    """ Downloads a dataset to the a folder. """
    r = requests.get(f"{REPO}{fname}.tsv")
    txt = r.text
    with open(f"{FOLDER}/{fname}.tsv", "w") as f:
        f.write(txt)

def parse_tsv_file(fname: str, dt: np.dtype) -> np.array:
    """ Loads a tsv file into a structured numpy array. """
    return np.loadtxt(f"{FOLDER}/{fname}.tsv", skiprows=1, delimiter="\t", dtype=dt)

def summary_stats(fname: str, data: np.array) -> None:
    """ Displays summary statistics about a numpy array. """
    print("-"*10 + f" {fname} " + "-"*10)
    print(f"Header: {', '.join(x for x in data.dtype.names)}")
    print(f"Number of rows: {len(data)}")
    print(f"Average length of responses (sentences): {np.average([len(x.split('.')) for x in data['text']]):.1f}")
    print(f"Average length of responses (words): {np.average([len(x.split()) for x in data['text']]):.1f}")
    print(f"Average length of responses (characters): {np.average([len(x) for x in data['text']]):.1f}")
    if fname == "train":
        print(f"Average score: {np.average((data['score1'] + data['score2'])/2):.3f}")
        print(f"Pearson's correlation between scorers: \
{stats.pearsonr(data['score1'], data['score2'])[0]:.3f}")
    print("-"*(22 + len(fname)) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data management script")
    parser.add_argument("-v", "--version", action="version", version="1.0.0")
    parser.add_argument("-d", "--download", dest="download", 
                        action="store_true", default=False,
                        help="download the various datasets needed")
    parser.add_argument("-s", "--summary", dest="summary", 
                        action="store_true", default=False,
                        help="display summary statistics about the datasets")
    parser.add_argument("-g", "--generate", dest="gen",
                        type=int, default=None, 
                        help="generate a text file with sentences")
    args = parser.parse_args()

    if args.download:
        if not os.path.exists(FOLDER):
            os.mkdir(FOLDER)

        for dataset in asap_datasets:
            parse_url(dataset)

    if os.path.exists(FOLDER):
        data = {}
        for dataset in asap_datasets:
            data[dataset] = parse_tsv_file(dataset, datatypes[dataset])
            if args.summary:
                summary_stats(dataset, data[dataset])
            # remove header row
            data[dataset] = data[dataset][1:]

        train = data["train"]
        validation = data["public_leaderboard"]
        test = data["private_leaderboard"]
    else:
        print("Datasets have not been downloaded.\nDid you run with -d?")

    if args.gen is not None:
        out_path = "word2mat/data/sentence.txt"

        with open("data/sentence.txt", encoding=ENCODING) as fin:
            with open(out_path, "w", encoding=ENCODING) as fout:
                for i in range(args.gen):
                    fout.write(fin.readline())

