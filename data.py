"""
Loads data from the https://www.kaggle.com/c/asap-aes/overview competition.

Exposes the following variables:
train: training set
validation: validation set
test: testing set
"""
import os, argparse, csv
import requests
import numpy as np
import pandas as pd
from scipy import stats

FOLDER = "data/asap-aes"
asap_datasets = ["training_set_rel3", "valid_set", "test_set"]
ENCODING = "utf-8"

def parse_tsv_file(fname: str) -> pd.DataFrame:
    """ Loads a .tsv file into a structured pandas dataframe. """
    return pd.read_csv(f"{FOLDER}/{fname}.tsv", header=0, sep="\t", encoding="latin")

def summary_stats(fname: str, data: pd.DataFrame) -> None:
    """ Displays summary statistics about a dataframe. """
    print("-"*10 + f" {fname} " + "-"*10)
    print(f"Header: {', '.join(x for x in data.columns)}")
    print(f"Number of rows: {len(data)}")
    print(f"Average length of responses (sentences): {np.average([len(x.split('.')) for x in data['essay']]):.1f}")
    print(f"Average length of responses (words): {np.average([len(x.split()) for x in data['essay']]):.1f}")
    print(f"Average length of responses (characters): {np.average([len(x) for x in data['essay']]):.1f}")
    if "train" in fname:
        print(f"Average score: {np.average(data['domain1_score']):.3f}")
        print(f"Pearson's correlation between scorers: \
{stats.pearsonr(data['rater1_domain1'], data['rater2_domain1'])[0]:.3f}")
    print("-"*(22 + len(fname)) + "\n")

def parse_output(fname: str) -> None:
    """ Parses the output of SentEval. """
    df = pd.read_csv(fname, header=0, sep=";")
    tasks = [("Probing", 
              ["Depth", "BigramShift", "SubjNumber", "Tense", "CoordinationInversion", 
               "Length", "ObjNumber", "TopConstituents", "OddManOut", "WordContent"]),
             ("Supervised downstream", 
              ["SNLI", "SUBJ", "CR", "MR", "MPQA", "TREC", "SICKEntailment", 
               "SST2", "SST5", "MRPC", "STSBenchmark", "SICKRelatedness"]),
             ("Unsupervised downstream",
              ["STS12", "STS13", "STS14", "STS15", "STS16"])
            ]

    # according to SentEval paper, average the two scores for STSx
    for sts in tasks[-1][-1] + ["STSBenchmark", "SICKRelatedness"]:
        score1, score2 = eval(df[sts][0])
        df[sts] = (score1 + score2)/2

    for name, task in tasks:
        print(f"{name}: ")
        print(df[task], "\n")

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
    parser.add_argument("-o", "--output", dest="output", 
                        action="store_true", default=False,
                        help="display the result of SentEval")
    args = parser.parse_args()

    if args.download:
        if not os.path.exists(FOLDER):
            os.mkdir(FOLDER)

    if os.path.exists(FOLDER):
        data = {}
        for dataset in asap_datasets:
            data[dataset] = parse_tsv_file(dataset)
            if args.summary:
                summary_stats(dataset, data[dataset])

        train = data["training_set_rel3"]
        validation = data["valid_set"]
        test = data["test_set"]
    else:
        print("Datasets have not been downloaded.\nDid you run with -d?")

    if args.gen is not None:
        out_path = "word2mat/data/sentence.txt"

        with open("data/sentence.txt", encoding=ENCODING) as fin:
            with open(out_path, "w", encoding=ENCODING) as fout:
                for i in range(args.gen):
                    fout.write(fin.readline())

    if args.output:
        parse_output("word2mat/output/output.csv")

