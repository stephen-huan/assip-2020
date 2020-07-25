"""
Loads data from the https://www.kaggle.com/c/asap-aes/overview competition.

Exposes the following variables:
train: training set
validation: validation set
test: testing set
"""
import os, argparse, csv, glob
import requests
import numpy as np
import pandas as pd
from scipy import stats

FOLDER = "data/asap-aes"
asap_datasets = ["training_set_rel3", "valid_set", "test_set"]
ENCODING = "utf-8"

# 1-indexed, gives the (min score, max score) for each essay set
score_range = [None, (2, 12), (1, 6), (0, 3), (0, 3), (0, 4), (0, 4), (0, 30), (0, 60), (1, 4)]
MAX_SCORE = 100

def parse_tsv_file(fname: str) -> pd.DataFrame:
    """ Loads a .tsv file into a structured pandas dataframe. """
    return pd.read_csv(f"{FOLDER}/{fname}.tsv", header=0, sep="\t", encoding="latin")

def summary_stats(fname: str, df: pd.DataFrame) -> None:
    """ Displays summary statistics about a dataframe. """
    print("-"*10 + f" {fname} " + "-"*10)
    print(f"Header: {', '.join(x for x in df.columns)}")
    print(f"Number of rows: {len(df)}")
    print(f"Average length of responses (sentences): {np.average([len(x.split('.')) for x in df['essay']]):.1f}")
    print(f"Average length of responses (words): {np.average([len(x.split()) for x in df['essay']]):.1f}")
    print(f"Average length of responses (characters): {np.average([len(x) for x in df['essay']]):.1f}")
    if "processed" in fname:
        print(f"Average score: {np.average(df['score']):.3f}")
    elif "train" in fname:
        print(f"Average score: {np.average(df['domain1_score']):.3f}")
        print(f"Pearson's correlation between scorers: \
{stats.pearsonr(df['rater1_domain1'], df['rater2_domain1'])[0]:.3f}")
    print("-"*(22 + len(fname)) + "\n")

def parse_output(path: str) -> None:
    """ Parses the output of SentEval. """
    tasks = [("Probing", 
                ["Depth", "BigramShift", "SubjNumber", "Tense", "CoordinationInversion", 
                "Length", "ObjNumber", "TopConstituents", "OddManOut", "WordContent"]),
            ("Supervised downstream", 
                ["SNLI", "SUBJ", "CR", "MR", "MPQA", "TREC", "SICKEntailment", 
                "SST2", "SST5", "MRPC", "STSBenchmark", "SICKRelatedness"]),
            ("Unsupervised downstream",
                ["STS12", "STS13", "STS14", "STS15", "STS16"])
            ]

    table = None
    for fname in glob.glob(f"{path}*.csv"):
        df = pd.read_csv(fname, header=0, sep=";")
        df = df.rename(index={0: fname.split("/")[-1].split(".")[0]})

        # according to SentEval paper, average the two scores for STSx
        for sts in tasks[-1][-1] + ["STSBenchmark", "SICKRelatedness"]:
            score1, score2 = eval(df[sts][0])
            df[sts] = (score1 + score2)/2

        if table is None:
            table = df
        else:
            table = table.append(df)

    for name, task in tasks:
        print(f"{name}: ")
        print(table[task], "\n")

def scale_score(essay_set: int, score: int) -> int:
    """ Returns the score scaled to an integer between 0 and 100. """
    essay_set, score = int(essay_set), int(score)
    mn, mx = score_range[essay_set]
    return round(MAX_SCORE*(score - mn)/(mx - mn))

def unscale_score(essay_set: int, scaled: int) -> int:
    """ Returns the closest score within the range from a scaled score. """
    essay_set, score = int(essay_set), int(scaled)
    mn, mx = score_range[essay_set]
    return round((mx - mn)*scaled/MAX_SCORE)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocesses the training data into a standard form. """
    # essay set 2 has two different domains, add second domain as new rows
    set2 = df[df["essay_set"] == 2].copy()
    set2["domain1_score"] = set2["domain2_score"]
    # for simplicity, treat set 2's secondary domain as a new set
    set2["essay_set"] = 9
    df = df.append(set2)

    # scale domain1_score into score
    df["score"] = df[["essay_set", "domain1_score"]].apply(lambda x: scale_score(*x), axis=1)

    # remove unnecessary columns
    cols = ["essay_id"] + \
           [f"rater{i}_domain{j}" for i in range(1, 4) for j in range(1, 3)][:-1] + \
           [f"domain{i}_score" for i in range(1, 3)] + \
           [f"rater{i}_trait{j}" for i in range(1, 4) for j in range(1, 7)]
    df = df.drop(columns=cols)

    return df

if os.path.exists(FOLDER):
    data = {}
    for dataset in asap_datasets:
        data[dataset] = parse_tsv_file(dataset)

    train_processed = preprocess(data["training_set_rel3"])
    train = train_processed
    validation = data["valid_set"]
    test = data["test_set"]
else:
    print("Datasets have not been downloaded.\nDid you run with -d?")

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

    if args.summary:
        for dataset in asap_datasets:
            summary_stats(dataset, data[dataset])
        summary_stats("training_set_rel3 (processed)", train)

    if args.gen is not None:
        out_path = "word2mat/data/sentence.txt"

        with open("data/sentence.txt", encoding=ENCODING) as fin:
            with open(out_path, "w", encoding=ENCODING) as fout:
                for i in range(args.gen):
                    fout.write(fin.readline())

    if args.output:
        parse_output("word2mat/output/")

