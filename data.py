"""
Loads data from the https://www.kaggle.com/c/asap-aes/overview competition.

Exposes the following variables:
train: training set, tuple of (X, y) where X is input and y is labels
validation: validation set
test: testing set
"""
import sys, os, argparse, csv, glob, pickle
import requests
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.autograd import Variable
sys.path = ["word2mat"] + sys.path
import cbow

FOLDER = "data/asap-aes"
asap_datasets = ["training_set_rel3", "valid_set", "test_set"]
ENCODING = "utf-8"

# 1-indexed, gives the (min score, max score) for each essay set
score_range = [None, (2, 12), (1, 6), (0, 3), (0, 3), (0, 4), (0, 4), (0, 30), (0, 60), (1, 4)]
MAX_SCORE = 100

VOCAB_PATH = "word2mat/test_model/mode:random-w2m_type:hybrid-word_emb_dim:400-.vocab"
# Load vocabulary
VOCAB = pickle.load(open(VOCAB_PATH, "rb"))[0]

MODEL_PATH = "word2mat/test_model/mode:random-w2m_type:hybrid-word_emb_dim:400-.cbow_net_10"
MODEL = torch.load(MODEL_PATH)
if not isinstance(MODEL, cbow.CBOWNet):
    MODEL = MODEL.module
MODEL = MODEL.encoder

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

def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
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

def preprocess_test(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocesses the testing data into a standard form. """
    # essay set 2 has two different domains, add second domain as new rows
    set2 = df[df["essay_set"] == 2].copy()
    set2["domain1_predictionid"] = set2["domain2_predictionid"]
    # for simplicity, treat set 2's secondary domain as a new set
    set2["essay_set"] = 9
    df = df.append(set2)

    df["prediction_id"] = df["domain1_predictionid"].apply(lambda x: int(x))
    df = df.drop(columns=["essay_id", "domain1_predictionid", "domain2_predictionid"])

    return df

def make_batch(text: str, fancy: bool=True) -> list:
    """ Creates a batch from a text to feed into an embedding model. """
    sentences = cbow.sentenize(text)
    return sorted([cbow.tokenize(s, fancy) for s in sentences], key=lambda s: len(s))

def _batcher_helper(batch: list) -> np.array:
    sent, _ = cbow.get_index_batch(batch, VOCAB)
    sent_cuda = Variable(sent)
    sent_cuda = sent_cuda.t()
    MODEL.eval() # Deactivate drop-out and such
    try:
        embeddings = MODEL.forward(sent_cuda).data.cpu().numpy()
    except RuntimeError:
        embeddings = None 

    return sent.size()[1], embeddings

def random_identity() -> np.array:
    """ Returns the identity matrix plus normal distribution. """
    return  np.identity(20).reshape(400) + np.random.normal(0, 0.1, 400)

def embed(text: str) -> np.array:
    """ Embeds text as a vector. """
    batch = make_batch(text)
    size, array = _batcher_helper(batch)
    # if no words in the sentence, default to random
    if array is None:
        cmow, cbow = random_identity(), random_identity() 
    else:
        # because it is a hybrid model, array is shape(n, 800)
        # where n is the number of sentences.
        cbow, cmow = array[:, 0:400], array[:, 400:]
        cbow = np.sum(cbow, axis=0)
        cmow = torch.from_numpy(cmow).view(-1, size, 20, 20)
        cmow = MODEL.cmow_encoder._continual_multiplication(cmow).view(400).numpy()
    return np.append(cbow, cmow)

def one_hot_encode(size, i) -> np.array:
    """ One hot encode a value. """
    v = np.zeros(size)
    v[i] = 1
    return v

def train_vec(df: pd.Series) -> np.array:
    """ Returns a numpy array for a row in the training dataframe. """
    return np.concatenate((one_hot_encode(9, df["essay_set"] - 1), embed(df["essay"])))

def make_train_set(df: pd.DataFrame, pre=preprocess_train) -> np.array:
    """ Returns a finalized numpy array for training. """
    return np.array([train_vec(row) for i, row in pre(df).iterrows()])

def path_npy(fname: str) -> str:
    """ Returns a path for npy files. """
    return f"data/{model_name}-npy/{fname}.npy"

make_test_set = lambda df: make_train_set(df, preprocess_test) 

if os.path.exists(FOLDER):
    data = {}
    for dataset in asap_datasets:
        data[dataset] = parse_tsv_file(dataset)

    _train, _validation, _test = data["training_set_rel3"], data["valid_set"], data["test_set"] 
    _train_pre = preprocess_train(_train)

    model_name = MODEL_PATH.split("/")[-2] 
    arrays = glob.glob(path_npy("*"))
    if len(arrays) == 3:
        trainX, trainy = np.load(path_npy("train")), _train_pre["score"].to_numpy()
        # trainy = np.array([one_hot_encode(MAX_SCORE + 1, x) for x in trainy])
        train = (trainX, trainy)
        validation = np.load(path_npy("validation"))
        test = np.load(path_npy("test"))
    else:
        print("Datasets have not been processed. Processing...")
        if not os.path.exists(f"data/{model_name}"):
            os.mkdir(f"data/{model_name}")
        np.save(path_npy("train"), make_train_set(_train))
        np.save(path_npy("validation"), make_test_set(_validation))
        np.save(path_npy("test"), make_test_set(_test))
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
        summary_stats("training_set_rel3 (processed)", _train_pre)

    if args.gen is not None:
        out_path = "word2mat/data/sentence.txt"

        with open("data/sentence.txt", encoding=ENCODING) as fin:
            with open(out_path, "w", encoding=ENCODING) as fout:
                for i in range(args.gen):
                    fout.write(fin.readline())

    if args.output:
        parse_output("word2mat/output/")

