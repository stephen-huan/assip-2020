# ASSIP 2020

## Setup
This project uses [word2mat](https://github.com/florianmai/word2mat),
specifically our [fork](https://github.com/stephen-huan/word2mat).

### Software
To install the dependencies, use [pipenv](https://pipenv.pypa.io/en/latest/): 
```bash
pipenv install
pipenv run pip install -r word2mat/requirements.txt
pipenv run python -c "import nltk; nltk.download('punkt')"
```

### Data
Datasets used are from the Hewlett Foundation's Automated Essay Scoring
competition on [Kaggle](https://www.kaggle.com/c/asap-aes/overview).

To download the datasets, run 
```bash
python data.py -d 
```

This will load the data in a new folder called `data`.
- train.tsv: training set 
(used to fit the model to the data, e.g. adjusted weights for a neural network)
- public_leaderboard.tsv: validation set 
(used to determine hyperparameters of the model, e.g. activation function or depth)
- private_leaderboard.tsv: test set 
(used to test final model)

The other dataset used is the [UMBC Corpus](https://ebiquity.umbc.edu/resource/html/id/351).
It comes as a .tar.gz file, which can be extracted
with `gzip -d` and `tar -xf`, in that order,
or just `tar -xzf` (where the `-z` option indicates gzip).

Finally, run the data preprocessing script from word2mat.
```bash
pipenv run python word2mat/data/extract_umbc.py data/webbase_all/ data/sentence.txt
```

On my computer, this took over 17 hours to run and the eventual file was 17 GB in size.
To compress the file, run `gzip sentence.txt`.
This reduced the size to 5.8GB and took 20 minutes.

