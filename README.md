# ASSIP 2020

## Setup
This project uses [word2mat](https://github.com/florianmai/word2mat),
specifically our [fork](https://github.com/stephen-huan/word2mat).

### Software
First, clone the repository and update the submodules:
```
git clone --recursive https://github.com/stephen-huan/assip-2020
```

To install the dependencies, use [pipenv](https://pipenv.pypa.io/en/latest/): 
```bash
pipenv install
pipenv run pip install -r word2mat/requirements.txt
pipenv run python -c "import nltk; nltk.download('punkt')"
```

### Data
Datasets used are from the Hewlett Foundation's Automated Essay Scoring
competition on [Kaggle](https://www.kaggle.com/c/asap-aes/overview).

To download the datasets, make a new folder called `data` and run
this command in the folder:
```bash
kaggle competitions download -c asap-aes
```

- training_set_rel3.tsv: training set 
(used to fit the model to the data, e.g. adjusted weights for a neural network)
- valid_set.tsv: validation set 
(used to determine hyperparameters of the model, e.g. activation function or depth)
- test_set.tsv: test set 
(used to test final model)

Another dataset used is the [UMBC Corpus](https://ebiquity.umbc.edu/resource/html/id/351).
It comes as a .tar.gz file, which can be extracted
with `gzip -d` and `tar -xf`, in that order,
or just `tar -xzf` (where the `-z` option indicates gzip).

Run the data preprocessing script from word2mat.
```bash
pipenv run python word2mat/data/extract_umbc.py data/webbase_all/ data/sentence.txt
```

On my computer, this took over 17 hours to run and the eventual file was 17 GB in size.
To compress the file, run `gzip sentence.txt`.
This reduced the size to 5.8GB and took 20 minutes.

Copy a sentence to word2mat with the command (the number is how many lines to copy):
```bash
python data.py -g 100
```

Finally, to get the dataset for SentEval, go into the `/word2mat/SentEval/data/downstream`
folder and run the command:
```bash
bash get_transfer_data.bash
```

### Running

First, create a folder called `test_model` in the `word2mat` folder.

Then, run this command in the `word2mat` folder to train the model:
```bash
python train_cbow.py --outputdir=test_model --temp_path test_temp --dataset_path=data --output_file output.csv --num_docs 100 --num_workers 2 --w2m_type hybrid --batch_size=1024 --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --stop_criterion train_loss --initialization identity
```
