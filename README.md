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
One dataset used is from the Hewlett Foundation's Automated Essay Scoring
competition on [Kaggle](https://www.kaggle.com/c/asap-aes/overview).

To download the dataset, make a new folder called `data` and run
this command in the folder:
```bash
kaggle competitions download -c asap-aes
```

- training_set_rel3.tsv: training set 
(used to fit the model to the data, e.g. adjust weights for a neural network)
- valid_set.tsv: validation set 
(used to determine hyperparameters of the model, e.g. activation function or depth)
- test_set.tsv: test set 
(used to test final model)

Another dataset used is the [UMBC Corpus](https://ebiquity.umbc.edu/resource/html/id/351).
It comes as a .tar.gz file, which can be extracted
with `gzip -d` and `tar -xf`, in that order,
or just `tar -xzf` (where the `-z` option indicates gzip).

Make a new folder called `text` in `data` and
run the data preprocessing script from word2mat:
```bash
pipenv run python word2mat/data/extract_umbc.py data/webbase_all/ data/text/sentence.txt
```

On my computer, this took over 17 hours to run and the eventual file was 17 GB in size.
To compress the file, run `gzip sentence.txt`.
This reduced the size to 5.8GB and took 20 minutes.

Either copy a sentence to word2mat with the command (the number is how many lines to copy):
```bash
python data.py -g 100
```

Or just specify the number of lines in the running command later.

Finally, to get the dataset for SentEval, go into the `/word2mat/SentEval/data/downstream`
folder and run the command:
```bash
bash get_transfer_data.bash
```

### Running

First, create folders called `test_model` and `output` in the `word2mat` folder.

Then, run this command in the `word2mat` folder to train the model 
and run SentEval on the trained model:
```bash
python train_cbow.py --outputdir=test_model --temp_path test_temp --dataset_path=../data/text --output_file output/output.csv --num_docs 100 --num_workers 2 --w2m_type hybrid --batch_size=1024 --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --stop_criterion train_loss --initialization identity
```

To run SentEval on an already trained model, 
run the same command with 0 epochs and the `--load_model` parameter:
```
python train_cbow.py --load_model test --outputdir=test_model --temp_path test_temp --dataset_path=../data/text --output_file output/output.csv --num_docs 100 --num_workers 2 --w2m_type hybrid --batch_size=1024 --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --stop_criterion train_loss --initialization identity
```

### Results

Run this command to get a tabulated version of SentEval results: 
```
python data.py -om
```

Outputs were generating by running 100 epochs on 10^6 sentences.
First two rows are with `--downstream_eval test`,
last two rows are with `--downstream_eval full`.

Probing: 
|                |   Depth |   BigramShift |   SubjNumber |   Tense |   CoordinationInversion |   Length |   ObjNumber |   TopConstituents |   OddManOut |   WordContent |
|:---------------|--------:|--------------:|-------------:|--------:|------------------------:|---------:|------------:|------------------:|------------:|--------------:|
| nonlinear      |   28.32 |         52.2  |        76.43 |   75.2  |                   54.47 |    89.29 |       74.84 |             62.26 |       49.62 |         90.08 |
| cmow           |   29.48 |         52.23 |        76.17 |   74.17 |                   54.74 |    89.78 |       74.97 |             63.98 |       50.51 |         89.53 |
| nonlinear_full |   32.54 |         51.53 |        69.71 |   62.97 |                   55.7  |    88.17 |       65.82 |             64.36 |       51.31 |         20.58 |
| cmow_full      |   32.84 |         51.09 |        69.87 |   63.42 |                   56.34 |    89.23 |       66.75 |             65.01 |       50.76 |         20.91 | 

Supervised downstream: 
|                |   SNLI |   SUBJ |    CR |    MR |   MPQA |   TREC |   SICKEntailment |   SST2 |   SST5 |   MRPC |   STSBenchmark |   SICKRelatedness |
|:---------------|-------:|-------:|------:|------:|-------:|-------:|-----------------:|-------:|-------:|-------:|---------------:|------------------:|
| nonlinear      |  62.87 |  86.45 | 76.56 | 68.98 |  83.49 |   79.4 |            76.01 |  72.76 |  38.1  |  70.9  |       0.565225 |          0.703384 |
| cmow           |  62.36 |  86.97 | 74.99 | 68.52 |  83.93 |   81   |            74.47 |  73.59 |  38.14 |  69.97 |       0.542191 |          0.698892 |
| nonlinear_full |  54.62 |  79.94 | 71.31 | 64.29 |  74.5  |   67.6 |            71.28 |  68.09 |  30.18 |  67.77 |       0.308914 |          0.541575 |
| cmow_full      |  54.73 |  80.52 | 71.84 | 64.83 |  75.17 |   66.8 |            71.87 |  66.72 |  32.99 |  68.7  |       0.324146 |          0.532514 | 

Unsupervised downstream: 
|                |    STS12 |    STS13 |    STS14 |    STS15 |    STS16 |
|:---------------|---------:|---------:|---------:|---------:|---------:|
| nonlinear      | 0.407626 | 0.389382 | 0.509538 | 0.516131 | 0.503199 |
| cmow           | 0.425122 | 0.450946 | 0.5215   | 0.532434 | 0.520201 |
| nonlinear_full | 0.22191  | 0.11622  | 0.286445 | 0.324294 | 0.363277 |
| cmow_full      | 0.21983  | 0.112364 | 0.284414 | 0.333384 | 0.381094 | 

