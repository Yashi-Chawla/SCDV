import sys
sys.path.append('scdv/')

import logging
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from baseline import BaselineEmbedding

parser = argparse.ArgumentParser(description='Train Baseline Word2Vec model')
parser.add_argument('--init_vector_type', type=str, default="word2vec_sg", help='Initial vector type', choices=["word2vec_sg", "word2vec_cbow", "fasttext_sg", "fasttext_cbow"])
parser.add_argument('--vector_size', type=int, default=100, help='Vector size')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--ngram_training', action='store_true', help='Use ngrams for training')
parser.add_argument('--data', type=str, default="20newsgroups", help='Path to dataset')
parser.add_argument('--save_model', type=str, help='Path to save model')
parser.add_argument('--log', type=str, help='Path to logging file')
args = parser.parse_args()

if args.log is None:
    Path.cwd().joinpath('log').mkdir(exist_ok=True)
    args.log = 'log/baseline_train_{}_{}.log'.format(args.init_vector_type, args.vector_size)
    
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename=args.log,
    filemode='w'
)
logging.info(f'Training baseline model with {args.init_vector_type} and {args.vector_size} dimensions')

corpus = list()
logging.info('Loading dataset')
if args.data == '20newsgroups':
    logging.info('Loading 20newsgroups dataset')
    newsgroup = fetch_20newsgroups(subset='all')
    for article in tqdm(newsgroup['data']):
        word_list = word_tokenize(article)
        corpus.append(word_list)
else:
    logging.info(f'Loading data from {args.data}')
    if args.data == "bbc":
        args.data = "data/bbc/all"
    elif args.data == "allthenews":
        args.data = "data/allthenews/"
    else:
        pass

    p = Path(args.data)
    files = list(p.glob("**/*.txt"))
    for file in tqdm(files):
        try:
            with open(file, "r", encoding='utf8') as f:
                text = f.read().strip()
            word_list = word_tokenize(text)
            corpus.append(word_list)
        except:
            pass

model = BaselineEmbedding(
    init_vector_type=args.init_vector_type,
    vector_size=args.vector_size,
    use_ngram_training=args.ngram_training,
    epochs=args.epochs
)

logging.info(f'Fitting model to {len(corpus)} documents')
model.fit(corpus)

if args.save_model is None:
    Path.cwd().joinpath('saved_models').mkdir(exist_ok=True)
    args.save_model = 'saved_models/baseline_{}_{}_{}.pkl'.format(Path(args.data).stem, args.init_vector_type, args.vector_size)

logging.info(f'Saving model to {args.save_model}')
model.save(args.save_model)