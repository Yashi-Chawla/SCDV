import sys
sys.path.append('scdv/')

import logging
import argparse
from scdv import SCDV
from pathlib import Path
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups

parser = argparse.ArgumentParser(description='Train SCDV model')
parser.add_argument('--init_vector_type', type=str, default="word2vec_sg", help='Initial vector type', choices=["word2vec_sg", "word2vec_cbow", "fasttext_sg", "fasttext_cbow"])
parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters')
parser.add_argument('--vector_size', type=int, default=100, help='Vector size')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--ngram_training', action='store_true', help='Use ngrams for training')
parser.add_argument('--data', type=str, default="20newsgroups", help='Path to dataset')
parser.add_argument('--save_model', type=str, help='Path to save model')
parser.add_argument('--log', type=str, help='Path to logging file')
args = parser.parse_args()

if args.log is None:
    Path.cwd().joinpath('log').mkdir(exist_ok=True)
    args.log = 'log/scdv_train_{}_{}_{}.log'.format(args.init_vector_type, args.num_clusters, args.vector_size)
    
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename=args.log,
    filemode='w'
)
logging.info(f'Training SCDV model with {args.init_vector_type}, {args.num_clusters} clusters, {args.vector_size} dimensions')

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
    with open(args.data, 'r') as f:
        for line in tqdm(f):
            word_list = word_tokenize(line)
            corpus.append(word_list)

model = SCDV(
    init_vector_type=args.init_vector_type,
    num_clusters=args.num_clusters,
    vector_size=args.vector_size,
    use_ngram_training=args.ngram_training,
    epochs=args.epochs
)

logging.info(f'Fitting model to {len(corpus)} documents')
model.fit(corpus)

if args.save_model is None:
    Path.cwd().joinpath('saved_models').mkdir(exist_ok=True)
    args.save_model = 'saved_models/{}_{}_{}_{}.pkl'.format(Path(args.data).stem, args.init_vector_type, args.num_clusters, args.vector_size)

logging.info(f'Saving model to {args.save_model}')
model.save(args.save_model)