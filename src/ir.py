import sys
sys.path.append('scdv/')

import random
import logging
import argparse
import numpy as np
from scdv import SCDV
from nltk.lm import MLE
from pathlib import Path
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from nltk.lm.preprocessing import padded_everygram_pipeline
import random
import ir_datasets

dataset = ir_datasets.load("trec-fair-2021/train")

random.seed(0)

parser = argparse.ArgumentParser(description='Run Information Retrieval')
parser.add_argument('--model', type=str, help='Path to trained SCDV model')
parser.add_argument('--lm_ngram', type=int, default=3, help='N-gram for language model')
parser.add_argument('--query', type=str, help='Path to queries')#, default='data/query.txt')
parser.add_argument('--documents', type=str, help='Path to documents', default='20newsgroups')
parser.add_argument('--lambda_', type=float, default=0.5, help='Lambda parameter for SCDV IR')
parser.add_argument('--p', type=float, default=0.04, help='Percentage Sparsity threshold SCDV IR')
parser.add_argument('--top', type=int, default=5, help='Top K documents to return')
parser.add_argument('--output', type=str, help='Path to store output files', default='results')
parser.add_argument('--log', type=str, help='Path to logging file')
args = parser.parse_args()

if args.log is None:
    Path.cwd().joinpath('log').mkdir(exist_ok=True)
    args.log = Path.cwd().joinpath('log').joinpath('ir.log')

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename=args.log,
    filemode='w'
)
logging.info(f'Running IR with {args.model}, {args.query}, {args.documents}')

def get_qd_score(model, query_idx, document_idx, query_unigram_probabilities, document_unigram_probabilities):
    _sum = 0
    for word in query_words[query_idx]:
        _sum += query_unigram_probabilities[query_idx][word] * get_probability_wd(word, document_idx, document_unigram_probabilities, model)
    return _sum

def get_scores(model, query_idx, document_idx, query_unigram_probabilities, document_unigram_probabilities):
    query_vector = query_vectors[query_idx]
    document_vector = document_vectors[document_idx]
    similarity = model.similarity(query_vector, document_vector)
    score_qd = get_qd_score(model, query_idx, document_idx, query_unigram_probabilities, document_unigram_probabilities)
    score_pv = (1 - args.lambda_) * score_qd + args.lambda_ * similarity
    return score_pv, score_qd

def get_probability_pv(word, document_idx, model):
    word_vector = model.get_word_vector(word)
    document_vector = document_vectors[document_idx]
    similarity = model.similarity(word_vector, document_vector)
    similarity_exponent = np.exp(similarity)
    _sum = 0
    for word in document_words[document_idx]: # model.vocabulary: # is this model vocabulary?
        _sum += np.exp(model.similarity(document_vector, model.get_word_vector(word)))
    probability_pv = similarity_exponent / _sum
    return probability_pv

def get_probability_wd(word, document_idx, document_unigram_probabilities, model):
    probability_lm = document_unigram_probabilities[document_idx].get(word, 0)
    return (1 - args.lambda_) * probability_lm + args.lambda_ * get_probability_pv(word, document_idx, model)

def make_sparse_document_vectors(document_vectors):
    ndim = document_vectors.shape[1]
    min_ndim = list()
    max_mdin = list()
    for i in range(ndim):
        min_ndim.append(np.min(document_vectors[:, i]))
        max_mdin.append(np.max(document_vectors[:, i]))
    a_min = np.mean(min_ndim)
    a_max = np.mean(max_mdin)
    t = (np.abs(a_min) + np.abs(a_max)) / 2
    pt = args.p * t
    document_vectors[np.abs(document_vectors) < pt] = 0
    return document_vectors

logging.info(f"Loading model from {args.model}")
model = SCDV.load(args.model)

logging.info(f"Loading queries from {args.query}")
with open(args.query, 'r') as f:
    queries = [line.strip() for line in f]
    queries = random.sample(queries, 100)
query_words = [word_tokenize(query) for query in queries]
query_unigram_probabilities = list()
for query in tqdm(query_words):
    unigram_probabilities = dict()
    train, vocab = padded_everygram_pipeline(1, [query])
    lm = MLE(1)
    lm.fit(train, vocab)
    for word in query:
        unigram_probabilities[word] = lm.score(word)
    query_unigram_probabilities.append(unigram_probabilities)

logging.info(f"Loading documents from TREC")
# if args.documents == '20newsgroups':
#     logging.info('Loading 20newsgroups dataset')
#     newsgroup = fetch_20newsgroups(subset='all')
#     documents = [article for article in tqdm(newsgroup['data'])][:5000]
# else:
#     logging.info(f'Loading data from {args.documents}')
#     documents = [Path(args.documents).joinpath(f).read_text() for f in Path(args.documents).iterdir()]
# document_words = [word_tokenize(document) for document in documents]

documents = [doc for doc in dataset.iter()]
print(documents)

# logging.info("Vectorizing queries")
# query_vectors = [model.get_document_vector(word_tokenize(query)) for query in tqdm(queries)]
# query_vectors = np.asarray(query_vectors)

# logging.info("Vectorizing documents")
# document_vectors = [model.get_document_vector(word_tokenize(document)) for document in tqdm(documents)]
# document_vectors = np.asarray(document_vectors)
# document_vectors = make_sparse_document_vectors(document_vectors)

# logging.info("Fitting Language Models and computing Uni-gram probabilities")
# document_lm = list()
# document_unigram_probabilities = list()
# for document in tqdm(document_words):
#     unigram_probabilities = dict()
#     train, vocab = padded_everygram_pipeline(args.lm_ngram, [document])
#     lm = MLE(args.lm_ngram)
#     lm.fit(train, vocab)
#     document_lm.append(lm)
#     for word in document:
#         unigram_probabilities[word] = lm.score(word)
#     document_unigram_probabilities.append(unigram_probabilities)

# logging.info("Computing scores")
# total_queries = len(queries)
# total_documents = len(documents)
# Path.cwd().joinpath(args.output).mkdir(exist_ok=True)
# with open(f"{args.output}/PV.txt", 'w') as f_pv, open(f"{args.output}/QD.txt", 'w') as f_qd, open(f"{args.output}/LM.txt", 'w') as f_lm:
#     for query_idx in tqdm(range(total_queries)):
        
#         if (query_idx + 1) % 20 == 0:
#             logging.info(f'Processing query {query_idx}/{total_queries}')
        
#         scores = list()
#         f_pv.write(queries[query_idx] + '\n\n')
#         f_qd.write(queries[query_idx] + '\n\n')
#         f_lm.write(queries[query_idx] + '\n\n')

#         for document_idx in tqdm(range(total_documents)):

#             if (document_idx + 1) % 1000 == 0:
#                 logging.info(f'{document_idx + 1}/{total_documents} documents processed')

#             score_lm = document_lm[document_idx].score(query_words[query_idx][-1], query_words[query_idx][:-1])
#             score_pv, score_qd = get_scores(model, query_idx, document_idx, query_unigram_probabilities, document_unigram_probabilities)
#             scores.append((score_pv, score_qd, score_lm, document_idx))

#         for i in range(0, 3):
#             scores.sort(key=lambda x: x[i], reverse=True)
#             if i == 0:
#                 f = f_pv
#             elif i == 1:
#                 f = f_qd
#             else:
#                 f = f_lm
#             for score in scores[:args.top]:
#                 f.write(f'Score: {score[i]}\n\nDocument: {documents[score[3]]}\n\n')
#             f.write('--------------------------------------------------------------------------------\n\n')
#             f.flush()
        
#         f_pv.flush()
#         f_qd.flush()
#         f_lm.flush()