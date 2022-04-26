import random
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from rake_nltk import Rake
from sklearn.datasets import fetch_20newsgroups

parser = argparse.ArgumentParser(description='Generate keywords to use as queries')
parser.add_argument('--documents', type=str, default="20newsgroups", help='Path to folder with documents')
parser.add_argument('--output', type=str, help='Path to output file', default='data/20newsgroupqueries.txt')
parser.add_argument('--num_words', type=str, default='random', help='Number of words to generate for each query')
parser.add_argument('--multiple', action='store_true', help='Whether to generate multiple queries per document or one query per document')
args = parser.parse_args()

r = Rake()
keywords = list()

if args.documents == '20newsgroups':
    newsgroup = fetch_20newsgroups(subset='all')
    documents = [article for article in tqdm(newsgroup['data'])]
else:
    documents = [Path(args.documents).joinpath(f).read_text() for f in Path(args.documents).iterdir()]

with open(args.output, 'w', encoding='utf8') as f:
    for document in tqdm(documents):
        r.extract_keywords_from_text(document)
        keywords = r.get_ranked_phrases()
        if not args.multiple:
            if args.num_words == 'random':
                num_words = random.randint(2, 4)
            else:
                num_words = int(args.num_words)
            keywords = keywords[:random.randint(1, 3)]
            keywords = ' '.join(keywords).split()[:num_words]
            f.write(' '.join(keywords) + '\n')
        else:
            keywords = keywords[:random.randint(1, 3)]
            for k in keywords:
                f.write(k + '\n')