import pickle
import logging
import numpy as np
import multiprocessing
from nltk import ngrams
from tqdm.auto import tqdm
from gensim.models import Word2Vec
from gensim.models import FastText

class BaselineEmbedding:
    def __init__(self, init_vector_type="word2vec_sg", vector_size=100, use_ngram_training=False, epochs=5):
        self.init_vector_type = init_vector_type
        self.vector_size = vector_size
        self.epochs = epochs
        self.use_ngram_training = use_ngram_training
        self.word_vectors = dict()

    def init_word_vectors(self, corpus):
        if "word2vec" in self.init_vector_type:
            vector_model = Word2Vec
        else:
            vector_model = FastText

        if "sg" in self.init_vector_type:
            use_sg = 1
        else:
            use_sg = 0

        self.init_vector_model = vector_model(
            corpus,
            vector_size=self.vector_size,
            window=5,
            min_count=1,
            sg=use_sg,
            negative=10,
            sample=1e-3,
            hs=0,
            epochs=self.epochs,
            seed=0,
            workers=multiprocessing.cpu_count()-1
        )
        self.init_vector_model.build_vocab(corpus)
        self.vocabulary = list(self.init_vector_model.wv.index_to_key)
        self.init_vector_model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    def get_word_vector_map(self):
        for word in self.vocabulary:
            self.word_vectors[word] = np.asarray(self.init_vector_model.wv[word])

    def ngram_training(self, corpus):
        logging.info("Obtaining ngrams...")
        new_corpus = list()
        sentences = [[f'<{word}>' for word in sentence] for sentence in corpus]
        sentences = list(map(' '.join, sentences))
        for n in tqdm(range(3, 7)):
            for sentence in tqdm(sentences):
                sentence_ngrams = list(map(''.join, list(ngrams(sentence, n))))
                new_corpus.append(sentence_ngrams)

        self.init_vector_model.build_vocab(new_corpus, update=True)
        self.vocabulary = list(self.init_vector_model.wv.index_to_key)
        self.init_vector_model.train(new_corpus, total_examples=len(new_corpus), epochs=int(self.epochs/2))
        return corpus + new_corpus

    def get_word_vector(self, word):
        if word in self.word_vectors:
            return self.word_vectors[word]
        elif f'<{word}>' in self.word_vectors:
            return self.word_vectors[f'<{word}>']
        else:
            word = f'<{word}>'
            vector = np.zeros(self.vector_size)
            for n in range(3, 7):
                word_ngrams = list(map(''.join, list(ngrams(word, n))))
                for word_ngram in word_ngrams:
                    if word_ngram in self.word_vectors:
                        vector += self.word_vectors[word_ngram]
            vector = vector / np.linalg.norm(vector)
            return vector

    def get_document_vector(self, document):
        document_vector = list()
        for word in document:
            document_vector.append(self.get_word_vector(word))
        document_vector = np.sum(document_vector, axis=0)
        document_vector = document_vector / np.linalg.norm(document_vector)
        return document_vector

    def similarity(self, vector_1, vector_2):
        return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

    def word_similarity(self, word_1, word_2):
        vector_1 = self.get_word_vector(word_1)
        vector_2 = self.get_word_vector(word_2)
        return self.similarity(vector_1, vector_2)

    def document_similarity(self, document_1, document_2):
        vector_1 = self.get_document_vector(document_1)
        vector_2 = self.get_document_vector(document_2)
        return self.similarity(vector_1, vector_2)

    def fit(self, corpus):
        logging.info("Initialising word vectors")
        self.init_word_vectors(corpus)

        if self.use_ngram_training:
            logging.info("Training word vectors with ngrams")
            corpus = self.ngram_training(corpus)

        logging.info("Obtaining word vectors")
        self.get_word_vector_map()

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        return pickle.load(open(filename, "rb"))