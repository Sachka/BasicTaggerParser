import corpus

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

"""
def read_conll_sentence(istream):
    x_seq = []
    y_seq = []
    line = istream.readline()
    while line and not line.isspace():s
        fields = line.split()
        x_seq.append(fields[1])
        y_seq.append(fields[3])
        line = istream.readline()
    return (x_seq, y_seq)


def read_conll_corpus(filename):
    X = []
    Y = []
    istream = open(filename)
    (x, y) = read_conll_sentence(istream)
    while x and y:
        X.append(x)
        Y.append(y)
        (x, y) = read_conll_sentence(istream)
    istream.close()
    return X, Y


X, Y = read_conll_corpus('sequoia-corpus.np_conll.test')
print(X[:3], Y[:3])

x_set = set([])
y_set = set([])
init_token = "__START__"
for x in X:
    x_set.update(x)
for y in Y:
    y_set.update(y)
rev_x_codes = [init_token]
rev_x_codes.extend(list(x_set))
rev_y_codes = list(y_set)
x_codes = dict((x, idx) for idx, x in enumerate(rev_x_codes))
y_codes = dict((y, idx) for idx, y in enumerate(rev_y_codes))
print(y_codes)"""

class NNTagger(object) :
	def __init__(self) :
		pass
	def train(self, corpus, **kwargs) :
		pass
	def tag(self, wordlist) :
		pass
	def predict(self, wordlist) :
		pass

