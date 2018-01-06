import corpus

import numpy
from keras.models import Model
from keras.layers import Input, Dense, Activation, Embedding, LSTM, TimeDistributed, Bidirectional
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
	def __init__(self, nb_dims) :
		self.NB_DIMS = 42
		#pass

	def __make_dataset(self, filename) :
		data = corpus.extract(corpus.load(filename))
		self.MAX_SENTENCE_LEN = max(map(len, data[0]))
		self.vocab = list({w for s in data[0] for w in s})
		self.word_index = {w : i for i, w in enumerate(self.vocab)}
		self.index_word = dict(enumerate(self.vocab))
		self.VOC_SIZE = len(self.vocab)
		self.classes = list({t for s in data[1] for t in s})
		self.class_index = {c : i for i, c in enumerate(self.classes)}
		self.index_class = dict(enumerate(self.classes))
		self.CLASSES_SIZE = len(self.classes)
		X, Y = numpy.zeros((len(data[0]), self.MAX_SENTENCE_LEN), dtype='float32'), numpy.zeros((len(data[1]), self.MAX_SENTENCE_LEN), dtype='float32')
		for i in range(len(data[0])) :
			for j in range(len(data[0][i])) :
				X[i,j] = self.word_index[data[0][i][j]]
				Y[i,j] = self.class_index[data[1][i][j]]
		return X, Y

	def train(self, filename, **kwargs) :
		X, Y = self.__make_dataset(filename)
		input_layer = Input(shape=(self.MAX_SENTENCE_LEN,))
		embedding_layer = Embedding(input_dim=self.VOC_SIZE, input_length=self.MAX_SENTENCE_LEN, output_dim=self.NB_DIMS)(input_layer)
		recurrent_layer = LSTM(self.NB_DIMS)(embedding_layer)
		output_layer = Dense(self.MAX_SENTENCE_LEN, activation='softmax')(recurrent_layer)
		self.model = Model(input_layer, output_layer)
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
		self.model.fit(X, Y, batch_size=64, epochs=100, validation_split=0.2)

	def tag(self, wordlist) :
		data = numpy.zeros((MAX_SENTENCE_LEN,), dtype='float32')
		for i, word in enumerate(wordlist) :
			data[i] = self.word_index[word] 
		return [self.index_class[i] for i in self.model.predict(data)][:len(word_list)]
	def predict(self, wordlist) :
		return self.tag(wordlist)
	def test(self, corpus) :
		pass

if __name__ == "__main__" :
	filename="sequoia-corpus.np_conll.dev"
	NNTagger("foo").train(filename)

