from __future__ import print_function

import corpus

import numpy
from keras.models import Model
from keras.layers import Input, Dense, Activation, Embedding, LSTM, TimeDistributed, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences



from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

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
		self.NB_DIMS = 256
		#pass

	def data_to_matrix(self, data) :
		
		X, Y = numpy.zeros((self.token_count,), dtype='int32'), numpy.zeros((self.token_count,), dtype='int32')
		idx = 0
		for i in range(len(data[0])) :
			for j in range(len(data[0][i])) :
				X[idx]=self.word_index[data[0][i][j]] if data[0][i][j] in self.word_index else self.word_index["__UNK__"]
				Y[idx]=self.class_index[data[1][i][j]] if data[1][i][j] in self.class_index else self.class_index["__UNK__"]
				idx += 1
		return X, Y
	def make_dataset(self, filename) :
		data = corpus.extract(corpus.load(filename))
		self.MAX_SENTENCE_LEN = max(map(len, data[0]))
		self.token_count = sum(map(len, data[0]))
		self.vocab = list({w for s in data[0] for w in s})
		self.vocab += ["__UNK__"]
		self.word_index = {w : i for i, w in enumerate(self.vocab)}
		self.index_word = dict(enumerate(self.vocab))
		self.VOC_SIZE = len(self.vocab)
		self.classes = list({t for s in data[1] for t in s})
		self.classes += ["__UNK__"]
		self.class_index = {c : i for i, c in enumerate(self.classes)}
		self.index_class = dict(enumerate(self.classes))
		self.CLASSES_SIZE = len(self.classes)
		return self.data_to_matrix(data)

	def train(self, filename, **kwargs) :
		X, Y = self.__make_dataset(filename)
		input_layer = Input(shape=(self.VOC_SIZE))
		embedding_layer = Embedding(self.VOC_SIZE, self.NB_DIMS)(input_layer)
		recurrent_layer1 = LSTM(self.NB_DIMS, dropout=.2, recurrent_dropout=.2, return_sequences=True)(input_layer)
		recurrent_layer2 = LSTM(self.NB_DIMS, dropout=.2, recurrent_dropout=.2)(recurrent_layer1)
		output_layer = Dense(128, activation='softmax')(recurrent_layer2)
		self.model = Model(input_layer, output_layer)
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		self.model.fit(X, Y, batch_size=64, epochs=20)

	def tag(self, wordlist) :
		data = numpy.zeros(self.MAX_SENTENCE_LEN, dtype='float32')
		print(data.shape)
		for i, word in enumerate(wordlist) :
			data[i] = self.word_index[word] if word in self.word_index else len(self.word_index)
		return [self.index_class[i] for i in self.model.predict(data)][:len(word_list)]
	def predict(self, wordlist) :
		return self.tag(wordlist)
	def test(self, data) :
		X, Y =  self.__data_to_matrix(data)
		return self.model.evaluate(X, Y, batch_size=64)

X, Y = corpus.extract(corpus.load("sequoia-corpus.np_conll.train"))
x_list = ["__START__"] + list({w for x in X for w in x}) + ["__UNK__"]
y_list = ["__START__"] + list({c for y in Y for c in y}) + ["__UNK__"]
x_codes = {x: idx for idx, x in enumerate(x_list)}
y_codes = {y: idx for idx, y in enumerate(y_list)}
def encode(X, Y) :
	Xcodes = []
	for x in X:
		Xcodes.append([x_codes[elt] if elt in x_codes else x_codes["__UNK__"] for elt in x])
	Ycodes = []
	for y in Y:
		ymat = numpy.zeros((len(y),len(y_codes)))
		for idx,elt in enumerate(y):
			ymat[idx,y_codes[elt]] = 1.0
		Ycodes.append(ymat)
	return Xcodes, Ycodes
Xcodes, Ycodes = encode(X, Y)
X_test, Y_test = corpus.extract(corpus.load("sequoia-corpus.np_conll.test"))
Xcodes_test, Ycodes_test =  encode(X_test, Y_test)


L = [len(y) for y in Ycodes]
mL = int(sum(L)/len(L))
print(mL) #longueur moyenne
Xcodes = pad_sequences(Xcodes,maxlen=mL)
Ycodes = pad_sequences(Ycodes,maxlen=mL)

Xcodes_test = pad_sequences(Xcodes_test,maxlen=mL)
Ycodes_test = pad_sequences(Ycodes_test,maxlen=mL)

x_size = len(x_codes)
y_size = len(y_codes)
embedding_size = 50
memory_size    = 20
model = Sequential()
ipt = Input(shape=(mL,))
e = Embedding(x_size,embedding_size)(ipt)
h = LSTM(memory_size,return_sequences=True)(e)
"""h = Bidirectional(LSTM(memory_size,return_sequences=True))(h)
h = Bidirectional(LSTM(memory_size,return_sequences=True))(h)
h = Bidirectional(LSTM(memory_size,return_sequences=True))(h)
h = Bidirectional(LSTM(memory_size,return_sequences=True))(h)"""
o = TimeDistributed(Dense(y_size, activation='softmax'))(h)
model = Model(ipt, o)
"""model.add(Embedding(x_size,embedding_size))
model.add(Bidirectional(LSTM(memory_size,return_sequences=True))) #bi-LSTM"""
#model.add(LSTM(memory_size,return_sequences=True))               #simple LSTM
"""model.add(TimeDistributed(Dense(y_size, activation='softmax')))""" 
model.summary()

sgd = Adam(lr=0.001)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(Xcodes,Ycodes,epochs=20,batch_size=64)
print(model.evaluate(Xcodes_test, Ycodes_test, batch_size=64))



"""
if __name__ == "__main__" :
	filename="sequoia-corpus.np_conll.train"
	x_train, y_train = NNTagger("foo").make_dataset(filename)
	filename="sequoia-corpus.np_conll.test"
	x_test, y_test = NNTagger("foo").make_dataset(filename)
	tagger.train(filename)
	max_features = 20000
	maxlen = 80  # cut texts after this number of words (among top max_features most common words)
	batch_size = 32

	print('Loading data...')
	#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')
	print(x_train[0])
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
	print(x_train[0])
	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)
	print(x_train[0])
	exit(0)

	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, 128))
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))

	# try using different optimizers and different optimizer configs
	model.compile(loss='binary_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'])

	print('Train...')
	model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=15,
		  validation_data=(x_test, y_test))
	score, acc = model.evaluate(x_test, y_test,
		                    batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)
	print(tagger.test("sequoia-corpus.np_conll.test"))
"""
