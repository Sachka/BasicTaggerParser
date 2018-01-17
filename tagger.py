from __future__ import print_function

import corpus

import numpy

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.preprocessing.sequence import pad_sequences

class NNTagger(object) :
	def __init__(self, embedding_size=50, memory_size=20) :
		self.embedding_size = embedding_size
		self.memory_size = memory_size

	def _encode(self, X, Y) :
		Xcodes = []
		for x in X:
			Xcodes.append([self.x_codes[elt] if elt in self.x_codes else self.x_codes["__UNK__"] for elt in x])
		Ycodes = []
		for y in Y:
			ymat = numpy.zeros((len(y),len(self.y_codes)))
			for idx,elt in enumerate(y):
				ymat[idx, self.y_codes[elt]] = 1.
			Ycodes.append(ymat)
		return Xcodes, Ycodes

	def train(self, filename, epochs=20, batch_size=64, verbose=0, optimizer=RMSprop(lr=.001), **kwargs) :
		X, Y = corpus.extract(corpus.load(filename))
		self.x_list = list({w for x in X for w in x}) + ["__UNK__"]
		self.y_list = list({c for y in Y for c in y}) + ["__UNK__"]
		self.x_codes = {x: idx for idx, x in enumerate(self.x_list)}
		self.y_codes = {y: idx for idx, y in enumerate(self.y_list)}
		Xcodes, Ycodes = self._encode(X, Y)
		L=list(map(len, Ycodes))
		self.mL = int(sum(L)/len(L))
		Xcodes = pad_sequences(Xcodes, maxlen=self.mL)
		Ycodes = pad_sequences(Ycodes, maxlen=self.mL)
		self.x_size = len(self.x_codes)
		self.y_size = len(self.y_codes)
		ipt = Input(shape=(self.mL,))
		e = Embedding(self.x_size, self.embedding_size, mask_zero=True)(ipt)
		h = LSTM(self.memory_size, return_sequences=True)(e)
		o = TimeDistributed(Dense(self.y_size, bias_regularizer=l1_l2(0.), activation='softmax'))(h)
		self.model = Model(ipt, o)
		if verbose : self.model.summary()
		self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		self.model.fit(Xcodes, Ycodes, epochs=epochs, batch_size=batch_size)

	def tag(self, wordlist) :
		data = numpy.zeros(self.MAX_SENTENCE_LEN, dtype='float32')
		print(data.shape)
		for i, word in enumerate(wordlist) :
			data[i] = self.word_index[word] if word in self.word_index else len(self.word_index)
		return [self.index_class[i] for i in self.model.predict(data)][:len(word_list)]

	def predict(self, wordlist) :
		return self.tag(wordlist)

	def test(self, filename) :
		X_test, Y_test = corpus.extract(corpus.load(filename))
		Xcodes_test, Ycodes_test =  self._encode(X_test, Y_test)
		Xcodes_test = pad_sequences(Xcodes_test,maxlen=self.mL)
		Ycodes_test = pad_sequences(Ycodes_test,maxlen=self.mL)
		return self.model.evaluate(Xcodes_test, Ycodes_test, batch_size=64)

if __name__ == "__main__" :
	tagger = NNTagger()
	tagger.train("sequoia-corpus.np_conll.train", verbose=1)
	print(tagger.test("sequoia-corpus.np_conll.test"))
