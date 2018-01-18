from __future__ import print_function

import corpus

import numpy

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.preprocessing.sequence import pad_sequences


class NNTagger(object):
    def __init__(self, embedding_size=50, memory_size=20):
        self.embedding_size = embedding_size
        self.memory_size = memory_size

    def _encode(self, X, Y):
        Xcodes = [[self.x_codes[elt] if elt in self.x_codes else self.x_codes["__UNK__"]
                   for elt in x] for x in X]
        Ycodes = []
        for y in Y:
            ymat = numpy.zeros((len(y), len(self.y_codes)))
            for idx, elt in enumerate(y):
                ymat[idx, self.y_codes[elt]] = 1.
            Ycodes.append(ymat)
        return Xcodes, Ycodes

    def train(self, filename, epochs=5, batch_size=64, verbose=0, optimizer=RMSprop(lr=.1), **kwargs):
        X, Y = corpus.extract(corpus.load(filename))
        self.x_list = ["__START__"] + \
            list({w for x in X for w in x}) + ["__UNK__"]
        self.y_list = ["__START__"] + \
            list({c for y in Y for c in y}) + ["__UNK__"]
        self.x_codes = {x: idx for idx, x in enumerate(self.x_list)}
        self.y_codes = {y: idx for idx, y in enumerate(self.y_list)}
        self.reverse_x_codes = {i: x for i, x in enumerate(self.x_list)}
        self.reverse_y_codes = {i: y for i, y in enumerate(self.y_list)}
        Xcodes, Ycodes = self._encode(X, Y)
        L = list(map(len, Ycodes))
        self.mL = max(L)
        Xcodes = pad_sequences(Xcodes, maxlen=self.mL)
        Ycodes = pad_sequences(Ycodes, maxlen=self.mL)
        self.x_size = len(self.x_codes)
        self.y_size = len(self.y_codes)
        ipt = Input(shape=(self.mL,))
        e = Embedding(self.x_size, self.embedding_size, trainable=True,
                      mask_zero=True, name="embedding_layer")(ipt)
        h = Bidirectional(LSTM(self.memory_size, return_sequences=True))(e)
        o = TimeDistributed(
            Dense(self.y_size, bias_regularizer=l1_l2(0.), activation='softmax'))(h)
        self.model = Model(ipt, o)
        if verbose:
            self.model.summary()
        self.model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(Xcodes, Ycodes, epochs=epochs,
                       verbose=verbose, batch_size=batch_size)
        return self

    def save(self, savename="tagger.model.h5"):
        self.model.save(savename)
        return self

    def predict(self, sentences):
        X = [[(self.x_codes[word] if word in self.x_codes else self.x_codes["__UNK__"])
              for word in sentence] for sentence in sentences]
        predictions = self.model.predict(pad_sequences(X, maxlen=self.mL))
        preds = []
        for i in range(len(predictions)):
            pred = predictions[i, -len(sentences[i]):]
            preds.append([self.reverse_y_codes[numpy.argmax(w)] for w in pred])
        indices = [list(range(1, len(s) + 1)) for s in sentences]
        return indices, sentences, preds

    def test(self, filename):
        X_test, Y_test = corpus.extract(corpus.load(filename))
        Xcodes_test, Ycodes_test = self._encode(X_test, Y_test)
        Xcodes_test = pad_sequences(Xcodes_test, maxlen=self.mL)
        Ycodes_test = pad_sequences(Ycodes_test, maxlen=self.mL)
        return self.model.evaluate(Xcodes_test, Ycodes_test, batch_size=64)


if __name__ == "__main__":
    for embedding_size in range(20, 100, 10):
        for memory_size in range(20, 100, 10):
            tagger = NNTagger()
            tagger.train("sequoia-corpus.np_conll.train", verbose=1)
            print(tagger.test("sequoia-corpus.np_conll.test"))
    # print(tagger.predict(corpus.extract(corpus.load("sequoia-corpus.np_conll.dev"))[0]))
