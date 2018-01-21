# from __future__ import print_function

import corpus

import numpy
import pickle

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Embedding, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

WE = "vectors50"


class NNTagger(object):
    def __init__(self, embedding_size=50, memory_size=20, use_ext_embeddings=True, external_embeddings=WE):
        self.embedding_size = embedding_size
        self.embeddings_index = None
        if use_ext_embeddings:
            self.embeddings_index = corpus.read_embeddings(external_embeddings)
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

    def train(self, filename, validation, epochs=20, batch_size=64, verbose=0, optimizer=RMSprop(lr=.01), use_ext_embeddings=True, ** kwargs):
        X, Y = corpus.extract(corpus.load(filename))
        D_X, D_Y = corpus.extract(corpus.load(validation))
        self.x_list = ["__START__"] + \
            list({w for x in X for w in x}) + ["__UNK__"]
        self.y_list = ["__START__"] + \
            list({c for y in Y for c in y}) + ["__UNK__"]
        self.x_codes = {x: idx for idx, x in enumerate(self.x_list)}
        self.y_codes = {y: idx for idx, y in enumerate(self.y_list)}
        self.reverse_x_codes = {i: x for i, x in enumerate(self.x_list)}
        self.reverse_y_codes = {i: y for i, y in enumerate(self.y_list)}

        Xcodes, Ycodes = self._encode(X, Y)
        D_Xcodes, D_Ycodes = self._encode(D_X, D_Y)
        L = list(map(len, Ycodes))
        self.mL = max(L)
        D_Xcodes = pad_sequences(D_Xcodes, maxlen=self.mL)
        D_Ycodes = pad_sequences(D_Ycodes, maxlen=self.mL)
        Xcodes = pad_sequences(Xcodes, maxlen=self.mL)
        Ycodes = pad_sequences(Ycodes, maxlen=self.mL)
        self.x_size = len(self.x_codes)
        self.y_size = len(self.y_codes)
        ipt = Input(shape=(self.mL,))
        e = Embedding(self.x_size, self.embedding_size, mask_zero=True)(ipt)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if use_ext_embeddings:
            embedding_matrix = numpy.zeros((self.x_size, self.embedding_size))
            for word, i in self.x_codes.items():
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            e = Embedding(self.x_size, self.embedding_size, weights=[
                embedding_matrix], mask_zero=True, trainable=True)(ipt)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        h = Bidirectional(LSTM(self.memory_size, return_sequences=True))(e)
        o = TimeDistributed(
            Dense(self.y_size, bias_regularizer=l1_l2(0.), activation='softmax'))(h)
        earlystop = EarlyStopping(monitor='val_acc', patience=0)
        self.model = Model(ipt, o)
        if verbose:
            self.model.summary()
        self.model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(Xcodes, Ycodes, epochs=epochs,
                       verbose=verbose, batch_size=batch_size, shuffle=True, validation_data=(D_Xcodes, D_Ycodes), callbacks=[earlystop])
        return self

    def save(self, model_save="tagger.model.h5", pickle_save="tagger.pickle"):
        self.model.save(model_save)
        with open(pickle_save, "wb") as ostr:
            pickle.dump((self.mL, self.x_codes), ostr)
        return self

    @classmethod
    def load(cls, model_save="tagger.model.h5", pickle_save="tagger.pickle"):
        tagger = NNTagger()
        tagger.model = load_model(model_save)
        with open(pickle_save, "rb") as istr:
            tagger.mL, tagger.x_codes = pickle.load(istr)
        return tagger

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

    def phrase2pos(self, phrase):
        x_c, _ = corpus.phrase2extraction(phrase)
        print(corpus.phrase2pos(self.predict(x_c)))


if __name__ == "__main__":
    # for embedding_size in range(20, 100, 10):
    #    for memory_size in range(20, 100, 10):
    #        print(NNTagger().train("sequoia-corpus.np_conll.train", verbose=1).test("sequoia-corpus.np_conll.test"))

    print()
    print("TAGGER WITH EXTERNAL EMBEDDINGS")
    print("_________________________________________________________________")
    tagger = NNTagger().train("sequoia-corpus.np_conll.train",
                              "sequoia-corpus.np_conll.dev", verbose=1)

    print()
    print("TAGGER WITHOUT EXTERNAL EMBEDDINGS")
    print("_________________________________________________________________")
    tagger_no_WE = NNTagger().train("sequoia-corpus.np_conll.train", "sequoia-corpus.np_conll.dev",
                                    use_ext_embeddings=False, verbose=1)
    # print(tagger.predict(corpus.extract(
    #     corpus.load("sequoia-corpus.np_conll.dev"))[0]))
    print()
    print("TAGGER WITH EXTERNAL EMBEDDINGS")
    print(tagger.test("sequoia-corpus.np_conll.test"))
    print("TAGGER WITHOUT EXTERNAL EMBEDDINGS")
    print(tagger_no_WE.test("sequoia-corpus.np_conll.test"))
    while True:
        print(">", end=' ')
        phrase = input()
        if phrase == "BREAK":
            break
        print(tagger.phrase2pos(phrase.strip()))
        print(tagger_no_WE.phrase2pos(phrase.strip()))
