# import tagger
import io
import numpy as np
from collections import defaultdict, namedtuple

from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Embedding, LSTM, TimeDistributed, Bidirectional, Flatten, concatenate, Masking
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.preprocessing.sequence import pad_sequences

import corpus
from tagger import NNTagger

#readability
Config = namedtuple("Config", ["stack", "buffer", "arcs"])

def initial_config(number_tokens) :
	return Config(tuple(), tuple(range(number_tokens)), tuple())

class DependencyParser :
	#transition system
	@classmethod
	def right(cls, config) :
		s, b, a = config.stack, config.buffer, config.arcs
		i, j = s[-2:]
		return Config(s[:-2] + (i,), b, a + ((i, j),))
	@classmethod
	def left(cls, config) :
		s, b, a = config.stack, config.buffer, config.arcs
		i, j = s[-2:]
		return Config(s[:-2] + (j,) , b, a + ((j, i),))
	@classmethod
	def shift(cls, config) :
		s, b, a = config.stack, config.buffer, config.arcs
		return Config(s + (b[0],), b[1:], a)
 
	ACTIONS = {"RIGHT" : lambda c : DependencyParser.right(c), "LEFT" : lambda c : DependencyParser.left(c), "SHIFT": lambda c: DependencyParser.shift(c)}

	def possible_actions(self, config) :
		actions =[]
		if len(config.stack) > 2 :
			actions.append("RIGHT")
			actions.append("LEFT")
		if len(config.buffer) :
			actions.append("SHIFT")
		return actions 
	#oracle
	@classmethod
	def oracle(cls, arcs) :
		def step(current_config, ref_arcs, indices) :
			if len(current_config.stack) >= 2 :
				i, j = current_config.stack[-2:]
				if (i, j) in ref_arcs and all([(j, k) in current_config.arcs for k in indices if (j, k) in ref_arcs]) :
					return "RIGHT"
				if (j, i) in ref_arcs and all([(i, k) in current_config.arcs for k in indices if (i, k) in ref_arcs]) :
					return "LEFT"
			if len(config.buffer) != 0 :
				return "SHIFT"
			return None #terminate
		idx = max({i for a in arcs for i in a}) + 1
		derivation = []
		config = initial_config(idx)
		indices = tuple(range(idx))
		action = step(config, arcs, indices)
		while action is not None :
			derivation.append((action, config))
			config = DependencyParser.ACTIONS[action](config)
			action = step(config, arcs, indices)
		print(config)
		return derivation, config

	def beam_parse(self, sentence, beam_size):
		N =  len(sentence)
		configs = [initial_config(N)]
		prev_scores = {config[0] : 0.}
		for i in range(2*N - 1) :
			for config in sorted(configs, key=lambda c: prev_scores[c], reverse=True)[:beam_size] :
				for action in self.possible_actions(c) :
					new_config = DependencyParser.ACTIONS[action](config)
					prev_score[new_config] = self.score(new_config, sentence, action) + prev_scores[config]
					configs.append(new_config)
		return max(config, key=lambda c: prev_scores[c])

	def _encode(self, sequence) :
		return pad_sequences([[self.x_codes[t] for t in sequence]])

	def _index_for(self, action) :
		return self.y_codes[action]

	def score(self, config, sentence, action) :
		return self.model.predict([self._encode(config.stack), self._encode(config.buffer)])[0,self._index_for(action)]

	def __init__(self) :
		self.model = None

	def train(self, dataset, tagger, epochs=3):
		"""
		@param dataset : a list of dependency trees
		"""
		N = len(dataset)
		sequences = dataset

		X_S, X_B, Y = [], [], []
		def map_tokens(S, B, tokens) :
			S = [tagger.x_codes[tokens[s]] if tokens[s] in tagger.x_codes else tagger.x_codes["__UNK__"] for s in S]
			B = [tagger.x_codes[tokens[b]] if tokens[b] in tagger.x_codes else tagger.x_codes["__UNK__"] for b in B]
			return S,B
		##### DARK SIDE #####
		for tokens, ref_derivation in sequences:
			for action, config in ref_derivation:
				S,B = config.stack, config.buffer
				S,B = map_tokens(S, B, tokens)
				X_S.append(S)
				X_B.append(B)
				Y.append(action)
		XS_encoded, XB_encoded = pad_sequences(X_S, maxlen=tagger.mL), pad_sequences(X_B, maxlen=tagger.mL)
		self.x_codes = tagger.x_codes
		self.mL = tagger.mL
		self.y_size = len(set(Y))
		self.y_list=list(set(Y))
		self.y_codes = {y: i for i, y in enumerate(self.y_list)}
		self.reverse_y_codes=dict(enumerate(self.y_list))
		Y_encoded = np.zeros(shape=(len(Y), self.y_size))
		for i,y in enumerate(Y) :
			Y_encoded[i, self.y_codes[y]] = 1.
		###$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###
		# print(XS_encoded)

		# exit()
		ipt_stack = Input(shape=(tagger.mL,))
		ipt_buffer = Input(shape=(tagger.mL,))
		e_stack = tagger.model.get_layer("embedding_1")(ipt_stack)
		e_buffer = tagger.model.get_layer("embedding_1")(ipt_buffer)
		l_s = tagger.model.get_layer("bidirectional_1")(e_stack)
		l_b = tagger.model.get_layer("bidirectional_1")(e_buffer)
		l1 = LSTM(122)
		
		l1 = concatenate([l1(l_s), l1(l_b)], axis=1)
		# l2 = Flatten())
		l3 = Dense(122)(l1)
		o = Dense(self.y_size, activation="softmax")(l3)
		self.model = Model([ipt_stack, ipt_buffer], o)
		self.model.summary()
		sgd = RMSprop(lr=.01)
		self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
		self.model.fit([XS_encoded,XB_encoded], Y_encoded, epochs=epochs, verbose=1, validation_split=.2)
		#print(self.model.evaluate([tXS, tXB], tY, batch_size=64))
		return self

if __name__ == "__main__" :
	def message(i,x,y, ref, cur):
		return "reference:\n" + str(list(sorted(ref))) + "\n\ncurrent:\n" +str(list(sorted(cur))) +"\n\nfor input:\n" + "\n".join(map("\t".join, zip(i,x,y)))
	print("Direct test")
	train_conll = "sequoia-corpus.np_conll.train"
	test_conll = "sequoia-corpus.np_conll.test"
	dev_conll = "sequoia-corpus.np_conll.dev"

	nnt = NNTagger()
	# # nnt.train("sequoia-corpus.np_conll.train", verbose=1)
	# nnt.save()
	nnt = NNTagger.load()
	# nnt.model.summary()

	I, X, Y = corpus.extract(corpus.load(dev_conll), columns=("index", "token","head"))
	dataset=[]
	for i, x, y in zip(I, X, Y) :
		arcs = set(zip(map(int, y), map(int, i)))
		derivation, last_config = DependencyParser.oracle(arcs)
		if list(sorted(arcs)) != list(sorted(last_config.arcs)) :
			print(str(last_config) +"\n\n\n" + message(i,x,y, arcs ,last_config.arcs))
			continue
		dataset.append((["__ROOT__"] + x, derivation))
	print(len(X), len(dataset))
	"""
	Xtest = corpus.extract_features_for_depency(test_conll)
	XtestIO = list(map(io.StringIO, Xtest))
	XtestD = list(map(DependencyTree.read_tree, XtestIO))
	"""

	p = DependencyParser()
	p.train(dataset, nnt)
	
	print(p.test(XtestD[:10]))
