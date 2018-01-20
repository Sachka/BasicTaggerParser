
# import tagger
import io
import numpy as np
from collections import defaultdict

from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Embedding, LSTM, TimeDistributed, Bidirectional, Flatten, concatenate, Masking
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.preprocessing.sequence import pad_sequences

import corpus
from tagger import NNTagger

class DependencyTree:

    def __init__(self,tokens=None, edges=None):
        self.edges  = [] if edges is None else edges                      #couples (gov_idx,dep_idx)
        self.tokens = [('$ROOT$','$ROOT$')] if tokens is None else tokens #couples (wordform,postag)
    
    def __str__(self):
        gdict = dict([(d,g) for (g,d) in self.edges])
        return '\n'.join(['\t'.join([str(idx+1),tok[0],tok[1],str(gdict[idx+1])]) for idx,tok in enumerate(self.tokens[1:])])
                     
    def __make_ngrams(self):
        """
        Makes word representations suitable for feature extraction
        """
        BOL = '@@@'
        EOL = '$$$'
        wordlist = [BOL] + list([w for w,t in self.tokens]) + [EOL]
        taglist = [BOL] + list([t for w,t in self.tokens]) + [EOL]
        word_trigrams = list(zip(wordlist,wordlist[1:],wordlist[2:]))
        tag_trigrams  = list(zip(taglist,taglist[1:],taglist[2:]))
        self.tokens   = list(zip(wordlist[1:-1],taglist[1:-1],word_trigrams,tag_trigrams))
        
    @classmethod
    def read_tree(cls, istream):
        """
        Reads a tree from an input stream
        @param istream: the stream where to read from
        @return: a DependencyTree instance 
        """
        deptree = DependencyTree()
        bfr = istream.readline()
        while True:
            if (bfr.isspace() or bfr == ''):
                if deptree.N() > 1:
                    deptree.__make_ngrams()
                    return deptree
                bfr = istream.readline()
            else:
                idx, word, tag, governor_idx = bfr.split()
                deptree.tokens.append((word,tag))
                deptree.edges.append((int(governor_idx),int(idx)))
                bfr = istream.readline()
        deptree.__make_ngrams()
        return deptree

    def accurracy(self,other):
        """
        Compares this dep tree with another by computing their UAS.
        @param other: other dep tree
        @return : the UAS as a float
        """
        assert(len(self.edges) == len(other.edges))
        S1 = set(self.edges)
        S2 = set(other.edges)
        return len(S1.intersection(S2)) / len(S1)
    
    def N(self):
        """
        Returns the length of the input
        """
        return len(self.tokens)
    
    def __getitem__(self,idx):
        """
        Returns the token at index idx
        """
        return self.tokens[idx]

class DependencyParser:

    #actions
    LEFTARC  = "L"
    RIGHTARC = "R"
    SHIFT    = "S"
    TERMINATE= "T"
    
    def __init__(self):
        self.weights = defaultdict(float)
        self.nn_parser = None 
        
    def oracle_derivation(self,ref_parse):
        """
        This generates an oracle reference derivation from a sentence
        @param ref_parse: a DependencyTree object
        @return : the oracle derivation
        """

        def static_oracle(configuration,reference_arcs,N):
            S,B,A,score = configuration
            all_words   = range(N)
            if len(S) >= 2:
               i,j = S[-2],S[-1]
               if j!=0 and (i,j) in reference_arcs and all([ (j,k) in A for k in all_words if (j,k)  in reference_arcs]):
                    return DependencyParser.RIGHTARC
               elif i!= 0 and (j,i) in reference_arcs and all([ (i,k) in A for k in all_words if (i,k)  in reference_arcs]):
                    return DependencyParser.LEFTARC
            if B:
               return DependencyParser.SHIFT
            return DependencyParser.TERMINATE

        sentence = ref_parse.tokens
        edges    = set(ref_parse.edges)
        N        = len(sentence)
        C = (tuple(),tuple(range(len(sentence))),tuple(),.0) #A config is a hashable quadruple with score 
        action     = None
        derivation = [(action,C)]
        
        while action != DependencyParser.TERMINATE :
                        
            action = static_oracle(C,edges,N)
            
            if action ==  DependencyParser.SHIFT:
                C = self.shift(C,sentence)
            elif action == DependencyParser.LEFTARC:
                C = self.leftarc(C,sentence)
            elif action == DependencyParser.RIGHTARC:
                C = self.rightarc(C,sentence)
            elif action ==  DependencyParser.TERMINATE:
                C = self.terminate(C,sentence)
                
            derivation.append((action,C))
            
        return derivation
                
    def shift(self,configuration,tokens):
        """
        Performs the shift action and returns a new configuration
        """
        S,B,A,score = configuration
        w0 = B[0]
        return (S + (w0,), B[1:], A, self.score(configuration,DependencyParser.SHIFT,tokens)) 

    def leftarc(self,configuration,tokens):
        """
        Performs the left arc action and returns a new configuration
        """
        S,B,A,score = configuration
        i,j = S[-2],S[-1]
        return (S[:-2]+(j,), B, A + ((j,i),), self.score(configuration,DependencyParser.LEFTARC,tokens)) 

    def rightarc(self,configuration,tokens):
        S,B,A,score = configuration
        i,j = S[-2],S[-1]
        return (S[:-1], B, A + ((i,j),), self.score(configuration,DependencyParser.RIGHTARC,tokens)) 

    def terminate(self,configuration,tokens):
        S,B,A,score = configuration
        return (S, B, A, self.score(configuration,DependencyParser.TERMINATE,tokens))        


    def parse_one(self,sentence,beam_size=4,get_beam=False):
        
        actions = [DependencyParser.LEFTARC,\
                   DependencyParser.RIGHTARC,\
                   DependencyParser.SHIFT,\
                   DependencyParser.TERMINATE]

        N = len(sentence)
        init = (tuple(),tuple(range(N)),tuple(),.0) #A config is a hashable quadruple with score 
        current_beam = [(-1,None,init)]
        beam = [current_beam]
            
        for i in range(2*N): #because 2N-1+terminate
            next_beam = []
            for idx, (_,action,config) in enumerate(current_beam):
                S,B,A,score = config
                for a in actions:
                    if a ==  DependencyParser.SHIFT:
                        if B:
                            newconfig = self.shift(config,sentence)
                            next_beam.append((idx,a,newconfig))
                    elif a == DependencyParser.LEFTARC:
                        if len(S) >= 2 and S[-2] != 0: #a word cannot dominate the dummy root
                            newconfig = self.leftarc(config,sentence)
                            next_beam.append((idx,a,newconfig))
                    elif a == DependencyParser.RIGHTARC:
                        if len(S) >= 2:
                            newconfig = self.rightarc(config,sentence)
                            next_beam.append((idx,a,newconfig))
                    elif a == DependencyParser.TERMINATE:
                        if len(S) < 2 and not B:
                            newconfig = self.terminate(config,sentence)
                            next_beam.append((idx,a,newconfig))
            next_beam.sort(key=lambda x:x[2][3],reverse=True)
            next_beam = next_beam[:beam_size]
            beam.append(next_beam)
            current_beam = next_beam
        if get_beam:
            return beam
        else:
            succ = beam[-1][0][2] #success in last beam, top position, newconfig
            print(beam[-1][0][1],succ)
            return DependencyTree(tokens=sentence,edges=succ[2])

    def encode(self, seq, toks):
        return pad_sequences([[self.x_dict[toks[t]] if toks[t] in self.x_dict else self.x_dict["__UNK__"] for t in seq]], maxlen=self.mL)

    def score(self,configuration,action,tokens):
        """
        Computes the prefix score of a derivation
        @param configuration : a quintuple (S,B,A,score,history)
        @param action: an action label in {LEFTARC,RIGHTARC,TERMINATE,SHIFT}
        @param tokens: the x-sequence of tokens to be parsed
        @return a prefix score
        """
        S,B,A,old_score = configuration
        """if self.nn_parser is not None :
           print(self.nn_parser.predict([self.encode(S, tokens), self.encode(B, tokens)]))
           print(self.nn_parser.predict([self.encode(S, tokens), self.encode(B, tokens)])[0, self.y_dict[action]])"""
        return old_score + (self.nn_parser.predict([self.encode(S, tokens), self.encode(B, tokens)])[0, self.y_dict[action]] if self.nn_parser is not None else 0.)
    
    def test(self,dataset,beam_size=4):
        """
        @param dataset: a list of DependencyTrees
        @param beam_size: size of the beam
        """
        N       = len(dataset)
        sum_acc = 0.0
        for ref_tree in dataset:
            tokens    = ref_tree.tokens
            pred_tree = self.parse_one(tokens,beam_size)
            print(pred_tree)
            print()
            sum_acc   += ref_tree.accurracy(pred_tree)
        return sum_acc/N

        
    def train(self, dataset, tagger, epochs=10):
        """
        @param dataset : a list of dependency trees
        """
        N = len(dataset)
        sequences = [(dtree.tokens, self.oracle_derivation(dtree)) for dtree in dataset]

        X_S, X_B, Y = [], [], []
        def map_tokens(S, B, tokens) :
            S = [tagger.x_codes[tokens[s][0]] if tokens[s][0] in tagger.x_codes else tagger.x_codes["__UNK__"] for s in S]
            B = [tagger.x_codes[tokens[b][0]] if tokens[b][0] in tagger.x_codes else tagger.x_codes["__UNK__"] for b in B]
            return S,B
        ##### DARK SIDE #####
        for tokens, ref_derivation in sequences:
            current_config = ref_derivation[0][1]
            for action, config in ref_derivation[1:]: #do not learn the "None" dummy action
                S,B,_,_ = current_config
                current_config = config
                S,B = map_tokens(S, B, tokens)
                X_S.append(S)
                X_B.append(B)
                Y.append(action)
        XS_encoded, XB_encoded = pad_sequences(X_S, maxlen=tagger.mL), pad_sequences(X_B, maxlen=tagger.mL)
        self.x_dict = tagger.x_codes
        self.mL = tagger.mL
        y_size = len(set(Y))
        y_list=list(set(Y))
        self.y_dict = {y: i for i, y in enumerate(y_list)}
        self.reverse_y_dict=dict(enumerate(y_list))
        Y_encoded = np.zeros(shape=(len(Y), y_size))
        for i,y in enumerate(Y) :
            Y_encoded[i, self.y_dict[y]] = 1.
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
        o = Dense(y_size, activation="softmax")(l3)
        self.nn_parser = Model([ipt_stack, ipt_buffer], o)
        self.nn_parser.summary()
        sgd = RMSprop(lr=.01)
        self.nn_parser.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        self.nn_parser.fit([XS_encoded,XB_encoded], Y_encoded, epochs=epochs, verbose=1)
        return self

if __name__ == "__main__" :
    print("Direct test")
    train_conll = "sequoia-corpus.np_conll.train"
    test_conll = "sequoia-corpus.np_conll.test"
    dev_conll = "sequoia-corpus.np_conll.dev"

    nnt = NNTagger()
    # # nnt.train("sequoia-corpus.np_conll.train", verbose=1)
    # nnt.save()
    nnt = NNTagger.load()
    # nnt.model.summary()

    X = corpus.extract_features_for_depency(train_conll)
    XIO = list(map(io.StringIO, X))
    XD = list(map(DependencyTree.read_tree, XIO))

    Xtest = corpus.extract_features_for_depency(test_conll)
    XtestIO = list(map(io.StringIO, Xtest))
    XtestD = list(map(DependencyTree.read_tree, XtestIO))




    p = DependencyParser()
    p.train(XD, nnt)
    print(p.test(XtestD[:10]))
