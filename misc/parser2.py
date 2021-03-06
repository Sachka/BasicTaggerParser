import io
from collections import defaultdict
from SparseWeightVector import SparseWeightVector
from keras.utils import pad_sequences

#DATA REPRESENTATION
class DependencyTree:

    def __init__(self, tagger,tokens=None, edges=None):
	self.intermediate_model=Model(tagger.model.inputs, tagger.model.get_layer("bidirectional_1").output)
        self.x_codes = defaultdict(lambda:0)
	self.x_codes.update(tagger.x_codes)
        self.mL = tagger.mL
        self.edges  = [] if edges is None else edges                      #couples (gov_idx,dep_idx)
        self.tokens = [('$ROOT$','$ROOT$')] if tokens is None else tokens #couples (wordform,postag)
    
    def prep(self, tokens_seq) : 
        return self.intermediate_model.predict(pad_sequences([[self.x_codes[t] for t in tokens_seq]], maxlen = self.mL))[-len(tokens_seq):]
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
        vectors = tagger.model.predict(pad_sequences([[tagger.x_codes[w] if w in tagger.x_codes else tagger.x_codes["__UNK__"] for w,t in self.tokens]]))[-len(self.tokens):] 
        self.tokens   = list(zip(wordlist[1:-1],taglist[1:-1],vectors))
        
    @staticmethod
    def read_tree(istream):
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

class ArcStandardTransitionParser:

    #actions
    LEFTARC  = "L"
    RIGHTARC = "R"
    SHIFT    = "S"
    TERMINATE= "T"
    
    def __init__(self):
        self.model = SparseWeightVector()
        
    @staticmethod
    def static_oracle(configuration,reference_arcs,N):
        """
        @param configuration: a parser configuration
        @param reference arcs: a set of dependency arcs
        @param N: the length of the input sequence
        """
        S,B,A,score = configuration
        all_words   = range(N)
        if len(S) >= 2:
           i,j = S[-2],S[-1]
           if j!=0 and (i,j) in reference_arcs and all([ (j,k) in A for k in all_words if (j,k)  in reference_arcs]):
                return ArcStandardTransitionParser.RIGHTARC
           elif i!= 0 and (j,i) in reference_arcs and all([ (i,k) in A for k in all_words if (i,k)  in reference_arcs]):
                return ArcStandardTransitionParser.LEFTARC
        if B:
            return ArcStandardTransitionParser.SHIFT
        return ArcStandardTransitionParser.TERMINATE

    
    def oracle_derivation(self,ref_parse):
        """
        This generates an oracle reference derivation from a sentence
        @param ref_parse: a DependencyTree object
        @return : the oracle derivation
        """
        sentence = ref_parse.tokens
        edges    = set(ref_parse.edges)
        N        = len(sentence)
        
        C = (tuple(),tuple(range(len(sentence))),tuple(),0.0) #A config is a hashable quadruple with score 
        action     = None
        derivation = [(action,C)]
        
        while action != ArcStandardTransitionParser.TERMINATE :
                        
            action = ArcStandardTransitionParser.static_oracle(C,edges,N)
            
            if action ==  ArcStandardTransitionParser.SHIFT:
                C = self.shift(C,sentence)
            elif action == ArcStandardTransitionParser.LEFTARC:
                C = self.leftarc(C,sentence)
            elif action == ArcStandardTransitionParser.RIGHTARC:
                C = self.rightarc(C,sentence)
            elif action ==  ArcStandardTransitionParser.TERMINATE:
                C = self.terminate(C,sentence)
                
            derivation.append((action,C))
            
        return derivation
                
    def shift(self,configuration,tokens):
        """
        Performs the shift action and returns a new configuration
        """
        S,B,A,score = configuration
        w0 = B[0]
        return (S + (w0,),B[1:],A,score+self.score(configuration,ArcStandardTransitionParser.SHIFT,tokens)) 

    def leftarc(self,configuration,tokens):
        """
        Performs the left arc action and returns a new configuration
        """
        S,B,A,score = configuration
        i,j = S[-2],S[-1]
        return (S[:-2]+(j,),B,A + ((j,i),),score+self.score(configuration,ArcStandardTransitionParser.LEFTARC,tokens)) 

    def rightarc(self,configuration,tokens):
        S,B,A,score = configuration
        i,j = S[-2],S[-1]
        return (S[:-1],B, A + ((i,j),),score+self.score(configuration,ArcStandardTransitionParser.RIGHTARC,tokens)) 

    def terminate(self,configuration,tokens):
        S,B,A,score = configuration
        return (S,B,A,score+self.score(configuration,ArcStandardTransitionParser.TERMINATE,tokens))        


    def parse_one(self,sentence,beam_size=4,get_beam=False):
        
        actions = [ArcStandardTransitionParser.LEFTARC,\
                   ArcStandardTransitionParser.RIGHTARC,\
                   ArcStandardTransitionParser.SHIFT,\
                   ArcStandardTransitionParser.TERMINATE]

        N = len(sentence)
        init = (tuple(),tuple(range(N)),tuple(),0.0) #A config is a hashable quadruple with score 
        current_beam = [(-1,None,init)]
        beam = [current_beam]
            
        for i in range(2*N): #because 2N-1+terminate
            next_beam = []
            for idx, (_,action,config) in enumerate(current_beam):
                S,B,A,score = config 
                for a in actions:
                    if a ==  ArcStandardTransitionParser.SHIFT:
                        if B:
                            newconfig = self.shift(config,sentence)
                            next_beam.append((idx,a,newconfig))
                    elif a == ArcStandardTransitionParser.LEFTARC:
                        if len(S) >= 2 and S[-2] != 0: #a word cannot dominate the dummy root
                            newconfig = self.leftarc(config,sentence)
                            next_beam.append((idx,a,newconfig))
                    elif a == ArcStandardTransitionParser.RIGHTARC:
                        if len(S) >= 2:
                            newconfig = self.rightarc(config,sentence)
                            next_beam.append((idx,a,newconfig))
                    elif a == ArcStandardTransitionParser.TERMINATE:
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

             
    def early_prefix(self,ref_parse,beam):
        """
        Finds the prefix for early update, that is the prefix where the ref parse fall off the beam.
        @param ref_parse: a parse derivation
        @param beam: a beam output by the parse_one function
        @return (bool, ref parse prefix, best in beam prefix)
                the bool is True if update required false otherwise
        """
        idx = 0
        for (actionR,configR),(beamCol) in zip(ref_parse,beam):
            found = False
            #print("seeking",configR, "at index",idx)
            for source_idx,action,configTarget in beamCol:
                #print("  ",configTarget)
                if action == actionR and configTarget[:-1] == configR[:-1]: #-1 -> does not test score equality
                    found = True
                    #print("   => found")
                    break
            if not found:
                #print("   => not found")
                #backtrace
                jdx = idx
                source_idx = 0
                early_prefix = []
                while jdx >= 0:
                    new_source_idx,action,config = beam[jdx][source_idx]
                    early_prefix.append( (action,config) )
                    source_idx = new_source_idx
                    jdx -= 1
                early_prefix.reverse()
                return (True, ref_parse[:idx+1],early_prefix)
            idx+=1
        #if no error found check that the best in beam is the ref parse
        last_ref_action,last_ref_config     = ref_parse[-1]
        _,last_pred_action,last_pred_config =  beam[-1][0]
        if last_pred_config[:-1] == last_ref_config[:-1]:
            return (False,None,None)#returns a no update message
        else:#backtrace
            jdx = len(beam)-1
            source_idx = 0
            early_prefix = []
            while jdx >= 0:
                new_source_idx,action,config = beam[jdx][source_idx]
                early_prefix.append( (action,config) )
                source_idx = new_source_idx
                jdx -= 1
            early_prefix.reverse()
            return (True,ref_parse,early_prefix)
                
    def score(self,configuration,action,tokens):
        """
        Computes the prefix score of a derivation
        @param configuration : a quintuple (S,B,A,score,history)
        @param action: an action label in {LEFTARC,RIGHTARC,TERMINATE,SHIFT}
        @param tokens: the x-sequence of tokens to be parsed
        @return a prefix score
        """
        S,B,A,old_score = configuration
        config_repr = self.__make_config_representation(S,B,tokens)
        return old_score + numpy.dot(config_repr,action)

    def __make_config_representation(self,S,B,tokens):
        """
        This gathers the information for coding the configuration as a feature vector.
        @param S: a configuration stack
        @param B  a configuration buffer
        @return an ordered list of tuples 
        """
        #default values for inaccessible positions
        s0w,s1w,s0t,s1t,b0w,b1w,b0t,b1t = "__START__","__START__","__START__","__START__","__START__","__START__","__START__","__START__"

        if len(S) > 0:
            s0w,s0t = tokens[S[-1]][0],tokens[S[-1]][1]
        if len(S) > 1:
            s1w,s1t = tokens[S[-2]][0],tokens[S[-2]][1]
        if len(B) > 0:
            b0w,b0t = tokens[B[0]][0],tokens[B[0]][1]
        if len(B) > 1:
            b1w,b1t = tokens[B[1]][0],tokens[B[1]][1]
            
        wordlist = [s0w,s1w,b0w,b1w]
        taglist  = [s0t,s1t,b0t,b1t]
        word_bigrams = list(zip(wordlist,wordlist[1:]))
        tag_bigrams = list(zip(taglist,taglist[1:]))
        word_trigrams = list(zip(wordlist,wordlist[1:],wordlist[2:]))
        tag_trigrams = list(zip(taglist,taglist[1:],taglist[2:]))
        return word_bigrams + tag_bigrams + word_trigrams + tag_trigrams
    
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

        
    def train(self,dataset,step_size=1.0,max_epochs=100,beam_size=4):
        """
        @param dataset : a list of dependency trees
        """
        N = len(dataset)
        sequences = list([ (dtree.tokens,self.oracle_derivation(dtree)) for dtree in dataset])
        
        for e in range(max_epochs):
            loss = 0.0
            for tokens,ref_derivation in sequences:
                pred_beam = self.parse_one(tokens,beam_size,get_beam=True)
                (update, ref_prefix,pred_prefix) = self.early_prefix(ref_derivation,pred_beam)
                #print('R',ref_derivation)
                #print('P',pred_prefix)
                #self.test(dataset,beam_size)

                if update:
                    #print (pred_prefix)
                    loss += 1.0
                    delta_ref = SparseWeightVector()
                    current_config = ref_prefix[0][1]
                    for action,config in ref_prefix:
                        S,B,A,score = current_config
                        x_repr = self.__make_config_representation(S,B,tokens)
                        delta_ref += SparseWeightVector.code_phi(x_repr,action)
                        current_config = config
                        
                    delta_pred = SparseWeightVector()
                    current_config = pred_prefix[0][1]
                    for action,config in pred_prefix:
                        S,B,A,score = current_config
                        x_repr = self.__make_config_representation(S,B,tokens)
                        delta_pred += SparseWeightVector.code_phi(x_repr,action)
                        current_config = config

                    self.model += step_size*(delta_ref-delta_pred)
            print('Loss = ',loss, "%Exact match = ",(N-loss)/N)
            if loss == 0.0:
                return

            
test = """
1 le   D     2
2 chat N     3
3 dort V     0
4 .    PONCT 3
"""
test2 = """
1 le      D     2
2 tapis   N     3
3 est     V     5
4 rouge   A     3
5 et      CC    0
6 le      D     7
7 chat    N     8
8 mange   V     5
9 la      D     10
10 souris N     8
11 .      PONCT 5
"""

istream = io.StringIO(test)
istream2 =  io.StringIO(test2)
d = DependencyTree.read_tree(istream)
d2 = DependencyTree.read_tree(istream2)
p = ArcStandardTransitionParser()
p.train([d,d2],max_epochs=100,beam_size=4)
print(p.test([d,d2],beam_size=4))
