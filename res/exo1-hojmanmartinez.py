# encoding: utf-8
import random
# import matplotlib.pyplot as plt
from SparseWeightVector import SparseWeightVector

test = "src/sequoia-corpus.np_conll"


def split(filename, randomize=False, proportions=(0.8, 0.1, 0.1)):
    """
    Divide file in three files: train, dev. test
    """
    sentence = []
    sentences = []
    instream = open(filename, 'r')
    line = instream.readline()
    while line:
        if line == "\n":
            if len(sentence) != 0:
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line)
        line = instream.readline()
    if len(sentence) != 0:
        sentences.append(sentence)
    if randomize:
        random.shuffle(sentences)
    instream.close()
    # print(len(sentences))
    train = sentences[:int(len(sentences) * proportions[0])]
    dev = sentences[int(len(sentences) * proportions[0]):int(
        len(sentences) * proportions[0] + len(sentences) * proportions[1])]
    test = sentences[int(
        len(sentences) * proportions[0] + len(sentences) * proportions[1]):]
    trainfile = open(filename + ".train", 'w')
    devfile = open(filename + ".dev", 'w')
    testfile = open(filename + ".test", 'w')
    for sentence in train:
        for word in sentence:
            trainfile.write(word)
        trainfile.write("\n")
    for sentence in dev:
        for word in sentence:
            devfile.write(word)
        devfile.write("\n")
    for sentence in test:
        for word in sentence:
            testfile.write(word)
        testfile.write("\n")
    trainfile.close()
    devfile.close()
    testfile.close()


def read_corpus(filename):
    """
    Returns corpus de train/dev ou test pour AVGPerceptron
    """
    sentence = ""
    sentences = []
    instream = open(filename, 'r', encoding="utf-8")
    line = instream.readline()
    while line:
        word_annotation = line.split()
        if line == "\n":
            if sentence != "":
                sentences.append(sentence)
            sentence = ""
        else:
            if RepresentsFloat(word_annotation[1]) or RepresentsInt(
                    word_annotation[1]):
                sentence += " " + "__NUM__" + "}" + word_annotation[3]
            else:
                # sentence += " " + word_annotation[1].lower() + "}" + word_annotation[3]
                sentence += " " + word_annotation[1] + "}" + word_annotation[3]
        line = instream.readline()
    if sentence != "":
        sentences.append(sentence)
    instream.close()
    return make_dataset(sentences)


def make_dataset(text):
    """
    @param text: a list of strings of the form : Le}D chat}N mange}V la}D souris}N .}PONCT
    @return    : an n-gram style dataset
    """
    BOL = '@@@'
    EOL = '$$$'

    dataset = []
    for line in text:
        line = list([tuple(w.split('}')) for w in line.split()])
        tokens = [BOL] + list([tok for (tok, pos) in line]) + [EOL]
        pos = list([pos for (tok, pos) in line])
        tags = [BOL] + list([tag for (tok, tag) in line]) + [EOL]
        tok_trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
        tok_bigramsL = list(zip(tokens, tokens[1:]))
        tok_bigramsR = list(zip(tokens[1:], tokens))
        toks = (tokens[1:-1])  # token itself
        suff = [tok[-3:] for tok in toks]  # suffixe
        preff = [tok[:3] for tok in toks]  # preffixe
        maj = [tok[0].isupper() for tok in toks]  # starts with uppercase
        tags_1 = (tags[:-2])  # tag token -1

        dataset.extend(
            zip(pos,
                zip(tok_trigrams, tok_bigramsL, tok_bigramsR, toks, suff,
                    preff, maj, tags_1)))  # maj

    return dataset


class AvgPerceptron:
    """
    Averaged Perceptron
    """

    def __init__(self):

        self.model = SparseWeightVector()
        self.Y = []  # classes
        self.model_avg = SparseWeightVector()

    def train(self, dataset, dev, step_size=0.1, max_epochs=20):

        self.Y = list(set([y for (y, x) in dataset]))
        dev_accs = []
        train_accs = []
        train_losses = []
        T = 1.
        avg_cumul = SparseWeightVector()
        for e in range(max_epochs):
            loss = 0.0
            random.shuffle(dataset)
            for y, x in dataset:
                ypred = self.tag(x)
                if y != ypred:
                    loss += 1.0
                    delta_ref = SparseWeightVector.code_phi(x, y)
                    delta_pred = SparseWeightVector.code_phi(x, ypred)
                    update = step_size * (delta_ref - delta_pred)
                    self.model += update
                    avg_cumul += T * update
                    T += 1
            # Calculate accuracy
            if len(dataset) != 0:
                acc = (len(dataset) - loss) / len(dataset)
            else:
                acc = (len(dataset) - loss)
            train_accs.append(acc)
            # Stock Loss
            train_losses.append(loss)
            # Update Avg Perceptron
            self.model_avg = self.model - avg_cumul / T
            # Calculate acc for dev corpus
            dev_acc = self.test(dev, True)
            dev_accs.append(dev_acc)
            # Print loss et acc
            print("Epoch = " + str(e) + ", Loss (#errors) = " + str(loss) +
                  ", Accuracy = " + str(acc * 100) + ", Dev acc = " +
                  str(dev_acc * 100))
            # Stop if loss null
            if loss == 0.0:
                return train_losses, train_accs, dev_accs
        return train_losses, train_accs, dev_accs

    def predict(self, dataline, avg=False):
        if avg:
            return list([self.model_avg.dot(dataline, c) for c in self.Y])
        else:
            return list([self.model.dot(dataline, c) for c in self.Y])

    def tag(self, dataline, avg=False):
        scores = self.predict(dataline, avg)
        imax = scores.index(max(scores))
        return self.Y[imax]

    def test(self, dataset, avg=False):

        result = list([(y == self.tag(x, avg)) for y, x in dataset])
        return sum(result) / len(result)


def RepresentsInt(s):
    """check if str is int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def RepresentsFloat(s):
    """check if str is float"""
    try:
        float(s)
        return True
    except ValueError:
        return False


# train = read_corpus(test+".train")
# dev = read_corpus(test+".dev")
# test = read_corpus(test+".test")
# perc = AvgPerceptron()
# train_loss, train_acc, dev_acc = perc.train(train, dev, step_size=1.0, max_epochs=5)
# print("Test : normal et avg")
# print(perc.test(test))
# print(perc.test(test, True))
#
# #plots :accuracy and loss
#
# plt.plot(train_acc, 'b', label='train_acc')
# plt.plot(dev_acc, 'r', label='dev_acc')
# plt.title("Accuracy")
# plt.legend()
# plt.show()
# plt.close()
#
#
# plt.plot(train_loss, 'b', label='train_loss')
# plt.title("Loss")
# plt.legend()
# plt.show()
# plt.close()

split(test)
trainc = read_corpus(test + ".train")
devc = read_corpus(test + ".dev")
testc = read_corpus(test + ".test")
p = AvgPerceptron()
p.train(trainc, devc)
print(p.test(testc, False))
