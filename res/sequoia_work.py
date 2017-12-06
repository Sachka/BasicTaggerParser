# Isabel Hojman & Hermes Martinez


def split(filename):
    corpus = []
    with open(filename, "r") as file:
        corpus = [line.strip() for line in file.readlines()]
    trainc = corpus[:int(len(corpus) * 0.8)]
    testc = corpus[int(len(corpus) * 0.8):int(len(corpus) * 0.9)]
    devc = corpus[int(len(corpus) * 0.9):]
    with open(filename + ".train", "w") as train_output:
        for line in trainc:
            train_output.write("%s\n" % line)
    with open(filename + ".test", "w") as test_output:
        for line in testc:
            test_output.write("%s\n" % line)
    with open(filename + ".dev", "w") as devc_output:
        for line in devc:
            devc_output.write("%s\n" % line)


def read_corpus():
