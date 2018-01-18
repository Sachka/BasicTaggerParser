import random

COLS = {k: i for i, k in enumerate(
    ["index", "token", "lemma", "POS", "XPOS", "features", "head", "rel", "dhead", "drel"])}


def load(filename, randomize=False):
    text = []
    with open(filename, "r") as i:
        text = [[t.split("\t") for t in s.split("\n")]
                for s in i.read().split("\n\n") if s != ""]
    if randomize:
        random.shuffle(text)
    return text


def split(filename, proportions=(("train", .8), ("dev", .1), ("test", .1)), randomize=False):
    text = load(filename, randomize=randomize)
    size = len(text)
    start = 0
    for p in proportions:
        end = int(start + p[1] * size)
        with open(filename + "." + p[0], "w") as o:
            o.write("\n\n".join(["\n".join(["\t".join(t) for t in s])
                                 for s in text[start:end]]))
        start = end
    return tuple([filename + "." + p[0] for p in proportions])


def extract(corpus, columns=("token", "POS")):
    return [[[w[COLS[c]] for w in s] for s in corpus] for c in columns]


def flat_extract(corpus, columns=("token", "POS")):
    return [[w[COLS[c]] for w in s] for s in corpus for c in columns]


def extract_features_for_depency(filename):
    index, token, POS, head = extract(
        load(filename), columns=("index", "token", "POS", "head"))
    sentence_list = []
    for idx in range(len(index)):
        sentence_conll = ""
        sentence_idx = index[idx]
        sentece_tokens = token[idx]
        sentence_POS = POS[idx]
        sentence_head = head[idx]
        for j in range(len(sentence_idx)):
            sentence_conll += sentence_idx[j] + "\t" + sentece_tokens[j] + \
                "\t" + sentence_POS[j] + "\t" + sentence_head[j] + "\n"
        sentence_list.append(sentence_conll)
    return sentence_list


if __name__ == "__main__":
    corpus = load(split("sequoia-corpus.np_conll", randomize=True)[0])
    extract(corpus)[1][0][0]
    flat_extract(corpus)[1][0]
