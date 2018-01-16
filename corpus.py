COLS = {k : i for i, k in enumerate(["index", "token", "lemma", "POS", "XPOS", "features", "head", "rel" , "dhead", "drel"])}

def load(filename) :
	text = []
	with open(filename, "r") as i :
		text = [[t.split("\t") for t in s.split("\n")] for s in i.read().split("\n\n")]
	return text

def split(filename, proportions=(("train", .8), ("dev", .1), ("test", .1))) :
	text = load(filename)
	size = len(text)
	start = 0
	for p in proportions :
		end = int(start + p[1]*size)
		with open(filename+"."+p[0], "w") as o :
			o.write("\n\n".join(["\n".join(["\t".join(t) for t in s]) for s in text[start:end]]))
		start = end
	return tuple([filename+"."+p[0] for p in proportions])

def extract(corpus, columns=("token", "POS")) :
	return [[[w[COLS[c]] for w in s ] for s in corpus] for c in columns]

if __name__ == "__main__" :
	corpus = load(split("sequoia-corpus.np_conll")[0])
	extract(corpus)[1][0][0]
