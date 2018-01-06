
def split(filename, proportions=(("train", .8), ("dev", .1), ("test", .1))) :
	with open(filename, "r") as i :
		text = [s.split("\n") for s in i.read().split("\n\n")]
	size = len(text)
	start = 0
	for p in proportions :
		end = int(start + p[1]*size)
		with open(filename+"."+p[0], "w") as o :
			o.write("\n\n".join(["\n".join(s) for s in text[start:end]]))
		start = end
	return tuple([filename+"."+p[0] for p in proportions])

if __name__ == "__main__" :
	split("sequoia-corpus.np_conll")
