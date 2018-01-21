# BasicTaggerParser
BTP
authors : hmartinez & tmickus

Download word embeddings from https://www.dropbox.com/s/aja3rygtlvz4qvd/vectors50.bz2

Once the embeddings are placed in the root folder of this project, the complete procedure can be launched just by running the parser.py script.
TODO: Display a projection of any french user input phrase.

A POS Tag direct test is also available by running tagger.py directly. It will display accuracies for both the pretrained embeddings method or by training new embeddings from scratch, we can afterwards try a new sentence in french and see its output prediction. Typing BREAK will terminate the script.

The following python libraries are required:
TensorFlow
Keras
Numpy
picle
pyh5s