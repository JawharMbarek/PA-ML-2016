import numpy
import matplotlib.pyplot as plt

def load_bin_vec(fname, words):
    '''Loads 300x1 word vecs from Google (Mikolov) word2vec.'''
    vocab = set(words)
    word_vecs = {}

    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = numpy.dtype('float32').itemsize * layer1_size
        print('vocab_size, layer1_size', vocab_size, layer1_size)
        count = 0

        for i, line in enumerate(range(vocab_size)):
            if i % 1000 == 0:
                print('.',)

            word = []

            while True:
                ch = f.read(1)

            if ch == ' ' or ch == '':
                print(i, word)
                word = ''.join(word)
                break

            if ch != '\n':
                word.append(ch)

            if word in vocab:
                count += 1
                word_vecs[word] = numpy.fromstring(f.read(binary_len),
                                                   dtype='float32')
            else:
                f.read(binary_len)

        print("done")
        print("Words found in wor2vec embeddings", count)
        return word_vecs


def load_glove_vec(fname, words, delimiter, dim):
    vocab = set(words)
    word_vecs = {}

    with open(fname) as f:
        count = 0

        for line in f:
            if line == '':
                continue

            splits = line.replace('\n', '').split(delimiter)
            word = splits[0]

            if (word in vocab) or (word.lower() in vocab) or len(vocab) == 0:
                count += 1
                word_vecs[word] = numpy.asarray(splits[1:dim + 1],
                                                dtype='float32')

                if count % 100000 == 0:
                    print('Word2Vec count: ', count)

    return word_vecs

def set_figure_size(X=12, Y=8):
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = X
    fig_size[1] = Y
    plt.rcParams['figure.figsize'] = fig_size
