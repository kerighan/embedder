from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.snowball import FrenchStemmer
from .stopwords import stopwords_fr
from .model import create_model
from collections import Counter
import numpy as np
import itertools
import re


class Embedder:
    def __init__(
        self,
        dim=128,
        hidden_dim=16,
        min_tf=4,
        max_features=75000,
        n_heads=8,
        batch_size=100,
        maxlen=None,
        epochs=5,
        validation_split=.1
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.min_tf = min_tf
        self.max_features = max_features
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.epochs = epochs
        self.validation_split = validation_split

        self.stopwords = stopwords_fr
        self.stemmer = FrenchStemmer()

    def add_stopwords(self, sw):
        if not isinstance(sw, set):
            sw = set(sw)
        self.stopwords = self.stopwords.union(sw)

    def preprocess(self, series):
        series = series.apply(self.tokenize)
        series = self.stem_series(series)
        return series

    def tokenize(self, text):
        words = re.findall(
            r'([\w]+)',
            text.lower(), re.UNICODE)
        words = [word for word in words if word not in self.stopwords]
        return words

    def stem_series(self, series):
        vocab = set(itertools.chain(*series))
        stem = {word: self.stemmer.stem(word) for word in vocab}
        return series.apply(lambda x: [stem[w] for w in x])

    def fit(self, series):
        # tokenize
        series = series.apply(self.tokenize)

        # filter by counts and stem
        count = Counter(itertools.chain(*series))
        self.vocab = {
            word for word, c in count.most_common(self.max_features)
            if c >= self.min_tf
        }
        self.stem = {
            word: self.stemmer.stem(word)
            for word in self.vocab
        }
        self.n_features = len(self.stem)
        self.word2id = {
            word: i for i, word in enumerate(self.stem.values(), 1)
        }

        # get reals tokens
        series = series.apply(
            lambda x: [self.word2id[self.stem[word]]
                       for word in x
                       if word in self.stem]).sample(frac=1)

        # create training set
        x, y = [], []
        for doc in series:
            if self.maxlen is not None:
                doc = doc[:self.maxlen]
            n = len(doc) // 2
            x.append(doc[:n])
            y.append(doc[n:])
        x = pad_sequences(x)
        y = pad_sequences(y)
        x = np.concatenate([x, x[1:]])
        y = np.concatenate([y, y[:-1]])
        z = np.zeros((x.shape[0]))
        z[:series.shape[0]] = 1

        # shuffle data
        index = np.arange(x.shape[0])
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
        z = z[index]

        # create keras model
        model, encoder = create_model(
            self.n_features, self.n_heads, self.dim, self.hidden_dim)
        model.fit([x, y], z,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=self.validation_split)
        self.encoder = encoder
    
    def vectorize(self, series):
        # tokenize
        series = series.apply(self.tokenize)
        series = series.apply(
            lambda x: [self.word2id[self.stem[w]]
                       for w in x if w in self.stem])
        x = pad_sequences(series.tolist())
        y = self.encoder.predict(x)
    
    def save(self, filename):
        import dill
        self.weights = self.encoder.get_weights()
        del self.encoder
        with open(filename, "wb") as f:
            dill.dump(self, f)
    
    @staticmethod
    def load(filename):
        import dill
        with open(filename, "rb") as f:
            obj = dill.load(f)
        _, encoder = create_model(
            obj.n_features, obj.n_heads, obj.dim, obj.hidden_dim)
        encoder.set_weights(obj.weights)
        del obj.weights
        obj.encoder = encoder
        return obj
