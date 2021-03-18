import pandas as pd
from embedder import Embedder

df = pd.read_pickle("news_ecolo.p")

embedder = Embedder(
    min_tf=5, dim=64, hidden_dim=32, n_heads=4,
    maxlen=600, epochs=10)
embedder.fit(df.content)
embedder.save("test.p")

embedder = Embedder.load("test.p")
embedder.vectorize(df.title)
