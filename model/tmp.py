import pickle
from tsv_data_loader import TsvDataLoader

v = pickle.load(open('../vocabularies/vocab_news_emb.pickle', 'rb'))
t = TsvDataLoader('../testdata/Custom_texts.tsv', v)
g = t.load_lazy()

print(next(g))