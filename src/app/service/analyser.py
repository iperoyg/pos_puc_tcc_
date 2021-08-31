from typing import List
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.domain.bigram import Bigram
from app.domain.internal_data import Internal_Data
from app.domain.tfidf_data import TfIdf_Data

class Analyser:
    def __init__(self) -> None:
        pass

    def find_bigrams(self, input_data: Internal_Data, top_n:int=3) -> List[Bigram]:
        bigrams = self.__find_bigrams(input_data)
        bigrams_counter = Counter([b.get() for b in bigrams])
        top = bigrams_counter.most_common(top_n)
        return [Bigram(token1=tp[0].split('_')[0], token2=tp[0].split('_')[1], freq=tp[1]) for tp in top]

    def calculate_tdidf(self, input_data:Internal_Data) -> TfIdf_Data:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(input_data.get())
        idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tfidf = dict(zip(vectorizer.get_feature_names(), vectors.todense().tolist()))
        return TfIdf_Data(tfidf, idf)

    def __find_bigrams(self, input_data: Internal_Data) -> List[Bigram]:
        data = input_data.get()
        bigrams = list()
        for line in data:
            tokens = line.split(' ')
            for i in range(len(tokens)-1):
                t1 = tokens[i]
                t2 = tokens[i+1]
                bigram = Bigram(t1,t2)
                bigrams.append(bigram)
        return bigrams
