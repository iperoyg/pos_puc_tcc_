import spacy
import pandas as pd

from typing import List
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


from app.domain.unigram import Unigram
from app.domain.bigram import Bigram, BigramList
from app.domain.trigram import Trigram
from app.domain.internal_data import Internal_Data
from app.domain.token import Token
from app.domain.tfidf_data import TfIdf_Data
from app.domain.sentiment_data import Sentiment_Data, Sentiment_Type
from app.LeIA.leia import SentimentIntensityAnalyzer

class Analyser:
    def __init__(self) -> None:
        self.spacy = spacy.load("pt_core_news_sm")
        self.leia = SentimentIntensityAnalyzer()
        pass
    
    def find_bigrams(self, input_data: Internal_Data) -> List[Bigram]:
        bigrams = self.__find_bigrams(input_data)
        return sorted(bigrams, key=lambda x : x.frequency, reverse=True)

    def find_trigrams(self, input_data: Internal_Data) -> List[Trigram]:
        bigrams = self.__find_trigrams(input_data)
        return sorted(bigrams, key=lambda x : x.frequency, reverse=True)

    def compute_tfidf(self, input_data:Internal_Data) -> TfIdf_Data:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(input_data.get(pruned=True))
        idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tfidf_flat = dict(zip(vectorizer.get_feature_names(), vectors.todense().tolist()))
        return TfIdf_Data(vectors, tfidf_flat, idf)

    def compute_tfidf_cluster(self, input_data:Internal_Data, tfidf_data:TfIdf_Data, n_clusters:int=3) -> List:
        kmeans = KMeans(n_clusters=n_clusters).fit(tfidf_data.tfidf)
        result = kmeans.predict(tfidf_data.tfidf)
        #print(type(result), len(result))
        #print(type(input_data.get(True)), len(input_data.get(True)))
        return list(zip(result, input_data.get(True)))

    def calculate_postaggs_ranking(self, pos_data:List[List[Token]], pos_type:str) -> List[Unigram]:
        '''
        POS Taggs (SpaCy)
        Verbo = VERB & AUX = 'V'
        Substantivo = NOUN & PROPN = 'S'
        Adjetivo = ADJ = 'A'
        '''
        pos_filter = ['VERB']
        pos_type_internal = ""
        if pos_type == "V":
            pos_filter = ['VERB']
            pos_type_internal = "VERBO"
            #pos_filter = ['VERB', 'AUX']
        elif pos_type == "S":
            pos_filter = ['NOUN', 'PROPN']
            pos_type_internal = "SUBSTANTIVO"
        elif pos_type == "A":
            pos_filter = ['ADJ']
            pos_type_internal = "ADJETIVO"
        else:
            raise Exception("Wrong pos_type")

        pos = pos_data
        flat_list = [item for sublist in pos for item in sublist]
        flat_list = [item for item in flat_list if item.pos in pos_filter]
        counter = dict()
        for item in flat_list:
            item_key = item.raw
            if item_key not in counter:
                counter[item_key] = 0
            counter[item_key] +=1
        return [Unigram(Token(k,pos_type_internal), v) for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)]

    def calculate_sentiment(self, input_data: Internal_Data) -> List[Sentiment_Data]:
        return [Sentiment_Data(text=s,polarity_score=self.leia.polarity_scores(s)) for s in input_data.get()]

    def get_top_sentiments(self, data: List[Sentiment_Data], sentiment:Sentiment_Type, top_n:int=5):
        filtered_sentiments = [i for i in data if i.polarity == sentiment]
        sorted_sentiments = sorted(filtered_sentiments, key=lambda x: x.weight, reverse=True)
        top = sorted_sentiments[:top_n]
        return top

    def define_postaggs(self, input_data: Internal_Data) -> List[List[Token]]:
        return [[Token(t.text, t.pos_) for t in self.spacy(s)] for s in input_data.get()]

    def __find_bigrams(self, input_data: Internal_Data) -> List[Bigram]:
        data = input_data.get(pruned=True)
        bigrams : dict[Bigram] = dict()
        for line in data:
            tokens = line.split(' ')
            for i in range(len(tokens)-1):
                t1 = tokens[i]
                t2 = tokens[i+1]
                bigram = Bigram(t1,t2,0)
                if bigram.get() not in bigrams:
                    bigrams[bigram.get()] = bigram
                bigrams[bigram.get()].frequency += 1
        return [v for _, v in bigrams.items()] 

    def __find_trigrams(self, input_data: Internal_Data) -> List[Trigram]:
        data = input_data.get(pruned=True)
        trigrams : dict[Trigram] = dict()
        for line in data:
            tokens = line.split(' ')
            for i in range(len(tokens)-2):
                t1 = tokens[i]
                t2 = tokens[i+1]
                t3 = tokens[i+2]
                trigram = Trigram(t1,t2,t3,0)
                if trigram.get() not in trigrams:
                    trigrams[trigram.get()] = trigram
                trigrams[trigram.get()].frequency += 1
        return [v for _, v in trigrams.items()] 
