import spacy
import pandas as pd

from typing import List
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    def find_all_bigrams(self, input_data: Internal_Data) -> BigramList:
        bigrams = self.__find_bigrams(input_data)
        bigrams = sorted(bigrams, key=lambda x : x.frequency, reverse=True)
        return BigramList(bigrams)

    def find_bigrams(self, input_data: Internal_Data, top_n:int=3) -> List[Bigram]:
        bigrams = self.__find_bigrams(input_data)
        bigrams_counter = Counter([b.get() for b in bigrams])
        top = bigrams_counter.most_common(top_n)
        return [Bigram(token1=tp[0].split('_')[0], token2=tp[0].split('_')[1], freq=tp[1]) for tp in top]

    def find_trigrams(self, input_data: Internal_Data, top_n:int=3) -> List[Trigram]:
        trigrams = self.__find_trigrams(input_data)
        trigrams_counter = Counter([b.get() for b in trigrams])
        top = trigrams_counter.most_common(top_n)
        return [Trigram(token1=tp[0].split('_')[0], token2=tp[0].split('_')[1], token3=tp[0].split('_')[2], freq=tp[1]) for tp in top]

    def calculate_tfidf(self, input_data:Internal_Data) -> TfIdf_Data:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(input_data.get(pruned=True))
        idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tfidf = dict(zip(vectorizer.get_feature_names(), vectors.todense().tolist()))
        return TfIdf_Data(tfidf, idf)

    def calculate_top_postaggs(self, input_data:Internal_Data, pos_type:str, top_n:int=5) -> List[Unigram]:
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

        pos = self.__define_postaggs(input_data)
        flat_list = [item for sublist in pos for item in sublist]
        flat_list = [item for item in flat_list if item.pos in pos_filter]
        counter = dict()
        for item in flat_list:
            item_key = item.raw
            if item_key not in counter:
                counter[item_key] = 0
            counter[item_key] +=1
        top_postaggs = [Unigram(Token(k,pos_type_internal), v) for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)[:top_n]]
        return top_postaggs

    def calculate_sentiment(self, input_data: Internal_Data) -> List[Sentiment_Data]:
        return [Sentiment_Data(text=s,polarity_score=self.leia.polarity_scores(s)) for s in input_data.get()]

    def get_top_sentiments(self, data: List[Sentiment_Data], sentiment:Sentiment_Type, top_n:int=5):
        filtered_sentiments = [i for i in data if i.polarity == sentiment]
        sorted_sentiments = sorted(filtered_sentiments, key=lambda x: x.weight, reverse=True)
        top = sorted_sentiments[:top_n]
        return top

    def __define_postaggs(self, input_data: Internal_Data) -> List[List[Token]]:
        return [[Token(t.text, t.pos_) for t in self.spacy(s)] for s in input_data.get()]

    def __find_bigrams(self, input_data: Internal_Data) -> List[Bigram]:
        data = input_data.get(pruned=True)
        bigrams = list()
        for line in data:
            tokens = line.split(' ')
            for i in range(len(tokens)-1):
                t1 = tokens[i]
                t2 = tokens[i+1]
                bigram = Bigram(t1,t2)
                bigrams.append(bigram)
        return bigrams

    def __find_trigrams(self, input_data: Internal_Data) -> List[Trigram]:
        data = input_data.get(pruned=True)
        trigrams = list()
        for line in data:
            tokens = line.split(' ')
            for i in range(len(tokens)-2):
                t1 = tokens[i]
                t2 = tokens[i+1]
                t3 = tokens[i+2]
                trigram = Trigram(t1,t2,t3)
                trigrams.append(trigram)
        return trigrams
