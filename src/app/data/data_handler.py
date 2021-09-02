from typing import List

from app.data.data_loader import DataLoader
from app.data.data_preprocessor import DataPreprocessor
from app.domain.unigram import Unigram
from app.domain.bigram import Bigram, BigramList
from app.domain.sentiment_data import Sentiment_Data
from app.domain.trigram import Trigram
from app.domain.internal_data import Internal_Data
from app.domain.tfidf_data import TfIdf_Data
from app.domain.token import Token



class DataHandler:
    def __init__(self, stopwords_list) -> None:
        self.file_name:str = None
        self.stopwords = stopwords_list
        self.dl = DataLoader()
        self.dpp = DataPreprocessor(stopwords_list)
        self.data : Internal_Data = None
        self.line_count = 0
        self.word_count = 0
        self.bigrams : List[Bigram] = None
        self.trigrams : List[Trigram] = None
        self.tfidf : TfIdf_Data = None
        self.tfidf_clusters = None
        self.pos : List[List[Token]] = None
        self.pos_verbs_ranking : List[Unigram]= None
        self.pos_nouns_ranking : List[Unigram]= None
        self.pos_adjs_ranking : List[Unigram]= None
        self.sentiment : List[Sentiment_Data] = None
        pass
    
    def receive_data(self, file: str) -> Internal_Data:
        internal_data = self.dl.load_text_file(file)
        internal_data = self.dpp.preprocess(internal_data)
        self.data = internal_data
        self.__report()
        self.file_name = file
        return self.data

    def get_plain_text(self, pruned=False) -> str:
        return self.data.get_text(pruned)

    def __report(self):
        self.line_count = len(self.data.pp_data)
        self.word_count = sum(len(x.split(' ')) for x in self.data.pp_data)
        