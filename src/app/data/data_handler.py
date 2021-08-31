from typing import List

from app.data.data_loader import DataLoader
from app.data.data_preprocessor import DataPreprocessor
from app.domain.bigram import Bigram
from app.domain.trigram import Trigram
from app.domain.internal_data import Internal_Data
from app.domain.tfidf_data import TfIdf_Data
from app.domain.token import Token


class DataHandler:
    def __init__(self) -> None:
        self.dl = DataLoader()
        self.dpp = DataPreprocessor()
        self.data : Internal_Data = None
        self.line_count = 0
        self.word_count = 0
        self.bigrams : List[Bigram] = None
        self.trigrams : List[Trigram] = None
        self.tfidf : TfIdf_Data = None
        self.pos : List[List[Token]] = None
        pass
    
    def receive_data(self, file: str) -> Internal_Data:
        internal_data = self.dl.load_text_file(file)
        internal_data = self.dpp.preprocess(internal_data)
        self.data = internal_data
        self.__report()
        return self.data

    def get_plain_text(self) -> str:
        return self.data.get_text()

    def __report(self):
        self.line_count = len(self.data.pp_data)
        self.word_count = sum(len(x.split(' ')) for x in self.data.pp_data)
        