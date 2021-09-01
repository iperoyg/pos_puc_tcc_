import string
from typing import List

from app.domain.internal_data import *

class DataPreprocessor:
    def __init__(self, stopwords_list) -> None:
        self.stopwords_list = stopwords_list
        pass

    def preprocess(self, internal_data: Internal_Data) -> Internal_Data:
        raw_data = internal_data.raw_data
        pp_data = self.__pp_tolowercase(raw_data)
        pp_data = self.__pp_remove_empty(pp_data)
        pp_data = self.__pp_strip(pp_data)
        internal_data.add_preprocessed_data(pp_data)
        pruned_data = self.__pp_remove_punct(pp_data)
        pruned_data = self.__pp_remove_sw(pruned_data)
        pruned_data = self.__pp_remove_empty(pruned_data)
        internal_data.add_pruned_data(pruned_data)
        return internal_data

    def __pp_tolowercase(self, data: List[str]) -> List[str]:
        return [l.lower() for l in data]

    def __pp_remove_empty(self, data:List[str]) -> List[str]:
        return [l for l in data if len(l.strip()) > 0]

    def __pp_strip(self, data:List[str]) -> List[str]:
        return [l.strip() for l in data]
    
    def __pp_remove_punct(self,  data:List[str]) -> List[str]:
        return [l.translate(str.maketrans('', '', string.punctuation)) for l in data]

    def __pp_remove_sw(self, data:List[str]) -> List[str]:
        return [" ".join([w for w in l.split(' ') if w not in self.stopwords_list]) for l in data]