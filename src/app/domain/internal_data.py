from typing import List

from app.domain.token import Token

class Internal_Data:
    def __init__(self, name, data) -> None:
        self.name = name
        self.raw_data : List[str] = data
        self.pp_data : List[str] = None
        self.pruned_data : List[str] = None #without stopwords and punctuation

    def add_preprocessed_data(self, pp_data):
        self.pp_data = pp_data

    def add_pruned_data(self, data):
        self.pruned_data = data

    def get(self, pruned=False):
        if pruned and self.pruned_data is not None:
            return self.pruned_data
        if self.pp_data is None:
            return self.raw_data
        return self.pp_data
    
    def get_text(self, pruned=False) -> str:
        return " ".join(self.get(pruned))