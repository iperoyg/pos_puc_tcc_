from typing import List
from app.domain.internal_data import *

class DataPreprocessor:
    def __init__(self) -> None:
        pass

    def preprocess(self, internal_data: Internal_Data) -> Internal_Data:
        raw_data = internal_data.raw_data
        pp_data = self.__pp_tolowercase(raw_data)
        internal_data.add_preprocessed_data(pp_data)
        return internal_data

    def __pp_tolowercase(self, data: List[str]) -> List[str]:
        return [l.lower() for l in data]
