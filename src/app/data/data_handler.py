from app.data.data_loader import DataLoader
from app.data.data_preprocessor import DataPreprocessor
from app.domain.internal_data import Internal_Data

class DataHandler:
    def __init__(self) -> None:
        self.dl = DataLoader()
        self.dpp = DataPreprocessor()
        self.data : Internal_Data = None
        self.line_count = 0
        self.word_count = 0
        pass
    
    def receive_data(self, file: str) -> Internal_Data:
        internal_data = self.dl.load_text_file(file)
        internal_data = self.dpp.preprocess(internal_data)
        self.data = internal_data
        self.__report()
        return self.data

    def __report(self):
        self.line_count = len(self.data.pp_data)
        self.word_count = sum(len(x.split(' ')) for x in self.data.pp_data)
        