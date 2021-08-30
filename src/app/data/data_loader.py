from app.domain.internal_data import *

class DataLoader:
    def __init__(self):
        pass

    def load_text_file(self, text_file) -> Internal_Data:
        with open(text_file, 'r') as f:
            data = f.readlines()
        return Internal_Data(text_file, data)