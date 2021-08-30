from app.domain.internal_data import Internal_Data

class DataLoader:
    def __init__(self):
        pass

    def load_text_file(self, text_file) -> Internal_Data:
        with open(text_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return Internal_Data(text_file, data)