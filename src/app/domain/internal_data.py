from typing import List

class Internal_Data:
    def __init__(self, name, data) -> None:
        self.name = name
        self.raw_data : List[str] = data
        self.pp_data : List[str] = None

    def add_preprocessed_data(self, pp_data):
        self.pp_data = pp_data
    
    def get(self):
        if self.pp_data is None:
            return self.raw_data
        return self.pp_data