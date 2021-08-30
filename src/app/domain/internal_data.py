from _typeshed import SupportsReadline


class Internal_Data:
    def __init__(self, name, data) -> None:
        self.name = name
        self.raw_data = data

    def add_preprocessed_data(self, pp_data):
        self.pp_data = pp_data