from app.domain.token import Token

class Unigram:
    def __init__(self, token:Token, frequency:int) -> None:
        self.token = token
        self.frequency = frequency
        
    def __repr__(self) -> str:
        return "{token}:{f}".format(token=self.token, f=self.frequency)