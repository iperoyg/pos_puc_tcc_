from typing import List

class Bigram:
    def __init__(self, token1:str, token2:str, freq:int=0) -> None:
        self.token1 = token1
        self.token2 = token2
        self.frequency = freq
        pass

    def __repr__(self):
        return "{t1}_{t2}({f})".format(t1=self.token1, t2=self.token2, f=self.frequency)

    def get(self):
        return "{t1}_{t2}".format(t1=self.token1, t2=self.token2)

class BigramList:
    def __init__(self, bigrams:List[Bigram]) -> None:
        self.bigrams = bigrams
        pass
    
    def top_n(self, n:int=3) -> List[Bigram]:
        return self.bigrams[:n]