

class Trigram:
    def __init__(self, token1:str, token2:str, token3:str, freq:int=0) -> None:
        self.token1 = token1
        self.token2 = token2
        self.token3 = token3
        self.frequency = freq
        pass

    def __repr__(self):
        return "{t1}_{t2}_{t3}({f})".format(t1=self.token1, t2=self.token2, t3=self.token3, f=self.frequency)

    def get(self):
        return "{t1}_{t2}_{t3}".format(t1=self.token1, t2=self.token2, t3=self.token3)