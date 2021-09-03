from enum import Enum

class Sentiment_Type(Enum):
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1

class Sentiment_Data:
    def __init__(self, text:str, polarity_score:dict ) -> None:
        self.text = text
        self.polarity_dict = polarity_score
        self.polarity : Sentiment_Type = None
        self.weight : float = 0
        self.__calculate_polarity()
    
    def __calculate_polarity(self) -> None:
        neg = self.polarity_dict['neg']
        neu = self.polarity_dict['neu']
        pos = self.polarity_dict['pos']
        if neg >= neu and neg >= pos:
            self.polarity = Sentiment_Type.NEGATIVE
            self.weight = neg
        elif neu >= pos:
            self.polarity = Sentiment_Type.NEUTRAL
            self.weight = neu
        else:
            self.polarity = Sentiment_Type.POSITIVE
            self.weight = pos
    
    def __repr__(self) -> str:
        return "{s}:{t}".format(s=self.polarity, t=self.text)

