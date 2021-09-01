from app.data.data_handler import DataHandler
from app.service.analyser import Analyser

import pandas as pd
import argparse
import spacy
import nltk

class RunAppTest:
    def __init__(self) -> None:
        nltk.download('stopwords')
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
        pass
    
    def execute(self, text_file: str):
        dh = DataHandler(stopwords_list=self.stopwords)
        dh.receive_data(text_file)
        anl = Analyser()
        dh.bigrams = anl.find_bigrams(dh.data)
        dh.trigrams = anl.find_trigrams(dh.data, 6)
        dh.tfidf = anl.calculate_tdidf(dh.data)
        dh.pos = anl.calculate_top_postaggs(dh.data, "S", 4)

        #print(dh.trigrams)
        #print(pd.DataFrame.from_dict(dh.tfidf.idf, orient="index", columns=["Rank"]))

        print(dh.data.pruned_data)
        return "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file', type=str, help='input file name')
    args = parser.parse_args()

    print(RunAppTest().execute(args.input_file))