from app.data.data_handler import DataHandler
from app.service.analyser import Analyser
import pandas as pd
import argparse

class RunAppTest:
    def __init__(self) -> None:
        pass
    
    def execute(self, text_file: str):
        dh = DataHandler()
        dh.receive_data(text_file)
        anl = Analyser()
        dh.bigrams = anl.find_bigrams(dh.data)
        dh.tfidf = anl.calculate_tdidf(dh.data)

        #tb = pd.DataFrame.from_dict(dh.tfidf.idf, columns=("Term", "Frequency"))
        print(pd.DataFrame.from_dict(dh.tfidf.idf, orient="index", columns=["Rank"]))

        return "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file', type=str, help='input file name')
    args = parser.parse_args()

    print(RunAppTest().execute(args.input_file))