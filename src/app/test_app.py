from app.data.data_handler import DataHandler

class RunAppTest:
    def __init__(self) -> None:
        pass
    
    def execute(self, text_file: str):
        dh = DataHandler()
        dh.receive_data(text_file)
        return "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)

if __name__ == "__main__":
    RunAppTest().execute("tmp/data1.txt")