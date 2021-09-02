class TfIdf_Data:
    def __init__(self, tfidf, tfidf_flat, idf_dict) -> None:
        self.tfidf = tfidf
        self.tfidf_flat = tfidf_flat
        self.idf = idf_dict