from typing import List, Tuple
from app.domain.sentiment_data import Sentiment_Type
import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from app.data.data_handler import DataHandler
from app.service.analyser import Analyser

class LocalItemData:
    def __init__(self) -> None:
        self.text_statistics = None
        self.wordcloud = None
        self.idf_ranking = None
        self.positives = None
        self.negatives = None
        self.bigrams = None
        self.bigrams_wordcloud = None
        self.trigrams = None
        self.trigrams_wordcloud = None
        self.top_verbs = None
        self.top_verbs_wordcloud = None
        self.top_nouns = None
        self.top_nouns_wordcloud = None
        self.top_adjs = None
        self.top_adjs_wordcloud = None        
        pass

def main():
    # Resolve dependencies

    st.title('My first app')
    st.markdown("""---""")

    # Run application
    run_app()

def hash_data_handler(data_handler):
    return (data_handler.file_name, data_handler.stopwords_list)

def run_app():

    @st.cache
    def get_stopwords() -> List[str]:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('portuguese')
        return stopwords
    
    @st.cache(allow_output_mutation=True, hash_funcs={DataHandler:hash_data_handler})
    def get_data(text_file: str) -> DataHandler:
        print("Cache miss")
        anl = Analyser()
        dh = DataHandler(stopwords_list=get_stopwords())
        dh.receive_data(text_file)
        dh.bigrams = anl.find_bigrams(dh.data)
        dh.trigrams = anl.find_trigrams(dh.data)
        dh.tfidf = anl.calculate_tfidf(dh.data)
        dh.pos = anl.define_postaggs(dh.data)
        dh.pos_verbs_ranking = anl.calculate_postaggs_ranking(dh.pos, "V")
        dh.pos_nouns_ranking = anl.calculate_postaggs_ranking(dh.pos, "S")
        dh.pos_adjs_ranking = anl.calculate_postaggs_ranking(dh.pos, "A")
        dh.sentiment = anl.calculate_sentiment(dh.data)
        return dh

    file_location = st.sidebar.text_input("Type file location")
    if not file_location:
        #TODO: Check files
        st.warning('Please input a full file name location.')
        st.stop()
    
    dh = get_data(file_location)
    anl = Analyser()
    
    report_selector = st.sidebar.selectbox("Choose report", ["Simple data", "Word focus"])
    idf_ranking_size = st.sidebar.slider('Inverse Doc Frequency Ranking Size', 5, 25, 10)
    ngrams_ranking_size = st.sidebar.slider('N-Grams Ranking Size', 2, 15, 10)
    postags_ranking_size = st.sidebar.slider('POS Taggins Ranking Size', 5, 25, 10)
    submmit_button = st.sidebar.button('Analyse file')

    if not file_location:
        st.warning('Please input a full file name location.')
        st.stop()

    if submmit_button and report_selector == "Simple data":
        local_execute = execute(dh, anl, file_location, idf_ranking_size, ngrams_ranking_size, postags_ranking_size)

        with st.container():
            st.markdown("""---""")
            st.write(local_execute.text_statistics)
            st.markdown("""---""")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.table(local_execute.idf_ranking)
            with col2:
                st.table(local_execute.positives)
            with col3:
                st.table(local_execute.negatives)

            st.markdown("""---""")
            st.image(local_execute.wordcloud.to_array())

        with st.container():
            st.markdown("""---""")
            col1, col2 = st.columns(2)
            with col1:
                st.image(local_execute.bigrams_wordcloud.to_array())
                st.table(local_execute.bigrams)

            with col2:
                st.image(local_execute.trigrams_wordcloud.to_array())
                st.table(local_execute.trigrams)

        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(local_execute.top_verbs_wordcloud.to_array())
                st.table(local_execute.top_verbs)

            with col2:
                st.image(local_execute.top_nouns_wordcloud.to_array())
                st.table(local_execute.top_nouns)
            
            with col3:
                st.image(local_execute.top_adjs_wordcloud.to_array())
                st.table(local_execute.top_adjs)
    else:
        pass

def execute(dh: DataHandler, anl : Analyser, text_file: str, idf_ranking_size:int=10, ngrams_n:int = 5, top_pos_n:int=5) -> LocalItemData:
    
    top_positives = anl.get_top_sentiments(dh.sentiment, Sentiment_Type.POSITIVE, idf_ranking_size)
    top_negatives = anl.get_top_sentiments(dh.sentiment, Sentiment_Type.NEGATIVE, idf_ranking_size)

    top_verbs = dh.pos_verbs_ranking[:top_pos_n]
    top_nouns = dh.pos_nouns_ranking[:top_pos_n]
    top_adjs = dh.pos_adjs_ranking[:top_pos_n]

    lid = LocalItemData()

    lid.text_statistics = "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)

    lid.idf_ranking = pd.DataFrame.from_dict(dh.tfidf.idf, orient="index", columns=["Rank"])
    lid.idf_ranking = lid.idf_ranking.sort_values(by=["Rank"], ascending=False).head(idf_ranking_size)
    lid.positives = pd.DataFrame([(i.text, i.weight) for i in top_positives], columns=("Positive Phrase", "Score"))
    lid.negatives = pd.DataFrame([(i.text, i.weight) for i in top_negatives], columns=("Negative Phrase", "Score"))

    lid.wordcloud = WordCloud().generate(dh.get_plain_text(pruned=True))

    lid.bigrams = pd.DataFrame([(i.get(), i.frequency) for i in dh.bigrams], columns=("Bigram", "Frequency"))
    lid.bigrams_wordcloud = WordCloud().generate_from_frequencies({i.get(): i.frequency for i in dh.bigrams})
    lid.trigrams = pd.DataFrame([(i.get(), i.frequency) for i in dh.trigrams], columns=("Trigram", "Frequency"))
    lid.trigrams_wordcloud = WordCloud().generate_from_frequencies({i.get(): i.frequency for i in dh.trigrams})

    lid.top_verbs = pd.DataFrame([(i.token.raw, i.frequency) for i in top_verbs], columns=("Verbs", "Frequency"))
    lid.top_verbs_wordcloud = WordCloud().generate_from_frequencies({i.token.raw : i.frequency for i in top_verbs})
    lid.top_nouns = pd.DataFrame([(i.token.raw, i.frequency) for i in top_nouns], columns=("Nouns", "Frequency"))
    lid.top_nouns_wordcloud = WordCloud().generate_from_frequencies({i.token.raw : i.frequency for i in top_nouns})
    lid.top_adjs = pd.DataFrame([(i.token.raw, i.frequency) for i in top_adjs], columns=("Adjs", "Frequency"))
    lid.top_adjs_wordcloud = WordCloud().generate_from_frequencies({i.token.raw : i.frequency for i in top_adjs})

    return lid

if __name__ == "__main__":
    main()