import pandas as pd
from pandas.tseries import frequencies
from app.domain.bigram import Bigram
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

def execute(text_file: str, idf_ranking_size:int=10, ngrams_n:int = 5, top_pos_n:int=5) -> LocalItemData:
    dh = DataHandler()
    dh.receive_data(text_file)
    anl = Analyser()
    dh.bigrams = anl.find_bigrams(dh.data, ngrams_n)
    dh.trigrams = anl.find_trigrams(dh.data, ngrams_n)
    dh.tfidf = anl.calculate_tdidf(dh.data)

    top_verbs = anl.calculate_top_postaggs(dh.data, "V", top_pos_n)
    top_nouns = anl.calculate_top_postaggs(dh.data, "S", top_pos_n)
    top_adjs = anl.calculate_top_postaggs(dh.data, "A", top_pos_n)

    lid = LocalItemData()

    lid.text_statistics = "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)

    lid.idf_ranking = pd.DataFrame.from_dict(dh.tfidf.idf, orient="index", columns=["Rank"])
    lid.idf_ranking = lid.idf_ranking.sort_values(by=["Rank"], ascending=False).head(idf_ranking_size)

    lid.wordcloud = WordCloud().generate(dh.get_plain_text())

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

st.title('My first app')
st.markdown("""---""")

file_location = st.sidebar.text_input("Type file location")
report_selector = st.sidebar.selectbox("Choose report", ["Simple data", "Word focus"])
idf_ranking_size = st.sidebar.slider('Inverse Doc Frequency Ranking Size', 5, 25, 10)
ngrams_ranking_size = st.sidebar.slider('N-Grams Ranking Size', 2, 15, 10)
postags_ranking_size = st.sidebar.slider('POS Taggins Ranking Size', 5, 25, 10)
submmit_button = st.sidebar.button('Analyse file')

if not file_location:
    st.warning('Please input a full file name location.')
    st.stop()

if submmit_button and report_selector == "Simple data":
    local_execute = execute(file_location, idf_ranking_size, ngrams_ranking_size, postags_ranking_size)

    with st.container():
        st.markdown("""---""")
        st.write(local_execute.text_statistics)
        st.table(local_execute.idf_ranking)
        st.markdown("""---""")
        fig, ax = plt.subplots()
        plt.imshow(local_execute.wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)

    with st.container():
        st.markdown("""---""")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.bigrams_wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)

            st.table(local_execute.bigrams)

        with col2:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.trigrams_wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)

            st.table(local_execute.trigrams)

    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.top_verbs_wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)
            st.table(local_execute.top_verbs)

        with col2:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.top_nouns_wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)
            st.table(local_execute.top_nouns)
        
        with col3:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.top_adjs_wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)
            st.table(local_execute.top_adjs)
else:
    pass

