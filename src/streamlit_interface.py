import pandas as pd
from pandas.tseries import frequencies
from app.domain.bigram import Bigram
import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud


from app.data.data_handler import DataHandler
from app.service.analyser import Analyser

class LocalItemData:
    def __init__(self, response, wc, report) -> None:
        self.response = response
        self.wordcloud = wc
        self.report = report
        self.idf_ranking = None
        pass

def execute(text_file: str, idf_ranking_size:int=10) -> LocalItemData:
    dh = DataHandler()
    dh.receive_data(text_file)
    anl = Analyser()
    dh.bigrams = anl.find_bigrams(dh.data)
    dh.tfidf = anl.calculate_tdidf(dh.data)

    response = "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)
    wordcloud = WordCloud().generate(dh.get_plain_text())

    df = pd.DataFrame([(i.get(), i.frequency) for i in dh.bigrams], columns=("Bigram", "Frequency"))

    report = df

    lid = LocalItemData(response, wordcloud, report)
    lid.idf_ranking = pd.DataFrame.from_dict(dh.tfidf.idf, orient="index", columns=["Rank"])
    lid.idf_ranking = lid.idf_ranking.sort_values(by=["Rank"], ascending=False).head(idf_ranking_size)

    return lid

st.title('My first app')
st.markdown("""---""")

file_location = st.sidebar.text_input("Type file location")
report_selector = st.sidebar.selectbox("Choose report", ["Simple data", "Word focus"])
idf_ranking_size = st.sidebar.slider('Inverse Doc Frequency Ranking Size', 5, 25, 10)
submmit_button = st.sidebar.button('Analyse file')

if not file_location:
    st.warning('Please input a full file name location.')
    st.stop()

if submmit_button and report_selector == "Simple data":
    local_execute = execute(file_location, idf_ranking_size)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.table(local_execute.idf_ranking)

        with col2:
            st.write(local_execute.response)
            st.table(local_execute.report)

    with st.container():
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            plt.imshow(local_execute.wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)

else:
    pass

