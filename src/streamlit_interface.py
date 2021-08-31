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
        pass

def execute(text_file: str) -> LocalItemData:
    dh = DataHandler()
    dh.receive_data(text_file)
    anl = Analyser()
    dh.bigrams = anl.find_bigrams(dh.data)
    
    response = "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)
    wordcloud = WordCloud().generate(dh.get_plain_text())

    df = pd.DataFrame([(i.get(), i.frequency) for i in dh.bigrams], columns=("Bigram", "Frequency"))

    report = df

    return LocalItemData(response, wordcloud, report)

st.title('My first app')

file_location = st.text_input("Type file location")
if not file_location:
    st.warning('Please input a full file name location.')
    st.stop()

if st.button('Analyse file'):
    local_execute = execute(file_location)
    st.write(local_execute.response)

    st.text("Wordcloud")
    fig, ax = plt.subplots()
    plt.imshow(local_execute.wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

    st.table(local_execute.report)
else:
    pass

