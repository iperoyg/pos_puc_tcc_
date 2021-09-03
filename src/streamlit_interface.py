from typing import List, Tuple
from app.domain.sentiment_data import Sentiment_Type
import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from app.data.data_handler import DataHandler
from app.service.analyser import Analyser

class UIData_TextInsights:
    def __init__(self) -> None:
        self.text_statistics = None
        self.sent_top_n = None
        self.ngram_top_n = None
        self.pos_top_n = None
        self.file_name = None
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

        self.tfidf_cluster_quality = None
        self.tfidf_cluster_top_words = None
        pass

def main():
    # Resolve dependencies

    st.title('My first app')
    st.markdown("""---""")

    # Run application
    run_app()

def hash_data_handler(data_handler):
    return (data_handler.file_name, data_handler.stopwords_list)

def hash_text_insights(data: UIData_TextInsights):
    return (data.sent_top_n, data.ngram_top_n, data.pos_top_n, data.file_name)

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

        dh.tfidf = anl.compute_tfidf(dh.data)

        dh.bigrams = anl.find_bigrams(dh.data)
        dh.trigrams = anl.find_trigrams(dh.data)
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
    
    report_selector = st.sidebar.selectbox("Choose report", ["Text Insights", "Word Insights"])
    sentiment_ranking_size = st.sidebar.slider('Sentiment Ranking Size', 5, 25, 5)
    ngrams_ranking_size = st.sidebar.slider('N-Grams Ranking Size', 2, 25, 5)
    postags_ranking_size = st.sidebar.slider('POS Taggins Ranking Size', 5, 50, 10)
    
    if report_selector == "Text Insights":
        dataui = get_text_insights(dh, sentiment_ranking_size, ngrams_ranking_size, postags_ranking_size)
        draw_text_insights(dataui)
    else:
        pass

def draw_text_insights(uidata_text_insighs:UIData_TextInsights) -> None:
    with st.container():
        st.write(uidata_text_insighs.text_statistics)
        st.markdown("""---""")
    
    with st.container():
        st.metric("Cluster Quality (0 is best)", uidata_text_insighs.tfidf_cluster_quality)
        st.dataframe(uidata_text_insighs.tfidf_cluster_top_words)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.table(uidata_text_insighs.positives)
        with col2:
            st.table(uidata_text_insighs.negatives)

        st.markdown("""---""")
        st.image(uidata_text_insighs.wordcloud.to_array())

    with st.container():
        st.markdown("""---""")
        col1, col2 = st.columns(2)
        with col1:
            st.image(uidata_text_insighs.bigrams_wordcloud.to_array())
            st.table(uidata_text_insighs.bigrams)

        with col2:
            st.image(uidata_text_insighs.trigrams_wordcloud.to_array())
            st.table(uidata_text_insighs.trigrams)

    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(uidata_text_insighs.top_verbs_wordcloud.to_array())
            st.table(uidata_text_insighs.top_verbs)

        with col2:
            st.image(uidata_text_insighs.top_nouns_wordcloud.to_array())
            st.table(uidata_text_insighs.top_nouns)
        
        with col3:
            st.image(uidata_text_insighs.top_adjs_wordcloud.to_array())
            st.table(uidata_text_insighs.top_adjs)


@st.cache(hash_funcs={UIData_TextInsights:hash_text_insights})
def get_text_insights(dh: DataHandler, sent_top_n:int=10, ngram_top_n:int = 5, pos_top_n:int=5) -> UIData_TextInsights:
    print("Insights - Cache Miss")
    
    anl = Analyser()
    top_positives = anl.get_top_sentiments(dh.sentiment, Sentiment_Type.POSITIVE, sent_top_n)
    top_negatives = anl.get_top_sentiments(dh.sentiment, Sentiment_Type.NEGATIVE, sent_top_n)

    dh.tfidf_clusters = anl.compute_tfidf_cluster(dh.data, dh.tfidf)

    top_verbs = dh.pos_verbs_ranking[:pos_top_n]
    top_nouns = dh.pos_nouns_ranking[:pos_top_n]
    top_adjs = dh.pos_adjs_ranking[:pos_top_n]

    uidata = UIData_TextInsights()
    uidata.sent_top_n = sent_top_n
    uidata.ngram_top_n = ngram_top_n
    uidata.pos_top_n = pos_top_n
    uidata.file_name = dh.file_name
    
    uidata.text_statistics = "This file has {nl} lines and {nw} words".format(nl = dh.line_count, nw=dh.word_count)

    uidata.tfidf_cluster_quality = "{x:.2f}".format(x=dh.tfidf_clusters.inertia)
    uidata.tfidf_cluster_top_words = pd.DataFrame([(x, ",".join(dh.tfidf_clusters.top_words_clusters[x])) for x in dh.tfidf_clusters.top_words_clusters], columns=("Cluster", "Top Words"))

    uidata.idf_ranking = pd.DataFrame.from_dict(dh.tfidf.idf, orient="index", columns=["Rank"])
    uidata.idf_ranking = uidata.idf_ranking.sort_values(by=["Rank"], ascending=False).head(sent_top_n)
    uidata.positives = pd.DataFrame([(i.text, i.weight) for i in top_positives], columns=("Positive Phrase", "Score"))
    uidata.negatives = pd.DataFrame([(i.text, i.weight) for i in top_negatives], columns=("Negative Phrase", "Score"))

    uidata.wordcloud = WordCloud().generate(dh.get_plain_text(pruned=True))

    uidata.bigrams = pd.DataFrame([(i.get(), i.frequency) for i in dh.bigrams[:ngram_top_n]], columns=("Bigram", "Frequency"))
    uidata.bigrams_wordcloud = WordCloud().generate_from_frequencies({i.get(): i.frequency for i in dh.bigrams[:ngram_top_n]}) # if i.frequency > 0
    uidata.trigrams = pd.DataFrame([(i.get(), i.frequency) for i in dh.trigrams[:ngram_top_n]], columns=("Trigram", "Frequency"))
    uidata.trigrams_wordcloud = WordCloud().generate_from_frequencies({i.get(): i.frequency for i in dh.trigrams[:ngram_top_n]})

    uidata.top_verbs = pd.DataFrame([(i.token.raw, i.frequency) for i in top_verbs], columns=("Verbs", "Frequency"))
    uidata.top_verbs_wordcloud = WordCloud().generate_from_frequencies({i.token.raw : i.frequency for i in top_verbs})
    uidata.top_nouns = pd.DataFrame([(i.token.raw, i.frequency) for i in top_nouns], columns=("Nouns", "Frequency"))
    uidata.top_nouns_wordcloud = WordCloud().generate_from_frequencies({i.token.raw : i.frequency for i in top_nouns})
    uidata.top_adjs = pd.DataFrame([(i.token.raw, i.frequency) for i in top_adjs], columns=("Adjs", "Frequency"))
    uidata.top_adjs_wordcloud = WordCloud().generate_from_frequencies({i.token.raw : i.frequency for i in top_adjs})

    return uidata

if __name__ == "__main__":
    main()