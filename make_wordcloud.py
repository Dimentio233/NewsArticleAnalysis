"""
Brandon Wong and Johnny He
CSE 163

This file contains functions that create a
wordcloud image for each respective news source
"""
from wordcloud import WordCloud, STOPWORDS


def fox_wordcloud():
    """
    This function saves a fox wordcloud image to
    files
    """
    text = open('all_text/fox_all.txt', mode='r', encoding='utf-8').read()
    stopwords = list(STOPWORDS) + ['S', 'provided', 'FOX News', 'real  time',
                                   'Implemented', 'Said', 'FOX', 'News',
                                   'FactSet Digital', 'rights reserved',
                                   'reserved Quotes', 'Network LLC']
    wc = WordCloud(
        background_color='white',
        stopwords=stopwords,
        height=600,
        width=400
    )
    wc.generate(text)
    wc.to_file('cloud_fox.png')


def cnn_wordcloud():
    """
    This function saves a cnn wordcloud image to
    files
    """
    text = open('all_text/cnn_all.txt', mode='r', encoding='utf-8').read()
    stopwords = list(STOPWORDS) + ["will", "said"]
    wc = WordCloud(
        background_color='white',
        stopwords=stopwords,
        height=600,
        width=400
    )
    wc.generate(text)
    wc.to_file('cloud_cnn.png')


def main():
    fox_wordcloud()
    cnn_wordcloud()


if __name__ == "__main__":
    main()
