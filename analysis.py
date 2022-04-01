"""
Brandon Wong and Johnny He
CSE 163

This file contains functions to reformat our combined CSV of
Fox News and CNN articles to be used to train and create our
Machine Learning Model.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import statistics as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from newspaper import Article
from textblob import TextBlob
sns.set()
nltk.download('stopwords')
nltk.download('punkt')


def csv_processing(df):
    """
    Returns a new dataframe with added columns news length, parsed content,
    and category target. these columns will make training the model later
    easier
    """
    df = df.dropna()
    sns.countplot(df.news_source)
    plt.savefig("my_plot.png")
    df["news_length"] = df["article_content"].str.len()
    df['parsed_content'] = df['article_content'].apply(process_text)
    label_encoder = preprocessing.LabelEncoder()
    df['Category_Target'] = label_encoder.fit_transform(df['news_source'])
    df.to_csv('Combined_Preprocess_Ult.csv')
    return df


def process_text(text):
    """
    This function parses through each article's content column
    and gets rid of any characters other than words. This is then
    stored in a new column named 'parsed content'
    """
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    text = ' '.join(filtered_sentence)
    return text


def generate_summary(data):
    """
    returns a dataframe with an added summary column
    that formats each article's content
    """
    summary_list = []
    sentiment_list = []
    for link in data['link']:
        article = Article(link)
        article.download()
        article.parse()
        article.nlp()
        summary_list.append(article.summary)
        analysis = TextBlob(article.text)
        sentiment_list.append(analysis.polarity)
    data['summary'] = summary_list
    data['sentiment'] = sentiment_list
    return data


def frequency_catplot(data):
    """
    creates multi-facteted cat plots displaying the
    number of times a term is mentioned in a article versus
    the count of articles. This created two graphs: one for the
    word 'vaccine' and the other 'mask mandate'
    """
    data['count'] = data['parsed_content'].str.split().str.len()
    df1 = data[['news_source', 'count', 'parsed_content']]
    df1['vaccine_count'] = df1['parsed_content'].apply(count_vaccine)
    df1 = df1[df1['vaccine_count'] != 0]
    g = sns.catplot(x="vaccine_count", col="news_source", col_wrap=4,
                    data=df1,
                    kind="count")

    (g.set_axis_labels("'Vaccine' Mentions", "Number of Articles")
     .set_titles("{col_name} {col_var}"))
    plt.savefig('catplot_vaccine.png', bbox_inches='tight')

    df1['mask_man_count'] = df1['parsed_content'].apply(count_mask)
    df1 = df1[df1['mask_man_count'] != 0]
    h = sns.catplot(x="mask_man_count", col="news_source", col_wrap=4,
                    data=df1,
                    kind="count")

    (h.set_axis_labels("'Mask Mandate' Mentions", "Number of Articles")
     .set_titles("{col_name} {col_var}"))

    plt.savefig('catplot_mask_man.png', bbox_inches='tight')


def count_vaccine(s):
    """
    returns the count of the number of occurrences
    of the term 'vaccine'
    """
    return s.count('vaccine')


def count_mask(s):
    """
    returns the count of the number of occurrences
    of the term 'vaccine'
    """
    return s.count('mask mandate')


def summary_stat(data):
    """
    prints multiple summary statistics
    for the mentions of 'Vaccine' and 'Mask Mandate'.
    This finds the average mentions per an article, total mentions
    of each word, and the standard deviation of each
    """
    data['count'] = data['parsed_content'].str.split().str.len()
    df = data[['news_source', 'count', 'parsed_content']]
    df['vaccine_count'] = df['parsed_content'].apply(count_vaccine)
    df['mask_man_count'] = df['parsed_content'].apply(count_mask)
    cnn_msk = df['news_source'] == 'CNN'
    fox_msk = df['news_source'] == 'FOX'
    fox = df[fox_msk]
    cnn = df[cnn_msk]

    # Vaccine
    print("fox mean" + ' ' + str(fox['vaccine_count'].mean()))
    print("fox sum" + ' ' + str(fox['vaccine_count'].sum()))
    print("cnn mean" + ' ' + str(cnn['vaccine_count'].mean()))
    print("cnn sum" + ' ' + str(cnn['vaccine_count'].sum()))
    print(st.stdev(cnn['vaccine_count']))
    print(st.stdev(fox['vaccine_count']))

    # Mask Mandate
    print("fox mean" + ' ' + str(fox['mask_man_count'].mean()))
    print("fox sum" + ' ' + str(fox['mask_man_count'].sum()))
    print("cnn mean" + ' ' + str(cnn['mask_man_count'].mean()))
    print("cnn sum" + ' ' + str(cnn['mask_man_count'].sum()))
    print(st.stdev(cnn['mask_man_count']))
    print(st.stdev(fox['mask_man_count']))


def fit_and_predict_source(df):
    """
    predicts whether an article cam from
    CNN or Fox News. This function prints the
    accuracy score of the test set, a classification report,
    and the model
    """
    # splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(df['summary'],
                                                        df['Category_Target'],
                                                        test_size=0.2,
                                                        random_state=8)
    ngram_range = (1, 2)
    min_df = 10
    max_df = 1.0
    tfidf = TfidfVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words=None,
                            lowercase=False,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=None,
                            sublinear_tf=True)
    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test

    # Model Training
    model = RandomForestClassifier()
    model.fit(features_train, labels_train)
    # Model Testing
    model_predictions = model.predict(features_test)
    print('Accuracy',
          accuracy_score(labels_test, model_predictions))
    print(classification_report(labels_test, model_predictions))
    print("here is printing the model fit:")
    print(model.fit)


def main():
    df = pd.read_csv("combined_ult.csv")
    processed_csv = csv_processing(df)
    summary_csv = generate_summary(processed_csv)
    frequency_catplot(summary_csv)
    summary_stat(summary_csv)
    fit_and_predict_source(summary_csv)


if __name__ == "__main__":
    main()
