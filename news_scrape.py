"""
Brandon Wong, Johnny He
CSE 163

This file contains all code for the web scraping portion of our project.
This file passes in two csv files where one contains article links from
CNN while the other contains articles from FOX News. Then, the various
functions extract the article text from each link and store it in a dataframe.
"""
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

"""
This file handles all of our webscraping functions.
We first pass in a csv of numerous article links from CNN and FOX
News and convert to dataframes with added columns such as the News
source and an article id.
"""

CNN_FILE = "cnn_link2.csv"
FOX_FILE = "fox_link2.csv"


def get_links(data):
    """
    This function returns a csv's link column
    into a list
    """
    return(list(data['link']))


def word_scrape_fox(URL):
    """
    returns a dataframe of all fox news articles with various columns.
    This includes an article id column, a article content column, and
    the news source it came from.
    """
    news = ''
    count = 36
    news_id = []
    articles = []
    source = []
    link = []
    for url in range(0, len(URL)):
        req = requests.get(URL[url])
        soup_content = bs(req.content, 'lxml')
        for pgh in soup_content.findAll('p'):
            news += pgh.text.strip()
        articles.append(news)
        news = ''
        count += 1
        news_id.append(count)
        source.append('FOX')
        link.append(URL[url])
    list_of_tuples = list(zip(news_id, link, articles, source))
    df = pd.DataFrame(list_of_tuples,
                      columns=['article_id', 'link',
                               'article_content', 'news_source'])
    return df


def word_scrape_cnn(URL):
    """
    returns a dataframe of all cnn news articles with various columns.
    This includes an article id column, a article content column, and
    the news source it came from. The reason why there is a separate
    function for each news source is that their article content is
    extracted differently.
    """
    news = ''
    count = 0
    news_id = []
    articles = []
    source = []
    link = []
    for url in range(0, len(URL)):
        req = requests.get(URL[url])
        soup_content = bs(req.content, 'lxml')
        for pgh in soup_content.findAll('div',
                                        {'class': 'zn-body__paragraph'}):
            news += pgh.text.strip()
        articles.append(news)
        news = ''
        count += 1
        news_id.append(count)
        source.append('CNN')
        link.append(URL[url])
    list_of_tuples = list(zip(news_id, link, articles, source))
    df = pd.DataFrame(list_of_tuples,
                      columns=['article_id', 'link',
                               'article_content', 'news_source'])
    return df


def to_txt(data, data2):
    """
    passes in a dataframe and returns
    a text file of all the rows in the
    'article_content' column
    """
    print("Here begins Article content!")
    paragraph_list = []
    paragraph_list2 = []
    for text in data["article_content"]:
        paragraph_list.append(text)
    my_txt = " ".join(paragraph_list)
    # print(my_txt)
    with open("all_text/cnn_all.txt", "w") as file:
        file.write((my_txt))

    for text in data2["article_content"]:
        paragraph_list2.append(text)
    my_txt2 = " ".join(paragraph_list2)
    print(my_txt2)
    with open("all_text/fox_all.txt", "w") as file:
        file.write((my_txt2))


def merged_df(df1, df2):
    """
    returns a combined csv dataset of CNN and FOX news articles
    """
    result_df = df1.append([df2])
    result_df.to_csv('combined_ult.csv', index=False)


def main():
    cnn_df = pd.read_csv(CNN_FILE)
    fox_df = pd.read_csv(FOX_FILE)
    all_cnn_links = get_links(cnn_df)
    all_fox_links = get_links(fox_df)
    cnn_pd_df = word_scrape_cnn(all_cnn_links)
    fox_pd_df = word_scrape_fox(all_fox_links)
    merged_df(cnn_pd_df, fox_pd_df)
    to_txt(cnn_pd_df, fox_pd_df)


if __name__ == '__main__':
    main()
