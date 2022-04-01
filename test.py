import pandas as pd


'''
Testing the scraping result accuracy
by testing if the length of each article
matches up.
'''
df = pd.read_csv("Combined_Preprocess_Ult.csv")
# print(df.head()["news_length"])
# print(df["news_length"][0])
# print(df[df["news_source"] == "CNN"])
assert df["news_length"][0] == 2803
assert df["news_length"][1] == 7383
assert df["news_length"][2] == 7811
assert df["news_length"][3] == 4068
assert df["news_length"][4] == 8870
'''
Testing the categorical labels for CNN are converted
correctly
'''
cnn_df = df[df["news_source"] == "CNN"]
cnn_list = cnn_df["Category_Target"].to_list()
assert 0 in cnn_list
'''
Testing the categorical labels for FOX are converted
correctly
'''
fox_df = df[df["news_source"] == "FOX"]
fox_list = fox_df["Category_Target"].to_list()
assert 1 in fox_list
