
import csv
import arrow
import pandas as pd
from bs4 import BeautifulSoup
from alpaca_trade_api import REST
import pandas_datareader.data as web
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']
def Getdata(period):
    #Get the recent period*3 days of date and save it into a csv file called data.csv
    API_KEY='AKFNZ4YIKLX0IJNNIR9D'
    API_SECRET='eryQytPBu9gJ9B5vmowi3sAF4FtfZukTEgULnCoo'
    rest_client = REST(API_KEY, API_SECRET)


    date_list = []  # Create an empty list to store dates
    list_of_news=[] # Create an empty list to store news
    #period = 10  # Set period, the data range is 3*period days forward of the current day's date
    date = arrow.utcnow().shift(days=3) #Get the days three days after the current date
    num = 0 #News quantity initialization

    for i in range(period):
        date = date.shift(days=-3)
        date_list.insert(0, date.format('YYYY-MM-DD'))  # Add the date to the beginning of the date list
    # Generate a list of dates

    for j in range(len(date_list)-1):
        news = rest_client.get_news("TSLA",date_list[j],date_list[j+1],limit=50, include_content=True)
        num += len(news)
    # Get the original news data, where you can change the news topic

        for p  in news:
            list_of_news.append(eval(str(p)[7:-1])) # convert each entry in to dic

        for q in list_of_news:
            soup = BeautifulSoup(q['content'], 'html.parser')
            q['content'] = soup.get_text()      # Cleaning text in 'content'
            q["created_at"] = str(q["created_at"])[0:10]    # Change the time format to the standard yyyy-mm-dd

    keys_to_extract = ["created_at","headline","content"]
    new_list_of_news = [{key: value for key, value in dictionary.items() if key in keys_to_extract} for dictionary in list_of_news]

    with open('news.csv', 'w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(keys_to_extract)
        for dictionary in new_list_of_news:
            row_values = [dictionary[key] for key in keys_to_extract]
            writer.writerow(row_values)

    #print(num) #Number of news items
    print('Finish get news!')
    # Get the news data and complete the text cleaning and store the data in ¡®news.csv¡¯



    # The following section performs sentiment analysis on news data
    news_data = pd.read_csv('news.csv')
    # Pre-processed data
    news_data['headline'] = news_data['headline'].fillna('')
    news_data['content'] = news_data['content'].fillna('')
    news_data['text'] = news_data['headline'] + " " + news_data['content']
    news_data['date'] = pd.to_datetime(news_data['created_at'])

    # Use VADER to calculate sentiment scores
    def vader_sentiment(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']

    news_data['vader_sentiment'] = news_data['text'].apply(vader_sentiment)

    # Grouped by date and calculate the average sentiment factor for each day
    daily_sentiment_vader = news_data.groupby(['date'])['vader_sentiment'].mean().reset_index()

    # Keep vader_sentiment to four decimal places
    daily_sentiment_vader['vader_sentiment'] = daily_sentiment_vader['vader_sentiment'].round(4)

    # Store the results in the ¡®emotion.csv¡¯ file
    daily_sentiment_vader.to_csv('emotion.csv', index=False)

    print('Finish get emotional factor!')



    # The following section obtains stock data
    today = datetime.today()
    # Get the date 3*period ago
    x_days_ago = today - timedelta(days=3*period)
    df = web.DataReader('TSLA','stooq',x_days_ago,today)
    df.to_csv('price.csv')

    print('Finish get stock price!')



    #Merging data sets
    df_price = pd.read_csv('price.csv', parse_dates=['Date'], index_col='Date')
    df_emotion = pd.read_csv('emotion.csv', parse_dates=['date'], index_col='date')
    # Merge two DataFrames by time
    df_merged = pd.merge(df_price, df_emotion, left_index=True, right_index=True)
    # Store the merged DataFrame in ¡®data.csv'
    df_merged.to_csv('data.csv')

    print('Finish get all data!')