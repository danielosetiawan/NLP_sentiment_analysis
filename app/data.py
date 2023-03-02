import pandas as pd

def sentiment_labeler(x):
    if x < 0.4:
        return 'negative'
    elif x < 0.6:
        return 'neutral'
    else:
        return 'positive'

topics = [
    'bonds', 'cryptocurrency', 'economy', 'forex', 'housing_market',
    'interest_rates', 'recession', 'unemployment', 'inflation', 
    'stocks', 'minimum_wage', 'taxes'
    ]

for topic in topics:
    df = pd.read_csv(f'../data/cleaned_up_data/cleaned_{topic}.csv').iloc[:, 1:]
    df.Datetime = pd.to_datetime(df.Datetime).dt.strftime('%Y-%m-%d %H:00:00')
    df['labels'] = df['sentiment'].apply(sentiment_labeler)
    data = df.groupby('Datetime').agg({'sentiment': 'mean'}).reset_index()
    data['labels'] = data['sentiment'].apply(sentiment_labeler)
    
    globals()[topic] = data
    globals()[f'{topic}_raw'] = df

# sent_df = pd.read_csv('../data/preprocessing/balanced_tokenized_cleaned_stocktwits.csv', 
#                 parse_dates=['created_at']).drop('body', axis=1)

# stocks_df = pd.read_csv('../data/scraped_stocks/scraped_stock_2015_2022.csv', 
#                         parse_dates=['Date']).iloc[:, 1:]

cleaned_df = pd.read_csv('../data/word_cloud_df.csv', index_col=None)

tweets_df = pd.read_csv('../data/cleaned_up_data/final_stock_tweets_summary.csv')

# combined_df = pd.read_csv('../data/combined_sentiment.csv', index_col=None)
# combined_df['sentiment'] = combined_df['sentiment'].replace(to_replace=0, value=-1)
# combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])
# combined_df['date'] = combined_df['created_at'].dt.date

# df = sent_df['raw_content'].str.upper().str.extractall(r'\$(\w+)')[0].reset_index()

# # remove indexes containing more than one ticker
# df = sent_df.drop(df[df['match'] == 1]['level_0'].tolist())

# # add ticker 
# df['ticker'] = df['raw_content'].str.upper().str.extract(r'\$(\w+)')

# # extract date only
# df['Date'] = df['created_at'].dt.date

# # find all indexes with tickers
# df = sent_df['raw_content'].str.upper().str.extractall(r'\$(\w+)')[0].reset_index()

# # remove indexes containing more than one ticker
# df = sent_df.drop(df[df['match'] == 1]['level_0'].tolist())

# # add ticker 
# df['ticker'] = df['raw_content'].str.upper().str.extract(r'\$(\w+)')

# # extract date only
# df['Date'] = df['created_at'].dt.date

# def clean_tickers(ticker):
#     if 'AAPL' in ticker:
#         ticker = 'AAPL'
#     elif 'AMZN' in ticker:
#         ticker = 'AMZN'
#     elif 'FB' in ticker:
#         ticker = 'META'
#     else:
#         ticker = ticker
        
#     return ticker

# df['ticker'] = df['ticker'].apply(lambda x: clean_tickers(x))

# # get average sentiment per day
# df = df.groupby(['ticker', 'Date']).agg({'sentiment': 'mean'}).reset_index()
# df['Date'] = pd.to_datetime(df['Date'])

# for tick in df['ticker'].unique():
#     df1 = df[df['ticker'] == tick]
#     df2 = stocks_df[stocks_df['Stock Name'] == tick]
    
#     globals()[tick] = pd.merge(df1, df2, on='Date').iloc[:, 1:]
#     globals()[tick]['color'] = globals()[tick]['sentiment'].apply(lambda x: 'green' if x > 0.5 else 'red')
#     globals()[tick]['label'] = globals()[tick]['sentiment'].apply(lambda x: 'Bullish' if x > 0.5 else 'Bearish')