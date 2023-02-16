import pandas as pd

sent_df = pd.read_csv('../data/balanced_tokenized_cleaned_stocktwits.csv', 
                 parse_dates=['created_at']).drop('body', axis=1)

stocks_df = pd.read_csv('../data/scraped_stock_2015_2022.csv', 
                        parse_dates=['Date']).iloc[:, 1:]

cleaned_df = pd.read_csv('../data/word_cloud_df.csv', index_col=None)

df = sent_df['raw_content'].str.upper().str.extractall(r'\$(\w+)')[0].reset_index()

# remove indexes containing more than one ticker
df = sent_df.drop(df[df['match'] == 1]['level_0'].tolist())

# add ticker 
df['ticker'] = df['raw_content'].str.upper().str.extract(r'\$(\w+)')

# extract date only
df['Date'] = df['created_at'].dt.date

# find all indexes with tickers
df = sent_df['raw_content'].str.upper().str.extractall(r'\$(\w+)')[0].reset_index()

# remove indexes containing more than one ticker
df = sent_df.drop(df[df['match'] == 1]['level_0'].tolist())

# add ticker 
df['ticker'] = df['raw_content'].str.upper().str.extract(r'\$(\w+)')

# extract date only
df['Date'] = df['created_at'].dt.date

def clean_tickers(ticker):
    if 'AAPL' in ticker:
        ticker = 'AAPL'
    elif 'AMZN' in ticker:
        ticker = 'AMZN'
    elif 'FB' in ticker:
        ticker = 'META'
    else:
        ticker = ticker
        
    return ticker

df['ticker'] = df['ticker'].apply(lambda x: clean_tickers(x))

# get average sentiment per day
df = df.groupby(['ticker', 'Date']).agg({'sentiment': 'mean'}).reset_index()
df['Date'] = pd.to_datetime(df['Date'])

for tick in df['ticker'].unique():
    df1 = df[df['ticker'] == tick]
    df2 = stocks_df[stocks_df['Stock Name'] == tick]
    
    globals()[tick] = pd.merge(df1, df2, on='Date').iloc[:, 1:]
    globals()[tick]['color'] = globals()[tick]['sentiment'].apply(lambda x: 'green' if x > 0.5 else 'red')
    globals()[tick]['label'] = globals()[tick]['sentiment'].apply(lambda x: 'Bullish' if x > 0.5 else 'Bearish')