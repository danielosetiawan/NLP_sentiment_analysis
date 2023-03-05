import pandas as pd

topics = [
    'bonds', 'economy', 'housing_market', 'taxes', 'loans',
    'interest_rates', 'recession', 'unemployment', 'inflation'
    ]

# cleaned_df = pd.read_csv('../data/word_cloud_df.csv', index_col=None)
tweets_df = pd.read_csv('../data/cleaned_up_data/final_stock_tweets_summary.csv')
topics_stats = pd.read_csv('../data/cleaned_up_data/topics/final_sent_topic_stats_summary.csv', index_col='Type')
topics_df = pd.read_csv('../data/cleaned_up_data/topics/final_sent_topic_chart_summary.csv')
wordcloud_df = pd.read_csv('../data/cleaned_up_data/topics/final_wordcloud_summary.csv', index_col='words')

topic_dct = {
    'bonds': {
        'title': 'US Bonds',
        'data': ['bonds_price', 'bonds_issued'],
        'name': ['Price', 'Issued'],
        'yaxis_title': ['Bonds Price', 'Bonds Issued'],
    },
    'economy': {
        'title': 'US Economy',
        'data': ['GDP', 'consumer_confidence'],
        'name': ['GDP', 'Confidence'],
        'yaxis_title': ['GDP', 'Consumer Confidence'],
    },
    'recession': {  
        'title': 'Recession',
        'data': ['CPI', 'debt'], 
        'name': ['Inflation', 'Debt'],
        'yaxis_title': ['Inflation', 'Total Debt'],
    },
    'unemployment': {  
        'title': 'Unemployment',
        'data': ['employment_rate', 'unemployment_rate'], 
        'name': ['Employment', 'Unemployment'],
        'yaxis_title': ['Employment Rate', 'Unemployment Rate'],
    },
    'housing_market': {  
        'title': 'Housing Market',
        'data': ['housing', 'mortgage_rates'], 
        'name': ['Prices', 'Rates'],
        'yaxis_title': ['Housing Prices', 'Mortgage Rates'],
    },
    'interest_rates': { 
        'title': 'Interest Rates',
        'data': ['federal_funds', 'consumer_confidence'], 
        'name': ['Rates', 'Confidence'],
        'yaxis_title': ['Interest Rates', 'Consumer Confidence'],
    },
    'inflation': {
        'title': 'Inflation',
        'data': ['inflation', 'CPI'], 
        'name': ['Inflation', 'CPI'],
        'yaxis_title': ['Inflation', 'CPI'],
    },
    'taxes': {  
        'title': 'Taxes',
        'data': ['IRS Tax', 'GDP'], 
        'name': ['Paid to IRS', 'GDP'],
        'yaxis_title': ['Taxes Paid to IRS', 'GDP'],
    },
    'loans': {  
        'title': 'Loans',
        'data': ['bank_loan', 'consumer_confidence'], 
        'name': ['Bank Loans', 'Confidence'],
        'yaxis_title': ['Bank Loan Rates', 'Consumer Confidence'],
    },
}