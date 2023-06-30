import pandas as pd
import tweepy
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import yaml
import praw
import constants as cn
import torch
import re
import emoji
import math
import requests
import twint
from bs4 import BeautifulSoup

import nest_asyncio
nest_asyncio.apply()

torch.manual_seed(globals.seed_val)
torch.cuda.manual_seed_all(globals.seed_val)


class StockScraper():
    def __init__(self, keyword, start_date, end_date):
        self.keyword = keyword
        self.start_date = start_date
        self.end_date = end_date
        self.query = f"{self.keyword} since:{self.start_date} until:{self.end_date}"
        self.scraped_yahoo = False
        self.scraped_twitter = False
        self.scraped_reddit = False
        self.scraped_news = False
        self.authentication = self.load_authentication()

    def __str__(self) -> str:
        return f'Searching for: {self.query}'
    
    def status(self):
        '''
        prints status of websites scraped
        '''
        print('Current Status:')
        for key, val in dict(
                Scraped_Yahoo=self.scraped_yahoo,
                Scraped_Twitter=self.scraped_twitter,
                Scraped_Reddit=self.scraped_reddit,
                Scraped_News=self.scraped_news
            ).items():
                print(f' - {key}: {val}')

    def load_authentication(self):
        '''
        loads authentication keys from yaml file
        :path: path to yaml file
        '''

        with open(globals.authentication_file, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
        
    def scrape_stocktwits(self, max_num) -> pd.DataFrame:
        url = f'https://stocktwits.com/symbol/{self.keyword}'

        # Send a GET request to the StockTwits page
        response = requests.get(url)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the relevant HTML elements and extract the desired information
        messages = soup.find_all('div', class_='message-stream-container')

        # keep count of number of messages scraped
        scraped_count = 0

        for message in messages:
            username = message.find('a', class_='username').text.strip()
            content = message.find('div', class_='message-body').text.strip()

            print(f"Username: {username}")
            print(f"Content: {content}")
            print()

            scraped_count += 1

            if scraped_count >= max_num:
                break

    def scrape_tweepy(self,
                      max_num:int=1000,
                      ) -> pd.DataFrame:
        '''
        Scrapes Tweet through official API
        :max_num: maximum number of tweets to scrape
        :verbose: whether to print progress. Can be set to int to print every n iterations
        '''

        # check if authentication is set
        authentication = self.authentication['twitter_authentication']
        if None in authentication.values():
            raise Exception('Authentication not set. Please set authentication first.')
        
        auth = tweepy.OAuthHandler(authentication['consumer_key'], 
                                    authentication['consumer_secret'])
        auth.set_access_token(authentication['access_token'], 
                                authentication['access_token_secret'])
        api = tweepy.API(auth)

        self.scraped_twitter = True
        return api.search_tweets(q=self.keyword, count=max_num, lang='en')

    def scrape_snscrape(self, 
                     max_num:int=1000,
                     verbose:'bool or int'=1000,
                    ) -> pd.DataFrame:
        '''
        Scrapes Tweet through unofficial backdoor
        :max_num: maximum number of tweets to scrape
        :verbose: whether to print progress. Can be set to int to print every n iterations
        '''

        tweets = []

        for idx, twt in enumerate(sntwitter.TwitterSearchScraper(self.query).get_items()):
            if idx>max_num:
                break

            if verbose and idx>verbose:
                print(f'{idx} reached')

            tweets.append([twt.date, twt.rawContent])

        self.scraped_twitter = True
        return pd.DataFrame(tweets, columns=['Datetime', 'Text'])
    
    def scrape_twint(self,
                     max_num:int=1000,
                     ) -> pd.DataFrame:
        c = twint.Config()
        c.Limit = max_num
        # c.Since = self.start_date
        # c.Until = self.end_date
        c.Search = self.keyword

        twint.run.Search(c)

        self.scraped_twitter = True
        return twint.storage.panda.Tweets_df


    def scrape_yahoo(self) -> pd.DataFrame:
        '''
        scrapes Yahoo Finance
        '''
        if self.start_date == self.end_date:
            self.end_date = datetime.today() + timedelta(days=1)
        
        self.scraped_yahoo = True
        return si.get_data(self.keyword, self.start_date, self.end_date, index_as_date=False)

    def scrape_reddit(self, 
                      subreddit:str='wallstreetbets',
                      post_id:str=None,
                      max_num:int=None,
                      type='posts'
                      ) -> pd.DataFrame:
        '''
        Scrapes Reddit
        :subreddit: subreddit to scrape
        :post_id: id of post to scrape comments from
        :max_num: maximum number of posts to scrape
        :type: type of data to scrape. Can be 'posts' or 'comments'
        '''
        all_posts, all_comments = [], []

        # check if authentication is set
        authentication = self.authentication['reddit_authentication']
        if None in authentication.values():
            raise Exception('Authentication not set. Please set authentication first.')

        reddit = praw.Reddit(
            client_id=authentication['client_id'],
            client_secret=authentication['client_secret'],
            user_agent=self.authentication['user_agent']
        )

        if type=='posts':
            subreddit = reddit.subreddit(subreddit)
            posts = subreddit.new(limit=max_num)

            for post in posts:
                all_posts.append([post.title, post.selftext])

            self.scraped_reddit = True
            return pd.DataFrame(all_posts, columns=['Title', 'Text'])

        if type=='comments':
            submission = reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list()[:max_num]:
                all_comments.append([comment.body])

            self.scraped_reddit = True
            return pd.DataFrame(all_comments, columns=['Text'])

    def scrape_news():
        pass

class PredictSentiment():
    def __init__(self):
        self.model = globals.model
        self.tokenizer = globals.tokenizer
        self.device = globals.device

    def preprocess_sentence(self, sent:str) -> str:
        '''
        preprocess sentence to be fed into model
        '''
        
        # lowercase
        # message = message.lower() # RoBERTa tokenizer is uncased
        # remove URLs
        sentence = re.sub(r'https?://\S+', "", sent)
        sentence = re.sub(r'www.\S+', "", sentence)
        # remove '
        sentence = sentence.replace('&#39;', "'")
        # remove symbol names
        sentence = re.sub(r'(\#)(\S+)', r'hashtag_\2', sentence)
        sentence = re.sub(r'(\$)([A-Za-z]+)', r'cashtag_\2', sentence)
        # remove usernames
        sentence = re.sub(r'(\@)(\S+)', r'mention_\2', sentence)
        # demojize
        sentence = emoji.demojize(sentence, delimiters=("", " "))

        return sentence.strip()
    
    def predict_sentiment(self, sent:str) -> tuple:
        '''
        predicts sentiment of sentence
        '''
        encoded_dict = self.tokenizer.encode_plus(
            sent, 
            add_special_tokens = True,
            truncation=True,
            max_length = 64,
            padding='max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        input_id = torch.LongTensor(encoded_dict['input_ids']).to(self.device)
        attention_mask = torch.LongTensor(encoded_dict['attention_mask']).to(self.device)
        model = self.model.to(self.device)

        with torch.no_grad():
            outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)

        logits = outputs[0]
        index = logits.argmax()
        return index, logits

    def check_sentence(self, sent,return_logits=True) -> str:
        '''
        add label to probability logits
        '''
        labels = ['Bearish','Bullish']
        sent_processed = self.preprocess_sentence(sent)
        index, logits = self.predict_sentiment(sent_processed)
        if return_logits:
            logit0 = math.exp(logits[0][0])
            logit1 = math.exp(logits[0][1])
            logits = [logit0/(logit0+logit1),logit1/(logit0+logit1)]
            return [labels[index], max(logits)]
        return labels[index]