import pandas as pd
import tweepy
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import yaml
import praw
import globals


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
        for key, val in dict(
                Scraped_Yahoo=self.scraped_yahoo,
                Scraped_Twitter=self.scraped_twitter,
                Scraped_Reddit=self.scraped_reddit,
                Scraped_News=self.scraped_news
            ).items():
                print(f'{key}: {val}')

    def load_authentication(self):
        '''
        loads authentication keys from yaml file
        :path: path to yaml file
        '''

        with open(globals.authentication_file, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def scrape_tweet(self, 
                     max_num:int=1000,
                     verbose:'bool or int'=1000,
                     api:bool=True,
                    ) -> pd.DataFrame:
        '''
        Scrapes Tweets
        :max_num: maximum number of tweets to scrape
        :api: whether to use official API or backdoor
        :verbose: whether to print progress. Can be set to int to print every n iterations
        '''

        tweets = []

        if api:
            # check if authentication is set
            authentication = self.authentication['twitter authentication']
            if None in authentication.values():
                raise Exception('Authentication not set. Please set authentication first.')
            
            auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            api = tweepy.API(auth)

            self.scraped_twitter = True
            return api.search_tweets(q=self.keyword, count=max_num, lang='en')
        
        else:
            
            for idx, twt in enumerate(sntwitter.TwitterSearchScraper(self.query).get_items()):
                if idx>max_num:
                    break

                if verbose and idx>verbose:
                    print(f'{idx} reached')

                tweets.append([twt.date, twt.rawContent])

            self.scraped_twitter = True
            return pd.DataFrame(tweets, columns=['Datetime', 'Text'])

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
                      max_num:int=1000,
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

