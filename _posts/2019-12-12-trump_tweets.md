---
title: "Quantitative Trading Project"
date: 2019-10-01
#tags: [Python, Twitter, Sentiment Analysis]
header:
  image: "/images/stock.jpg"
mathjax: "true"
---

# Performing Sentiment Analysis on Tweets and Analyzing Their Effect on Major Stock Indexes

## Part 1: Getting Tweets Using Twitter API

In order to access Twitter's API, you will need to register for a Twitter developer account. You should get the API key in 1-2 weeks after registering.

After you receive the API keys, connect to Twitter API:

```python
auth = tweepy.OAuthHandler(api_keys['twitter']['consumer_key'], api_keys['twitter']['consumer_secret'])
auth.set_access_token(api_keys['twitter']['access_token'], api_keys['twitter']['access_token_secret'])
```

Authenticate connection to API:

```python
api = tweepy.API(auth)
```

Let's get Donald Trump's tweets given he's known to have a noticeable impact on the stock market. 
We will first get a list of pages to loop through then get the tweets from each page. The maximum amount of tweets we can get is 3000 since older tweets are archived by Twitter.

```python
page_list = []
for page in tweepy.Cursor(api.user_timeline, screen_name='@realDonaldTrump', count=200).pages(16):
    page_list.append(page)

for page in page_list:
    for status in page:
        created = status.created_at
        message = status.text.encode('utf-8')
        num_replies = status.favorite_count
        num_retweets = status.retweet_count
        tweet_df = tweet_df.append({'created': created, 'msg': message, 'replies': num_replies,
                                    'retweets': num_retweets}, ignore_index=True)

tweet_df.to_csv(os.path.join(paths['csv'], 'tweet_df.csv'), encoding='utf-8', index=False)```
````
















