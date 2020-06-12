---
title: Performing Sentiment Analysis on Tweets and Analyzing Effect on Major Stock Indexes
date: 2019-12-15
#tags: [Python, Data Visualisation, Fundamental Data]
header:
  image: "/images/stock.jpg"
mathjax: "true"
---

# Part 1: Getting Tweets Using Twitter API

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



Here's what the tweet_df looks like:

[![](/assets/images/tweet_sentiment/tweet_df.JPG)](/assets/images/tweet_sentiment/tweet_df.JPG)


# Part 2: Preprocessing Test Data and Training a Model

There are many pre-labeled datasets that we can use to train a model. We will start off with NiekSanders dataset of pre-labeled tweets. Hopefully this dataset will be able to capture the informal language and slang commonly used on social media. 

I've already downloaded the dataset so I'll just load it in from a local drive:

```python
train_path = paths['train']
raw_train_df = pd.read_csv(os.path.join(train_path, 'NiekSanders.csv'), encoding='ISO-8859-1')
```

We need to do a bit of preprocessing on it. First, we will only keep the tweets that are labeled as 'positive' and 'negative' sentiment. Then we will use a preprocess function to tokenize each tweet and remove stop words, punctuation, URLs, @users, hashtags, convert characters to lower case, etc. 

Here's the preprocess function that we will be using (credits to [AnasAlmasri](https://gist.github.com/AnasAlmasri/853f0af319f3938754bdd447b8c56302))

We will store this function in a separate utilities module and call it to preprocess our training and test data.

```python
class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTweets(self, list_of_tweets):
        processedTweets = []
        featureVector = []
        for tweet in list_of_tweets:
            #print tweet
            processedTweet = self._processTweet(tweet[0]), tweet[1]
           # print processedTweet[0]
            processedTweets.append(processedTweet)
            featureVector.append(processedTweet[0])
        return processedTweets, featureVector

    def _processTweet(self, tweet):
        tweet = tweet.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)
        tweet = [word for word in tweet if re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word) is not None]  # remove words that doesn't start with alphabet
        return [word.strip('\'"') for word in tweet if word not in self._stopwords and len(word) > 2]
```


Now we'll do some pre-processing:

First, only keep the tweets that have positive and negative sentiment (remove the neutral sentiment)
```python
train_df = train_df[(train_df['Sentiment'] == 'positive') | (train_df['Sentiment'] == 'negative')]
train_df = train_df[['TweetText', 'Sentiment']]
```

Split the training data into training and validation set to check model accuracy after
```python
X_train, X_test, y_train, y_test = train_test_split(train_tweets['TweetText'], train_tweets['Sentiment'], test_size=0.2)
```

Convert the split up training set back into df
```python
raw_train_tweets = pd.concat([X_train, y_train], axis=1)
raw_test_tweets = pd.concat([X_test, y_test], axis=1)
```

Convert it to a list of lists to feed it to PreProcessTweets function
```python
raw_train_tweets = raw_train_tweets.values.tolist()
```

Call on the PreProcessTweets method above
```python
tweet_processor = utilities.PreProcessTweets()
preprocessed_training_set, feature_list = tweet_processor.processTweets(raw_train_tweets)
```

Flatten nested list and remove duplicates
```python
feature_list = list(set(list(chain.from_iterable(feature_list))))
```


## Part 2a: Naive Bayes Classifier:


Now that our training data is preprocessed we are ready to extract the features and feed that to the Naive Bayes Classifier.

```python
# extract feature vector for all tweets
training_set = nltk.classify.util.apply_features(extract_features, preprocessed_training_set)

# train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
```

Test it on the validation set
```python
validation_df = raw_test_tweets
validation_df['processed_tweet'] = ''
validation_df['pred_sentiment'] = ''
for i in range(len(validation_df)):
    validation_df['processed_tweet'].iloc[i] = tweet_processor._processTweet(raw_test_tweets['TweetText'].iloc[i])
    validation_df['pred_sentiment'].iloc[i] = NBClassifier.classify(extract_features(validation_df['processed_tweet'].iloc[i]))

print('accuracy %s' % accuracy_score(validation_df['pred_sentiment'], validation_df['Sentiment']))

# accuracy: 0.8401826484018264
```
Naive Bayes accuracy is 84% which isn't too bad. We will use this 84% as our baseline accuracy in assessing other classification models.


We can also run a quick test on a sample tweet to check that it's working:

```python
# test the classifier
testTweet = 'European Central Bank, acting quickly, Cuts Rates 10 Basis Points. They are trying, and succeeding, in depreciatingâ€¦ https://t.co/VtA2cMv6fm'
processedTestTweet = tweet_processor._processTweet(testTweet)
print (NBClassifier.classify(extract_features(processedTestTweet)))

# Print informative features about the classifier
print (NBClassifier.show_most_informative_features(10))
```


Now we can run our trained Naive Bayes Classifier model on the tweet_df. We will loop through the tweet_df and preprocess the tweet before classifying it.

```python
# read in test data
raw_test_df = pd.read_csv(os.path.join(paths['csv'], 'tweet_df.csv'), encoding='ISO-8859-1')
pred_tweet_df = raw_test_df
pred_tweet_df['processed_tweet'] = ''
pred_tweet_df['pred_sentiment'] = ''
for i in range(len(pred_tweet_df)):
    pred_tweet_df['processed_tweet'].iloc[i] = tweet_processor._processTweet(pred_tweet_df['msg'].iloc[i])
    pred_tweet_df['pred_sentiment'].iloc[i] = NBClassifier.classify(extract_features(pred_tweet_df['processed_tweet'].iloc[i]))

# output our prediction 
classifier_type = 'naive_bayes'
pred_tweet_df.to_csv(os.path.join(paths['prediction'], 'tweet_df-' + classifier_type + '.csv'), encoding='utf-8', index=False)
```

Here's what the pred_tweet_df looks like:
[![](/assets/images/tweet_sentiment/NB_pred_tweet_df.JPG)](/assets/images/tweet_sentiment/NB_pred_tweet_df.JPG)



# Part 3: Graphing the Sentiment vs Major Stock Index

Let's first filter for only Trump's tweets (where replies > 0)

```python
filtered_pred_df = pred_tweet_df[(pred_tweet_df['replies'] > 0)].reset_index(drop=True)
```

Since we only have the daily closing price for stocks, we should only plot a maximum of 1 sentiment a day but Trump
tweets a lot so we need a function to compress his sentiment into 1 a day.
We are also going to filter out his most popular tweets based on the amount of replies assuming that popular 
tweets are more influential.
Let's first graph a histogram of the number of replies

```python
filtered_pred_df['replies'].hist(bins=100)
```

[![](/assets/images/tweet_sentiment/filtered_pred_df_histo.JPG)](/assets/images/tweet_sentiment/filtered_pred_df_histo.JPG)


Let's see where the 90th percentile for number of replies is:

```python
filtered_pred_df['replies'].quantile(0.90) 

# returns 125882.19999999998
```


Let's filter the df again for only the most popular tweets (by number of replies), we'll set the limit to 125,000 since it's close to the 90th percentile

```python
filtered_pred_df = filtered_pred_df[filtered_pred_df['replies'] > 125000].reset_index(drop=True)
```

The filtered_pred_df looks like:
[![](/assets/images/tweet_sentiment/filtered_pred_df.JPG)](/assets/images/tweet_sentiment/filtered_pred_df.JPG)

We can see that even after filtering for the top 10% of tweets with the most replies, there are still many dates where there are multiple tweets and hence multiple, possibly conflicting sentiments. 
We need to somehow compress these sentiments so that there is only a maximum of 1 sentiment per day. 

This function will produce the avg sentiment (weighted by number of replies) for each day based on the dates in the filtered_pred_df.

```python
def compress_to_avg_daily_sentiment(filtered_pred_df):
    filtered_pred_df['created'] = pd.to_datetime(filtered_pred_df['created']).dt.date
    filtered_pred_df['sentiment_num'] = 0  # create the column
    filtered_pred_df['sentiment_num'][filtered_pred_df['pred_sentiment'] == 'negative'] = -1
    filtered_pred_df['sentiment_num'][filtered_pred_df['pred_sentiment'] == 'positive'] = 1
    filtered_pred_df['sentiment_num_times_replies'] = filtered_pred_df['sentiment_num'] * filtered_pred_df['replies']
    weighted_sentiment = filtered_pred_df.groupby('created')['sentiment_num_times_replies'].mean()
    avg_daily_sentiment_df = weighted_sentiment.to_frame(name='sentiment')
    avg_daily_sentiment_df['sentiment'] = avg_daily_sentiment_df['sentiment'].apply(lambda x: 'negative' if float(x) < 0 else 'positive')
    avg_daily_sentiment_df = avg_daily_sentiment_df.reset_index()
    return avg_daily_sentiment_df

avg_daily_sentiment_df = compress_to_avg_daily_sentiment(filtered_pred_df)
```
Let's choose our stock index to be the S&P 500, we can query this from our daily_data_yf from our MySQL database created in this [post](https://h-xiao.github.io/setup) 

```python
sp_df = frames_dict['daily_data_yf'][(frames_dict['daily_data_yf']['symbol_id'] == 2) & (frames_dict['daily_data_yf']['date'] > '2019-03-29')]
sp_df['date'] = pd.to_datetime(sp_df['date']).dt.date
```

We need to  convert tweet created date to the closest business date. We can use this function which will create a US trading calendar on the fly for a given year:

```python
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
    
def get_trading_close_holidays(year):
    inst = USTradingCalendar()
    return inst.holidays(datetime.datetime(year, 1, 1), datetime.datetime(year, 12, 31))

def get_nearest_business_day(date):
    if date.weekday() == 5:
        date =  date - datetime.timedelta(1)
    elif date.weekday() == 6:
        date = date - datetime.timedelta(2)

    holidays = get_trading_close_holidays(date.year)
    if date in holidays:
        date = date - datetime.timedelta(1)
    return date


avg_daily_sentiment_df['date'] = avg_daily_sentiment_df['created'].apply(get_nearest_business_day)
avg_daily_sentiment_df=avg_daily_sentiment_df.merge(sp_df[['date', 'close']], on='date', how='left')

```


Now we can plot the S&P500 closing price as a line graph and overlay a scatterplot showing the tweet sentiment. 


```python
sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(11, 8.5))
sns.lineplot(x='date', y='close', data=avg_daily_sentiment_df, ax=ax)
sns.scatterplot(x='created', y='close', data=avg_daily_sentiment_df, hue='sentiment')
```


Here's the plot:

[![](/assets/images/tweet_sentiment/NB_pred_sp500_plot.JPG)](/assets/images/tweet_sentiment/NB_pred_sp500_plot.JPG)



Let's run this process through a few differnt classifiers and also try using the IMDB dataset as the training set.

I'll quickly run through this process for the different datasets and plot results below.















