---
title: Performing Sentiment Analysis on Tweets and Analyzing Their Effect on Major Stock Indexes
date: 2019-10-01
#tags: [Python, Twitter, Sentiment Analysis]
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



```python
# pre-process data
raw_train_df = raw_train_df[(raw_train_df['Sentiment'] == 'positive') | (raw_train_df['Sentiment'] == 'negative')]
raw_train_tweets = raw_train_df[['TweetText', 'Sentiment']]
raw_train_tweets = raw_train_tweets.values.tolist()

tweet_processor = utilities.PreProcessTweets()
preprocessed_training_set, feature_list = tweet_processor.processTweets(raw_train_tweets)

# flatten nested list and remove duplicates
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


We can run a quick test on a sample tweet to check that it's working

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













