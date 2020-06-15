---
title: Analyzing Sentiment from Earnings Call Transcripts
date: 2020-05-15
#tags: [Python, Sentiment, Stock Price, Graphs, Analysis]
header:
  image: "/images/stock_l0.1w.jpg"
mathjax: "true"
---

So far, we have been able to gather daily stock price data, quarterly and annual fundamental data and earnings call transcript sentiment predictions. In this section I will play around with plotting these against each other to see if we can gather some interesting insights.


# Part 1: Plotting Stock Price Against Earnings Call Sentiment

Using the historic transcript data that we got from parsing earning call PDFs and the daily prices we got from Yahoo Finance we can plot these.

Load in our price and fundamental stock data from SQL tables, details for how to create this is in this [post](https://h-xiao.github.io/setup).
Also load in the earnings call sentiment predicted from XLNet from the previous [post](https://h-xiao.github.io/predict_earning_call_sentiment_p2). 

```python
# load in our tables from SQL into a dict of dfs 
# run time: ~ 5 min
t0 = time.time()
frames_dict = get_tables_from_db(False, 'all')
t1=time.time()
print(t1-t0)


# read in predicted trans sentiment & split folder column to ticker & date column
path = os.path.join(paths['trans'])
pred_df_all = pd.read_csv(os.path.join(path, 'combined_trans_predictions.csv'))
pred_df_all[['ticker', 'date']] = pred_df_all['folder'].str.split('_', expand=True)
```

Since we saved down the transcripts using 2 different methods, I wrote a function to remove any duplicate transcripts that we might have saved down unintentionally:

```python
def drop_dup_in_parsed_hist_pred(pred_df_raw):
    # if there's duplicate dates from the historic & parsed, have a flag to decide which one to use
    dup_use_parsed_flag = True

    pred_parsed_df = pred_df_raw[pred_df_raw['folder'].str[-4:] != '.csv']
    pred_hist_df = pred_df_raw[pred_df_raw['folder'].str[-4:] == '.csv']
    pred_parsed_df['date'] = pred_parsed_df['date'].apply(parse)
    pred_hist_df['date'] = pred_hist_df['date'].str.replace('.csv', '')
    pred_hist_df['date'] = pred_hist_df['date'].apply(parse)

    date_list_parsed = list(set(pred_parsed_df['date'].tolist()))
    date_list_hist = list(set(pred_hist_df['date'].tolist()))

    if dup_use_parsed_flag == True:
        pred_hist_df_filtered = pred_hist_df[~pred_hist_df['date'].isin(date_list_parsed)]
        pred_df = pred_parsed_df.append(pred_hist_df_filtered)
    return pred_df
```


This function will filter the daily pricing data df for the ticker and for a start date after the first date that we have sentiment predictions for:

```python
def filter_for_price_df(pred_df, ticker):
    sent_df = pred_df.groupby('date', as_index=False)['pred'].mean()
    start_date = sent_df['date'].sort_values()[0]
    end_date = sent_df['date'].sort_values()[len(sent_df)-1]
    
    # filter for pricing data of ticker
    ticker_dict = dict(zip(frames_dict['ticker_ids'].ticker, frames_dict['ticker_ids'].index))
    price_df_raw = frames_dict['daily_data_yf'][frames_dict['daily_data_yf']['symbol_id'] == ticker_dict[ticker]]
    price_df_raw = price_df_raw.drop_duplicates(subset='Date', keep="first")
    price_df_raw = price_df_raw.rename(columns={'Date':'date'})
    price_df = price_df_raw[price_df_raw['date'] >= prev_n_business_day(start_date,1)]
    return price_df
```


I will create some 4x4 subplots for tickers that I have historic transcript data for (these are the transcripts that we parsed from PDFs):

```python
ticker_list = ['CMCSA', 'EL', 'FB', 'ICE']

fig = plt.figure()
sns.set(font_scale = 0.6)
fig.subplots_adjust(hspace=0.35, wspace=0.25)
for i,v in enumerate(ticker_list):
    # filter for sentiment for ticker, and price data for ticker
    pred_df_raw = pred_df_all[pred_df_all['ticker'] == v]
    pred_df = drop_dup_in_parsed_hist_pred(pred_df_raw)
    sent_df = pred_df.groupby('date', as_index=False)['pred'].mean()
    price_df = filter_for_price_df(sent_df, v)

    data1 = price_df['close'].tolist()
    times1 = price_df['date'].tolist()

    data2 = sent_df['pred'].tolist()
    times2 = sent_df['date'].tolist()

    ax1 = fig.add_subplot(2, 2, i+1)
    ax2 = ax1.twinx()
    ax1.plot(times1, data1, '-', color='b')
    ax2.plot(times2, data2, '-', color='r')

    ax1.set_xlabel('date', fontdict={'fontsize': 8})
    ax1.set_ylabel('close', color='b', fontdict={'fontsize': 8})
    ax2.set_ylabel('pred', color='r', fontdict={'fontsize': 8})
    ax1.set_title(v, fontdict={'fontsize': 10}, fontweight='bold')

    plt.show()
```

This is what the plot looks like:

[![](/assets/images/graph_earning_call_sentiment/sent_vs_price_plot1.JPG)](/assets/images/graph_earning_call_sentiment/sent_vs_price_plot1.JPG)


Running it on more tickers:

[![](/assets/images/graph_earning_call_sentiment/sent_vs_price_plot2.JPG)](/assets/images/graph_earning_call_sentiment/sent_vs_price_plot2.JPG)


[![](/assets/images/graph_earning_call_sentiment/sent_vs_price_plot3.JPG)](/assets/images/graph_earning_call_sentiment/sent_vs_price_plot3.JPG)



