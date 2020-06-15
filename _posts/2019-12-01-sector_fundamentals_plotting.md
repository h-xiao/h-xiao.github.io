---
title: Data Visualisation with Fundamental Stock Data
date: 2019-12-01
#tags: [Python, Twitter, Sentiment Analysis]
header:
  image: "/images/stock2.jpg"
mathjax: "true"
---

# Part 1: Calculating Financial Ratios for Tickers in S&P500

Back in my first [post](https://h-xiao.github.io/setup), we were able to get some fundamental stock data for tickers in the S&P500. 

Let's start off by calculating some return ratios and plotting these on a 3D graph after. We can separate these graphs by subsector to see if we can visually find outperforming or underperforming companies. The return ratios we will calculate are: Return on Asset (ROA),  Return on Equity (ROE), and Return on Investment (ROI).

Let's do this analysis on the quarterly financial data since the last four quarters provides the most recent data. This exercise can also work for annual data with slight adjustments. 


In the first [post](https://h-xiao.github.io/setup), we obtained:
* quarterly financial data for S&P500 tickers (fundamental_qt_data)
* closing prices for S&P500 tickers (daily_data_yf)
* industries, and sub-industries from Wiki for S&P500 tickers (sp_ticker_data)
* tickers table which maps symbol ids to actual tickers (tickers) 



Here's a recap of what all these tables look like. 

fundamental_qt_data:
[![](/assets/images/sector_fundamentals_plotting/fundamental_qt_data.JPG)](/assets/images/sector_fundamentals_plotting/fundamental_qt_data.JPG)


daily_data_yf:
[![](/assets/images/sector_fundamentals_plotting/daily_data_yf.JPG)](/assets/images/sector_fundamentals_plotting/daily_data_yf.JPG)


sp_ticker_data:
[![](/assets/images/sector_fundamentals_plotting/sp_ticker_data.JPG)](/assets/images/sector_fundamentals_plotting/sp_ticker_data.JPG)


tickers:
[![](/assets/images/sector_fundamentals_plotting/tickers.JPG)](/assets/images/sector_fundamentals_plotting/tickers.JPG)


To make calculations easier, we will first join these tables together into one table that has all the data we need to do the calculations. We'll call it merged_df: 

```python
merged_df = pd.merge(frames_dict['fundamental_qt_data'], frames_dict['tickers'], how='left', left_on='symbol_id', right_on='index', sort=False).drop(['index'], axis=1)

merged_df = pd.merge(merged_df, frames_dict['sp_ticker_data'][['Symbol', 'GICS Sector', 'GICS Sub Industry']], how='left', left_on='ticker', right_on='Symbol', sort=False).drop(['Symbol'], axis=1)
```


Since some of the dates in the fundamental_qt_data are not business dates, we will use the get_nearest_business_day function to convert it to the previous business date. We are using the same get_nearest_business_day function from this [post](https://h-xiao.github.io/trump_tweets).

```python
merged_df['bus_date'] = merged_df.date.swifter.apply(get_nearest_business_day)
```


Now we need to create a key to join merged_df and daily_data_yf on (date + '_' + symbol_id).
The merged_df contains our quartly financial statement data and the daily_data_yf contains our yahoo finance closing prices. So we would like to extract the closing price from daily_data_yf for the closest business date of each quaterly financial statement.   

```python
merged_df['temp_key'] = merged_df['bus_date'].astype(str) + '_' + merged_df['symbol_id'].astype(str)

frames_dict['daily_data_yf']['temp_key'] = frames_dict['daily_data_yf']['date'].astype(str) + '_' + 

frames_dict['daily_data_yf']['symbol_id'].astype(str)

merged_df = pd.merge(merged_df, frames_dict['daily_data_yf'][['temp_key', 'close']], how='left', on='temp_key', sort=False).drop(['temp_key'], axis=1)
```


In order to calculate ROI, we will need the dividends paid out in each quarter which I have created a function for:

```python
def calc_dividends_in_qt(daily_data_yf_df, merged_df):
    distinct_symbol_id_list = list(set(merged_df['symbol_id'].tolist()))
    merged_df_out = pd.DataFrame(columns = merged_df.columns.tolist() + ['dividends_qt_sum', 'previous_qt_close'])
    for i in range(len(distinct_symbol_id_list)):
        curr_merged_df = merged_df[merged_df['symbol_id'] == distinct_symbol_id_list[i]]
        curr_merged_df = curr_merged_df.reset_index()
        curr_merged_df = curr_merged_df.sort_values(by='date')
        curr_daily_data_yf_df = daily_data_yf_df[daily_data_yf_df['symbol_id'] == distinct_symbol_id_list[i]]
        curr_daily_data_yf_df = curr_daily_data_yf_df.sort_values(by='date')
        curr_daily_data_yf_df = curr_daily_data_yf_df.set_index('date')
        curr_merged_df['dividends_qt_sum'] = 0  # create column
        curr_merged_df['previous_qt_close'] = 0  # create column
        for j in range(len(curr_merged_df)):
            if j == 0:
                time_delta = 90  # used to approximate date of last quarter
                date_start = get_nearest_business_day(curr_merged_df['date'].iloc[j] - datetime.timedelta(time_delta))
                date_end = get_nearest_business_day(curr_merged_df['date'].iloc[j])
            else:
                date_start = get_nearest_business_day(curr_merged_df['date'].iloc[j-1])
                date_end = get_nearest_business_day(curr_merged_df['date'].iloc[j])

            curr_merged_df.loc[j, 'dividends_qt_sum'] = curr_daily_data_yf_df.loc[date_start:date_end,:]['dividends'].sum()
            try:
                curr_merged_df.loc[j, 'previous_qt_close'] = curr_daily_data_yf_df.loc[date_start,:]['close']
            except:
                curr_merged_df.loc[j, 'previous_qt_close'] = np.nan

        merged_df_out = merged_df_out.append(curr_merged_df, ignore_index=True, sort=False).drop(['index'], axis=1)

    return merged_df_out


merged_df = calc_dividends_in_qt(frames_dict['daily_data_yf'], merged_df)
```


Finally we have all the data that we need to calculate the ROA, ROE, and ROI. These are calculated below:

```python
merged_df['ROA'] = merged_df['netIncome_IS'] / merged_df['totalAssets_BS']
merged_df['ROE'] = merged_df['netIncome_IS'] / merged_df['totalStockholderEquity_BS']
merged_df['ROI'] = ((merged_df['close'] + merged_df['dividends_qt_sum']) / merged_df['previous_qt_close']) -1
```


# Part 2: Plotting Financial Ratios by Sector to Find Outperforming Stocks

Now let's plot the ROA, ROE, and ROI for stocks in the S&P 500 grouped by Sub-Industry. The reason I separate it by sub industry is because the balance sheet for companies in different sub industries will vary widely and will become uncomparable if we plot everything in a single scatterplot.

We'll create a distinct industry list to loop through after when we create the sub plots. We'll also define a color list that each sub plot will use to distinguish the different companies.

```python
distinct_sub_sector_list = list(set(merged_df['GICS Sub Industry'].tolist()))

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'teal', 'plum', 'coral', 'darksalmon', 'darkcyan', 'sienna', 'mediumpurple', 'fuchsia', 'slateblue', 'aqua', 'skyblue', 'peru', 'lime', 'lightblue']
```


Start the plotting:

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure()
sns.set(font_scale = 0.6)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i in range(1, 13):
    # do some industry analysis
    sub_sector_df = merged_df[merged_df['GICS Sub Industry'] == distinct_sub_sector_list[i-1]].reset_index()

    # define our figure
    #ax = Axes3D(fig)
    #ax = fig.add_subplot(2, 3, i)
    column_names = ['ROA', 'ROE', 'ROI']
    ax = fig.add_subplot(3, 4, i, projection='3d')

    distinct_symbol_list = list(set(sub_sector_df['ticker'].tolist()))
    for j in range(len(distinct_symbol_list)):
        symbol_id_df = sub_sector_df[sub_sector_df['ticker'] == distinct_symbol_list[j]].sort_values('date')

        # define the x, y & z-axis
        x = list(symbol_id_df[column_names[0]])
        y = list(symbol_id_df[column_names[1]])
        z = list(symbol_id_df[column_names[2]])

        all = ax.scatter(x[:-1], y[:-1], z[:-1], c=color_list[j], marker='o', label=distinct_symbol_list[j])
        last = ax.scatter(x[-1], y[-1], z[-1], c=color_list[j], marker='x', label='_nolegend_')

    # define axis labels
    ax.set_xlabel(column_names[0], fontdict={'fontsize': 8})
    ax.set_ylabel(column_names[1], fontdict={'fontsize': 8})
    ax.set_zlabel(column_names[2], fontdict={'fontsize': 8})
    ax.set_title(distinct_sub_sector_list[i-1], fontdict={'fontsize': 10}, fontweight='bold')

    plt.legend(numpoints=1, loc='lower left')
```




Here's the what the plot looks like:


[![](/assets/images/sector_fundamentals_plotting/ROA,ROE,ROI_industry_plot.JPG)](/assets/images/sector_fundamentals_plotting/ROA,ROE,ROI_industry_plot.JPG)


There are 4 points on each scatter that are the same color which shows that they are from the same company. You'll also notice that there are dots and x's in the scatter, the x's show the calculation using the most recent quarterly financials. 




