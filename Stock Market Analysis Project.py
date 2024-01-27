'''
DS 2500 Stock Market Analysis on Volatility, Price, and Sentiment
In this program we will analyze the net change in the underlying price of a stock based
off of the following criteria using the Yahoo Finance API to get historical data and
the Reddit API to scrape post titles for sentiment analysis
- Volatility
- Sharpe Ratio
- Market Returns
- Reddit Sentiment (WSB Subreddit)
- Balanced Portfolio (Berkshire Hathaway Portfolio)
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import numpy as np
import plotly.graph_objects as go
import praw
from Keys import client_id, client_secret, user_agent,username,password
import yfinance as yf
import scipy.stats as stats

class Reddit_Analysis:
    def __init__(self):
        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent,
                                  username=username, password=password)
        self.analyzer = SentimentIntensityAnalyzer()
        self.new_titles = []
        self.top_titles = []
        self.tickers = []
        self.clean_list = []
        self.top_posts_daily = []
    def reddit_praw(self):
        '''Function: reddit_praw
        param:self
        :return: a list of the top rate posts for the given time frame and limited amount of posts given an input
        in this case 365 days or 1 year
        '''
        subreddit = self.reddit.subreddit('wallstreetbets')
        top_subreddit = subreddit.top(time_filter="year",limit=365)
        for submission in top_subreddit:
            title = submission.title
            title_words = title.split()
            self.top_titles.append(title_words)
        return print(self.top_titles)

    def clean_reddit(self):
        '''Func: clean_reddit
        param: self
        :return:a clean list of reddit titles without any emojis
        '''
        '''Chat GPT: how to remove emojis from list'''
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)

        for title in self.top_titles:
            clean_title = emoji_pattern.sub('', ' '.join(title))
            self.clean_list.append(clean_title)

        return print(self.clean_list)


    def title_analysis(self):
        '''Func: title_analysis
        :param: self
        Does: uses sent analysis from nltk and calcs polarity scores using vader lexicon
        :return: a list of sentiment score dictionaries that include neg,neu,pos,and compound
        '''
        self.sent_score = []
        for title in self.clean_list:
            sentiment_score = self.analyzer.polarity_scores(title)
            self.sent_score.append(sentiment_score)
        return self.sent_score
    def plot_sentiment_vs_time(self,alpha=.05):
        '''Func: plot_sentiment_vs_time
        :param alpha:
        :return: An output from the results of the regression testing and plots of the values
        '''
        sentiment_df = pd.DataFrame(self.sent_score, columns=['compound', 'neg', 'neu', 'pos'])
        sentiment_df = sentiment_df.dropna(subset=['compound', 'neu', 'neg', 'pos'])
        sentiment_df['post_number'] = range(1, len(sentiment_df) + 1)

        plt.figure(figsize=(10, 6))
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['compound'], label='Compound',
                    line_kws={'color':'darkblue'},scatter_kws={'color':'blue'})
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['neg'], label='Negative',
                    line_kws={'color':'firebrick'}, scatter_kws={'color':'red'})
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['neu'], label='Neutral',
                    line_kws={'color':'coral' }, scatter_kws={'color':'orange'})
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['pos'], label='Positive',
                    line_kws={'color':'darkgreen'},scatter_kws={'color':'green'})
        plt.xlabel('Days')
        plt.ylabel('Sentiment Score (Pos,Neg,Neu)')
        plt.title('Sentiment Score Trend')
        plt.legend()
        plt.show()
        # Calculate correlation coefficients
        corr_compound, _ = stats.pearsonr(sentiment_df['post_number'], sentiment_df['compound'])
        corr_neg,p_value_neg = stats.pearsonr(sentiment_df['post_number'], sentiment_df['neg'])
        corr_neu,p_value_neu = stats.pearsonr(sentiment_df['post_number'], sentiment_df['neu'])
        corr_pos,p_value_pos = stats.pearsonr(sentiment_df['post_number'], sentiment_df['pos'])
        if p_value_pos < alpha:
            interpretation_pos = "There is a significant correlation between Portfolio_Return and Positive sentiments scores"
        else:
            interpretation_pos = "There is not a significant correlation between Portfolio_Return and Positive sentiment scores"

        if p_value_neg < alpha:
            interpretation_neg = "There is a significant correlation between negative sentiment scores and Portfolio_Return"
        else:
            interpretation_neg = "There is not a significant correlation between negative sentiment scores and Portfolio_Return."

        if p_value_neu < alpha:
            interpretation_neu = "There is a significant correlation between neutral sentiment scores and Portfolio_Return."
        else:
            interpretation_neu = "There is not a significant correlation between neutral sentiment scores and Portfolio_Return."
        return print('\nResults for sent vs time:', '\nP-Value Pos:', p_value_pos, interpretation_pos,
                     '\nP-Value Neu:',
                     p_value_neu, interpretation_neu, '\nP-Value Neg:', p_value_neg, interpretation_neg)

    def combined_df(self, sentiment_scores, final_return):
        '''Function: combined_df
        :param sentiment_scores:
        :param final_return:
        Does: combines the data frame that contains the sentiment scores and final returns dataframe
        :return: a concatenated dataframe
        '''
        sentiment_df = pd.DataFrame(sentiment_scores, columns=['compound', 'neg', 'neu', 'pos'])
        # reset the index to match with sentiment_df
        final_return.reset_index(drop=True, inplace=True)

        # Combine sentiment scores and portfolio returns into a single DataFrame
        combined_df = pd.concat([sentiment_df, final_return], axis=1)
        # Clean any Nan or Inf avlues
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna(subset=['Portfolio_Return','Volatility','Sharpe_Ratio','compound','neu','neg','pos'])
        return combined_df

    def plot_sentiment_vs_returns(self, combined_df,sentiment_scores, final_return,alpha=.05):
        '''Function: plot sentiment vs returns
        :param combined_df:
        :param sentiment_scores:
        :param final_return:
        :param alpha:
        :return: plot and analysis of the t test
        '''
        corr_compound, _ = stats.pearsonr(combined_df['Portfolio_Return'], combined_df['compound'])
        corr_neg,p_value_neg = stats.pearsonr(combined_df['Portfolio_Return'], combined_df['neg'])
        corr_neu,p_value_neu = stats.pearsonr(combined_df['Portfolio_Return'], combined_df['neu'])
        corr_pos,p_value_pos = stats.pearsonr(combined_df['Portfolio_Return'], combined_df['pos'])
        if p_value_pos < alpha:
            interpretation_pos = "There is a significant correlation between Portfolio_Return and Positive sentiments scores"
        else:
            interpretation_pos = "There is not a significant correlation between Portfolio_Return and Positive sentiment scores"

        if p_value_neg < alpha:
            interpretation_neg = "There is a significant correlation between negative sentiment scores and Portfolio_Return"
        else:
            interpretation_neg = "There is not a significant correlation between negative sentiment scores and Portfolio_Return."

        if p_value_neu < alpha:
            interpretation_neu = "There is a significant correlation between neutral sentiment scores and Portfolio_Return."
        else:
            interpretation_neu = "There is not a significant correlation between neutral sentiment scores and Portfolio_Return."
        # Plot correlation
        sns.regplot(data=combined_df, x='compound', y='Portfolio_Return')
        sns.regplot(data = combined_df,x='neg', y='Portfolio_Return', label='Negative',
                    line_kws={'color': 'firebrick'}, scatter_kws={'color': 'red'})
        sns.regplot(data=combined_df,x='neu', y='Portfolio_Return', label='Neutral', line_kws={'color': 'coral'},
                    scatter_kws={'color': 'orange'})
        sns.regplot(data=combined_df,x='pos', y='Portfolio_Return', label='Positive',
                    line_kws={'color': 'darkgreen'}, scatter_kws={'color': 'green'})
        plt.xlabel('Sentiment Score (Neg,Neu,Pos,Compound)')
        plt.ylabel('Portfolio Return')
        plt.title('Sentiment Scores vs Portfolio Returns')
        plt.legend()
        plt.show()
        return print('\nResults for sent vs returns:', '\nP-Value Pos:', p_value_pos, interpretation_pos,
                     '\nP-Value Neu:',
                     p_value_neu, interpretation_neu, '\nP-Value Neg:', p_value_neg, interpretation_neg)
    def plot_sentiment_vs_sharpe(self, combined_df,sentiment_scores, final_return,alpha=.05):
        '''Function: plot_sentiment_vs_sharpe
        :param combined_df:
        :param sentiment_scores:
        :param final_return:
        :param alpha:
        :return: plots of sentiment vs sharpe ratio with returned t-test results
        '''
        corr_neg,p_value_neg = stats.pearsonr(combined_df['Sharpe_Ratio'], combined_df['neg'])
        corr_neu,p_value_neu = stats.pearsonr(combined_df['Sharpe_Ratio'], combined_df['neu'])
        corr_pos,p_value_pos = stats.pearsonr(combined_df['Sharpe_Ratio'], combined_df['pos'])
        if p_value_pos < alpha:
            interpretation_pos = "There is a significant correlation between Sharpe_Ratio and Positive sentiments scores"
        else:
            interpretation_pos = "There is not a significant correlation between Sharpe_Ratio and Positive sentiment scores"

        if p_value_neg < alpha:
            interpretation_neg = "There is a significant correlation between negative sentiment scores and Sharpe_Ratio"
        else:
            interpretation_neg = "There is not a significant correlation between negative sentiment scores and Sharpe_Ratio."

        if p_value_neu < alpha:
            interpretation_neu = "There is a significant correlation between neutral sentiment scores and Sharpe_Ratio."
        else:
            interpretation_neu = "There is not a significant correlation between neutral sentiment scores and Sharpe_Ratio."
        # Plot correlation
        sns.regplot(data=combined_df, x='compound', y='Sharpe_Ratio')
        sns.regplot(data = combined_df,x='neg', y='Sharpe_Ratio', label='Negative',
                    line_kws={'color': 'firebrick'}, scatter_kws={'color': 'red'})
        sns.regplot(data=combined_df,x='neu', y='Sharpe_Ratio', label='Neutral', line_kws={'color': 'coral'},
                    scatter_kws={'color': 'orange'})
        sns.regplot(data=combined_df,x='pos', y='Sharpe_Ratio', label='Positive',
                    line_kws={'color': 'darkgreen'}, scatter_kws={'color': 'green'})
        plt.xlabel('Sentiment Score (Neg,Neu,Pos,Compound)')
        plt.ylabel('Sharpe_Ratio')
        plt.title('Sentiment Scores vs Sharpe_Ratio')
        plt.legend()
        plt.show()
        return print('\nResults for sent vs sharpe:', '\nP-Value Pos:', p_value_pos, interpretation_pos, '\nP-Value Neu:',
                     p_value_neu, interpretation_neu,'\nP-Value Neg:', p_value_neg, interpretation_neg)

    def plot_sentiment_vs_volatility(self, combined_df,sentiment_scores, final_return,alpha=.05):
        '''Function: plot_sentiment_vs_volatility
        :param combined_df:
        :param sentiment_scores:
        :param final_return:
        :param alpha:
        :return: plots of sentiment vs volatility with returned t-test results
        '''
        corr_pos, _ = stats.pearsonr(combined_df['Volatility'], combined_df['compound'])
        corr_pos,p_value_pos = stats.pearsonr(combined_df['Volatility'], combined_df['pos'])
        corr_neu,p_value_neu = stats.pearsonr(combined_df['Volatility'], combined_df['neu'])
        corr_neg,p_value_neg = stats.pearsonr(combined_df['Volatility'], combined_df['neg'])
        if p_value_pos < alpha:
            interpretation_pos = "There is a significant correlation between volatility and Positive sentiments scores"
        else:
            interpretation_pos = "There is not a significant correlation between volatility and Positive sentiment scores"

        if p_value_neg < alpha:
            interpretation_neg = "There is a significant correlation between negative sentiment scores and volatility."
        else:
            interpretation_neg = "There is not a significant correlation between negative sentiment scores and volatility."

        if p_value_neu < alpha:
            interpretation_neu = "There is a significant correlation between neutral sentiment scores and volatility."
        else:
            interpretation_neu = "There is not a significant correlation between neutral sentiment scores and volatility."
        # Plot correlation
        sns.regplot(data=combined_df, x='compound', y='Volatility')
        sns.regplot(data = combined_df,x='neg', y='Volatility', label='Negative',
                    line_kws={'color': 'firebrick'}, scatter_kws={'color': 'red'})
        sns.regplot(data=combined_df,x='neu', y='Volatility', label='Neutral', line_kws={'color': 'coral'},
                    scatter_kws={'color': 'orange'})
        sns.regplot(data=combined_df,x='pos', y='Volatility', label='Positive',
                    line_kws={'color': 'darkgreen'}, scatter_kws={'color': 'green'})
        plt.xlabel('Sentiment Score (Neg,Neu,Pos,Compound)')
        plt.ylabel('Volatility')
        plt.title('Sentiment Scores vs Volatility')
        plt.legend()
        plt.show()
        return print('\nResults for sent vs vol:','\nP-Value Pos:', p_value_pos, interpretation_pos, '\nP-Value Neu:',
                     p_value_neu, interpretation_neu, '\nP-Value Neg:', p_value_neg, interpretation_neg)


class Yahoo:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.volatility = None
        self.df = None
        self.var = None
        self.combined_df = pd.DataFrame()

    def get_historical_data(self):
        """
            Function: get_historical_data
            Parameter: ticker symbol, start date, and end date
            Return: Corresponding stock data from Yahoo Finance Database
            Does: Returns stock data as a dataframe
        """
        self.stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.stock_data.reset_index(inplace=True)

    def print_data(self):
        '''
        :return: print stock data
        '''
        print("\nStock:", self.ticker, self.stock_data)

    def get_data_as_dataframe(self):
        '''
        :return: returns the yahoo finance as dataframe
        '''
        return pd.DataFrame(self.stock_data)

    def candlestick(self):
        '''
        :return: candlestick visualization using historical data using 'Date', 'Open', 'High', 'Low', 'Close'
        '''
        fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
                                             open=self.df['Open'], high=self.df['High'],
                                             low=self.df['Low'], close=self.df['Close'])
                              ])
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()


    def create_closing(self):
        '''
        :return: creates the initial column to merge the closing data
        '''
        # Initialize the first pandas dataframe that takes in all the closing prices of each stock in the portfolio
        closing_df = pd.DataFrame()
        dataframe = pd.DataFrame(self.stock_data)
        # creating the first column for the pandas to help merge each of the pd df based on the date column
        closing_df['Date'] = dataframe['Date']
        return closing_df

    def close_column(self, column_name):
        '''
        :param column_name:
        :return: returns pandas dataframe of only the column that are specified
        '''
        dataframe = pd.DataFrame(self.stock_data)
        df1_close = dataframe[column_name].copy()
        return df1_close

    def merge_dataframes(self, df1, df2):
        '''
        :param df1: create_closing dataframe with just the date to merge data on
        :param df2: close_column where the dataframe is the columns specified
        :return: merged dataframe on the date merged to the left to return 'Date' on the left
        '''
        df1 = pd.merge(df1, df2, on="Date", how="left")
        return df1

    def calculate_volatility(self, final_df, dataframe):
        '''
        :param final_df: the final dataframe with where we eventually append volatility, sharpe ratio, and portfolio return
        :param dataframe: volatility series
        :return: final dataframe including column volatility
        '''
        returns = dataframe.iloc[:, 1:].pct_change()
        volatility = returns.std(axis=1)
        final_df['Volatility'] = volatility
        return final_df

    def calculate_return(self, port_data):
        '''
        :param port_data: pandas dataframe including all of the historical data for a specific stock
        :return: pandas datafarme with all individual stock percentage return
        '''
        port_return = pd.DataFrame()
        port_return['Date'] = port_data['Date']
        for stock in port_data.columns[1:]:
            port_return[stock + '_Return'] = port_data[stock].pct_change().fillna(0)
        return port_return


    def portfolio_return(self, dataframe, weights):
        '''
        :param dataframe: pandas datafarme with all individual stock percentage return
        :param weights: individual weights for each stock (in this case 1/20 )
        :return: entire portfolio stock return pandas dataframe
        '''
        final_return = pd.DataFrame()
        final_return['Date'] = dataframe['Date']
        final_return['Portfolio_Return'] = dataframe.iloc[:, 1:].mul(weights).sum(axis=1)
        return final_return

    def calculate_portfolio_std(self, dataframe, weights):
        """
        Calculates the portfolio standard deviation using a pandas DataFrame of different stocks and their returns.
        Args: dataframe (pandas.DataFrame): DataFrame containing stock returns.
        Each column represents a stock and each row represents a return value.
        Returns: float: Portfolio standard deviation.
        """
        # Calculate the covariance matrix
        covariance_matrix = dataframe.cov(numeric_only=True)

        # Convert weights to NumPy array
        weights = np.array(weights)

        # Calculate the portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

        # Calculate the portfolio standard deviation
        portfolio_std = np.sqrt(portfolio_variance)
        return portfolio_std

    def sharpe_ratio(self, dataframe, rf, std):
        '''
        :param dataframe: entire portfolio stock return pandas dataframe
        :param rf: risk free rate
        :param std: portfolio standard deviation
        :return: sharpe ratio pandas datframe
        '''
        #dataframe['Sharpe_Ratio'] = dataframe['Portfolio_Return'].apply(lambda x: (x - rf) / std)
        excess_return = dataframe['Portfolio_Return'] - rf
        sharpe_ratio = excess_return / std
        dataframe['Sharpe_Ratio'] = sharpe_ratio
        return dataframe


def main():
    # Create the Berkshire Hathaway Portfolio by combining stock data for multiple datasets
    # Import all of the assets in Berkshire Porfolio into one pandas data frame of all of the closing prices
    # Merge the stock data into a single DataFrame based on the 'Date' column

    # Yahoo Finance API parameters
    tickers = ['TSM', 'V', 'MA', 'PG', 'KO', 'UPS', 'AXP', 'C', 'MMC', 'MCK', 'GM', 'OXY', 'BK', 'HPQ', 'MKL', 'GL',
               'ALLY', 'JEF', 'RH', 'LPX']
    start_date = "2022-06-19"
    end_date = "2023-06-19"

    # Initial create closing pandas dataframe with all dates to eventually merge following data
    data_fetcher = Yahoo(tickers[0], start_date, end_date)
    data_fetcher.get_historical_data()
    closing_df = data_fetcher.create_closing()
    print(closing_df)

    # Create Dataframe with 'Close' Prices for each stock by date
    for ticker in tickers:
        data_fetcher = Yahoo(ticker, start_date, end_date)
        data_fetcher.get_historical_data()
        # data_fetcher.print_data()
        # data = data_fetcher.get_data_as_dataframe()
        df_close = data_fetcher.close_column(["Date", "Close"])
        df_close = df_close.rename(columns={"Close": ticker})
        closing_df = data_fetcher.merge_dataframes(closing_df, df_close)
        # closing_df = closing_df.merge(df_close, on="Date", how="left")
    print(closing_df)

    # weight of each stock in portfolio
    weights = [1 / 20] * 20
    print(weights)

    # Calculation of percentage return data
    closing_copy = closing_df.copy()
    port_return = data_fetcher.calculate_return(closing_copy)
    print(port_return)

    # Calculation of entire Portfolio return
    final_return = (data_fetcher.portfolio_return(port_return, weights))
    print(final_return)

    # Calculation of standard deviation to eventually be used to calculate sharpe ratio
    std = data_fetcher.calculate_portfolio_std(port_return, weights)
    print(std)

    rf = .00174 # Risk free rate based off the 3 month treasury yield divided by 3 to get the 1 month
    # Calculation of sharpre ratio
    final_return = data_fetcher.sharpe_ratio(final_return, rf, std)
    print(final_return)

    # Calculation of volatility
    final_return = data_fetcher.calculate_volatility(final_return, closing_copy)
    print(final_return)

    # Initialize Reddit Object
    reddit_analysis = Reddit_Analysis()
    # Use reddit Praw to scrape posts
    reddit_analysis.reddit_praw()
    # Clean the posts titles
    reddit_analysis.clean_reddit()
    # Create Sentiment Analysis Object
    sent_scores = reddit_analysis.title_analysis()
    # Show Results
    print(sent_scores)
    # Combine the sentiment and returns into one dataframe for plotting
    combined_df = reddit_analysis.combined_df(sent_scores,final_return)
    print(combined_df)
    # Plot all analysis + Analyze the outputs of correlation using regression testing
    reddit_analysis.plot_sentiment_vs_time()
    reddit_analysis.plot_sentiment_vs_returns(combined_df,sent_scores, final_return)
    reddit_analysis.plot_sentiment_vs_sharpe(combined_df,sent_scores,final_return)
    reddit_analysis.plot_sentiment_vs_volatility(combined_df, sent_scores, final_return)




if __name__ == '__main__':
    main()

'''Pearson R Output Results 
Sample Size 365 or 1 Year, With an Alpha of .05 or 5% 

Results for sent vs time: 
P-Value Pos: 0.4030400843624842 There is not a significant correlation between Portfolio_Return and Positive sentiment scores 
P-Value Neu: 0.9308111406502669 There is not a significant correlation between neutral sentiment scores and Portfolio_Return. 
P-Value Neg: 0.520356043179645 There is not a significant correlation between negative sentiment scores and Portfolio_Return.

Results for sent vs returns: 
P-Value Pos: 0.16832724870682353 There is not a significant correlation between Portfolio_Return and Positive sentiment scores 
P-Value Neu: 0.4332210806574682 There is not a significant correlation between neutral sentiment scores and Portfolio_Return. 
P-Value Neg: 0.5141758786408769 There is not a significant correlation between negative sentiment scores and Portfolio_Return.

Results for sent vs sharpe: 
P-Value Pos: 0.16832724870682353 There is not a significant correlation between Sharpe_Ratio and Positive sentiment scores 
P-Value Neu: 0.4332210806574682 There is not a significant correlation between neutral sentiment scores and Sharpe_Ratio. 
P-Value Neg: 0.5141758786408769 There is not a significant correlation between negative sentiment scores and Sharpe_Ratio.

Results for sent vs vol: 
P-Value Pos: 0.4898319543887323 There is not a significant correlation between volatility and Positive sentiment scores 
P-Value Neu: 0.24152772543470266 There is not a significant correlation between neutral sentiment scores and volatility. 
P-Value Neg: 0.04310127995349184 There is a significant correlation between negative sentiment scores and volatility.
'''
