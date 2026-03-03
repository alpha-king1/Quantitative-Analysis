import numpy as np
import pandas as pd

class DatasetStructure():
    '''
        This class is to get the structure of each candle stick,
        after initialization and creation if instance, pass the poth of the file to the instance of this class

        A dataframe is created and it add columns like the body size, upper wick, lower wick, true range, total range and direction which is represented in -1, 0 1.

    '''

    def __init__(self, dataframe):
        self.dataframe = pd.read_csv(dataframe, parse_dates=["time"], index_col="time")

        self.dataframe['time'] = self.dataframe.index

        self.dataframe['body'] = (self.dataframe['c'] - self.dataframe['o']).abs()

        self.dataframe['upper_wick'] = self.dataframe['h'] - self.dataframe[['o', 'c']].max(axis=1)

        self.dataframe['lower_wick'] = self.dataframe[['o', 'c']].min(axis=1) - self.dataframe['l']

        self.dataframe['total_range'] = self.dataframe['h'] - self.dataframe['l']

        self.dataframe['true_range'] = np.maximum.reduce([self.dataframe['h'] - self.dataframe['l'],
                                                          self.dataframe['h'] - self.dataframe['c'].shift(periods=1).abs(),
                                                          self.dataframe['l'] - self.dataframe['c'].shift(periods=1).abs()
                                                          ])

        self.dataframe['direction'] = np.sign(self.dataframe['c'] - self.dataframe['o'])

    def show_data(self):
        '''
        thia shows the dataset which has been updated using this class.
        :return: returns the dataset
        '''
        data = self.dataframe
        return data

    def vol_regime(self):
        self.dataframe['atr_norm'] = self.dataframe['ATR_14'] / self.dataframe['ATR_14'].rolling(252).mean()

        self.dataframe['vol_regime'] = pd.qcut(
            self.dataframe['atr_norm'],
            q=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )

    def time_control(self):
        '''
            this method is used to set the hourly timeframe and which sessions each timeframe are being acted.
        :return:
        '''
        self.dataframe['hour'] = self.dataframe.index.hour

        conditions = [
            self.dataframe['hour'].between(0, 8),
            self.dataframe['hour'].between(8, 9),
            self.dataframe['hour'].between(9, 13),
            self.dataframe['hour'].between(13, 17),
            self.dataframe['hour'].between(17, 22),
            self.dataframe['hour'].between(22, 23)
        ]

        choices = [
            'asian',
            'asian/london',
            'london',
            'london/NY',
            'NY',
            'Closing'
        ]

        self.dataframe['sessions'] = np.select(conditions, choices, default='off')

    def volatility(self, number=20):
        '''
        this method is to add the volatility column and you can edit mean range by the number param
        :param number: the number is used to get the amounts used to  calculate the range or volatility.
        :return: nothing
        '''
        self.dataframe[f'volatility_{number}'] = self.dataframe['total_range'].rolling(number).mean()

    def daily_range(self, number = 20):
        '''
        this method is used to add the daily range column in a dataset and can only be used with timeframe below daily timeframe
        :input: the amount of dailyh roll up required
        :return: creates the daily high, daily low, daily range, and rolling daily range based on the number inputed
        '''
        daily = self.dataframe.resample('D')

        self.dataframe['daily_high'] = daily['h'].transform('max')
        self.dataframe['daily_low'] = daily['l'].transform('min')

        self.dataframe['daily_range'] = self.dataframe['daily_high'] - self.dataframe['daily_low']
        self.dataframe[f'rolling_daily_range_{number}'] = self.dataframe['daily_range'].rolling(number).mean()

    def savedata(self, name):
        '''
        this method saved the dataset which has being edited into a csv file and save in system storage.
        :param name: add the name you want to save the dataset as, dont include .csv, as this has being done already.
        :return:
        '''
        self.dataframe.to_csv(f'{name}.csv')

    def ATR(self, number = 20):
        '''
        this is to get the atr of the dataframe, and user can input the numbers of data to use to get the atr
        :param number: input number of rolling true range needed, or just one number
        :return: creates a column where with ATR(number)
        '''
        for n in [number]:
            self.dataframe[f'ATR_{n}'] = self.dataframe['true_range'].rolling(n).mean()
