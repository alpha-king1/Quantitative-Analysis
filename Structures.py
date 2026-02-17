import pandas as pd
import numpy as np

class Structures:
    def __init__(self, data):
        self.data = data.copy()

    def impulse_candle(self):
        data = self.data.copy()
        data['impulse_candle'] = np.where((data['body'] >= 1.6 * data['ATR_14']) &
                                          (data['body'] / data['total_range'] > 0.7)
                                          , 1, None)
        return data.dropna()

    def bearish_impulse(self):
        data = self.impulse_candle()
        data['bearish_impulse'] = np.where(data['direction'] == -1,
                                           True, None)
        return data.dropna()

    def bullish_impulse(self):
        data = self.impulse_candle()
        data['bullish_impulse'] = np.where(data['direction'] == 1,
                                           True, None)
        return data.dropna()

    def UWR(self):
        data = self.data.copy()

        data['upper_wick_rejection'] = np.where(
            (data['upper_wick'] >= 0.7 * data['total_range'])  # upperweak >= 70% of total range
            &
            (data['upper_wick'] / data['true_range'] >= 0.7),
            1, None)
        return data.dropna()

    def LWR(self):
        data = self.data.dropna()
        data['lower_wick_rejection'] = np.where(
            (data['lower_wick'] >= 0.7 * data['total_range'])  # lowerweak >= 70% of total range
            &
            (data['lower_wick'] / data['true_range'] >= 0.7),
            1, None)
        return data.dropna()

    def compression(self):
        data = self.data.copy()
        data['small_candle'] = data['total_range'] < 0.8 * data['ATR_14']
        window = 10
        data['small_count'] = (data['small_candle'].rolling(window).sum())
        data['compression'] = data['small_count'] >= 5

        return data

    def ranging(self):
        data = self.data.copy()
        window = 20
        data['total_movement'] = data['true_range'].rolling(window).sum()
        data['net_movement'] = (
                data['c'] - data['c'].shift(window)
        ).abs()
        data['directionality'] = (
                data['net_movement'] / data['total_movement']
        )
        data['ranging'] = data['directionality'] < 0.2

        return data

    def trending(self):
        data = self.data
        window = 20
        data['total_movement'] = data['true_range'].rolling(window).sum()

        data['net_movement'] = (
                data['c'] - data['c'].shift(window)
        ).abs()
        data['directionality'] = (
                data['net_movement'] / data['total_movement']
        )
        data['trending'] = data['directionality'] >= 0.2

        return data

    def show_data(self):
        return self.data
