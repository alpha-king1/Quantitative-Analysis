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
            (data['upper_wick'] >= 0.8 * data['total_range'])  # upperweak >= 70% of total range
            &
            (data['upper_wick'] / data['true_range'] >= 0.8),
            1, None)
        return data.dropna()

    def LWR(self):
        data = self.data.dropna()
        data['lower_wick_rejection'] = np.where(
            (data['lower_wick'] >= 0.8 * data['total_range'])  # lowerweak >= 70% of total range
            &
            (data['lower_wick'] / data['true_range'] >= 0.8),
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
        window = 10
        data['total_movement'] = data['true_range'].rolling(window).sum()

        data['net_movement'] = (
                data['c'] - data['c'].shift(window)
        ).abs()
        data['directionality'] = (
                data['net_movement'] / data['total_movement']
        )
        data['trending'] = data['directionality'] >= 0.15

        return data[data['trending'] == True]

    def analyze_gold_obs(self, type, displacement_mult=2.0, forward_window=10):
        """
        df: DataFrame with ['o', 'h', 'l', 'c', 'ATR_14']
        type: either bullish or bearish
        displacement_mult: How much stronger the move must be than the OB candle to count.
        forward_window: How many candles to look ahead for return after a hit.
        """
        obs = []
        active_zones = []

        if type == 'bullish':
            for i in range(1, len(self.data) - 1):
                curr = self.data.iloc[i]
                prev = self.data.iloc[i - 1]
                # --- 1. IDENTIFY NEW ORDER BLOCKS ---
                # Bullish OB: Last Bearish candle before a strong Bullish move
                if curr['c'] > curr['o'] and (curr['c'] - curr['o']) > (prev['h'] - prev['l']) * displacement_mult and \
                        prev['body'] < prev['ATR_14']:
                    if prev['c'] < prev['o']:
                        active_zones.append({
                            'type': 'Bullish',
                            'top': prev['h'],
                            'bottom': prev['l'],
                            'created_at': i,
                            'created_time': self.data['time'].iloc[i],
                            'status': 'Active'
                        })

                for zone in active_zones:
                    if zone['status'] != 'Active': continue

                    # Check for INVALIDATION (Body Close through zone)
                    if zone['type'] == 'Bullish' and curr['c'] < zone['bottom']:
                        zone['status'] = 'Invalidated'
                        continue
                    hit = False
                    if zone['type'] == 'Bullish':
                        # Low enters zone, but Close stays above bottom
                        if curr['l'] <= zone['top'] and curr['c'] >= zone['top'] and i - zone['created_at'] > 10:
                            hit = True
                    if hit:
                        # Capture the Return
                        future_idx = min(i + forward_window, len(self.data) - 1)
                        future_price = self.data.iloc[future_idx]['c']
                        ret = ((future_price - curr['c']) / curr['c']) if zone['type'] == 'Bullish' else (
                                    curr['c'] - future_price)

                        obs.append({
                            'Type': zone['type'],
                            'Created_At': self.data.index[zone['created_at']],
                            'time': self.data['time'].iloc[i],
                            'vol_regime': self.data['vol_regime'].iloc[i],
                            'sessions': self.data['sessions'].iloc[i],
                            'Hit_At': self.data.index[i],
                            'Return': ret,
                            'Zone_Top': zone['top'],
                            'Zone_Bottom': zone['bottom']
                        })
                        zone['status'] = 'Mitigated'  # Mark as done

            return pd.DataFrame(obs)

        elif type == 'bearish':
            for i in range(1, len(self.data) - 1):
                curr = self.data.iloc[i]
                prev = self.data.iloc[i - 1]
                if curr['c'] < curr['o'] and (curr['o'] - curr['c']) > (prev['h'] - prev['l']) * displacement_mult:
                    if prev['c'] > prev['o']:
                        active_zones.append({
                            'type': 'Bearish',
                            'top': prev['h'],
                            'bottom': prev['l'],
                            'created_time': self.data['time'].iloc[i],
                            'created_at': i,
                            'status': 'Active'
                        })

                # --- 2. INSPECT ACTIVE ZONES ---
                for zone in active_zones:
                    if zone['status'] != 'Active': continue
                    if zone['type'] == 'Bearish' and curr['c'] > zone['top']:
                        zone['status'] = 'Invalidated'
                        continue

                    hit = False
                    if zone['type'] == 'Bearish':  # Bearish
                        if curr['h'] >= zone['bottom'] and curr['c'] <= zone['bottom'] and i - zone['created_at'] > 10:
                            hit = True

                    if hit:
                        future_idx = min(i + forward_window, len(self.data) - 1)
                        future_price = self.data.iloc[future_idx]['c']
                        ret = ((future_price - curr['c']) / curr['c'])

                        obs.append({
                            'Type': zone['type'],
                            'Created_At': self.data.index[zone['created_at']],
                            'time': self.data['time'].iloc[i],
                            'vol_regime': self.data['vol_regime'].iloc[i],
                            'sessions': self.data['sessions'].iloc[i],
                            'Hit_At': self.data.index[i],
                            'Return': ret,
                            'Zone_Top': zone['top'],
                            'Zone_Bottom': zone['bottom']
                        })
                        zone['status'] = 'Mitigated'  # Mark as done

            return pd.DataFrame(obs)

    def show_data(self):
        return self.data
