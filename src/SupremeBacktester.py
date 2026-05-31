import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tpqoa

class SupremeBacktester(tpqoa.tpqoa):
    """
        This is a very comprehensive script, used to backtest across alot of strategies, it just runs and give results
        all the user does is to input a comprehensive timeseries dataset that with the following columns:o, h, l, and c
    """
    def __init__(self, data, conf_file):
        super().__init__(conf_file)
        self.data = data

    def bullish_ob(self, displacement_mult=2.0, forward_window=10):
        """
        df: DataFrame with ['o', 'h', 'l', 'c', 'ATR_14']
        type: either bullish or bearish
        displacement_mult: How much stronger the move must be than the OB candle to count.
        forward_window: How many candles to look ahead for return after a hit.
        """
        obs = []
        active_zones = []

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
                    if curr['l'] <= zone['top'] <= curr['c'] and i - zone['created_at'] > 10:
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

    def bearish_ob(self, displacement_mult=2.0, forward_window=10):
        """
                df: DataFrame with ['o', 'h', 'l', 'c', 'ATR_14']
                type: either bullish or bearish
                displacement_mult: How much stronger the move must be than the OB candle to count.
                forward_window: How many candles to look ahead for return after a hit.
                """
        obs = []
        active_zones = []
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
                    if curr['h'] >= zone['bottom'] >= curr['c'] and i - zone['created_at'] > 10:
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


