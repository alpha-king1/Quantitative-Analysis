import tpqoa
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import warnings


class ConTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, risk_percentage):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.Timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.last_bar = None
        self.units = 0
        self.signal = {}  #contains strategy_id, timestamp, symbol, direction, entry_type, sl, tprofit, time_stop, ml_probability, ml_regime_label, confidence_score, valid_until, context_tags,reason_code
        self.position = 0
        self.trade_created_at = 0
        self.summary = self.get_account_summary()
        self.balance = self.summary['balance']
        self.riskable = float(self.balance) * (risk_percentage/100)
        self.pl = self.summary['pl']
        self.profits = [] # NEW

        # unit = amount risk/sl distance
        # amount risk = risk_percentage * balance
        # sl distance = entryprice * 0.005

        #*****************add strategy-specific attributes here******************
        self.obs = []
        self.active_zones = []
        #************************************************************************

    def get_most_recent(self, days = 5):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        now = now.replace(second=0, microsecond=0)  # floor to the hour
        past = now - timedelta(days = days)
        df = self.get_history(instrument = self.instrument,
                              start = past,
                              end = now,
                              granularity = 'M1',
                              price = "M")
        # df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]

        self.raw_data = df.copy()
        self.dataset_structure()
        for i in range(1, len(self.raw_data)):
            self.bullish_obs(i, take_trade=False)
            self.bearish_obs(i, take_trade=False)

        self.last_bar = self.raw_data.index[-1]

    def dataset_structure(self):
        self.raw_data['body'] = (self.raw_data['c'] - self.raw_data['o']).abs()
        self.raw_data['ATR_14'] = self.raw_data['c'].rolling(14).mean()
        self.raw_data['time'] = self.raw_data.index
        self.raw_data['atr_norm'] = self.raw_data['ATR_14'] / self.raw_data['ATR_14'].rolling(252).mean()

        self.raw_data['vol_regime'] = pd.qcut(
            self.raw_data['atr_norm'],
            q=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High'])

        self.raw_data['hour'] = self.raw_data.index.hour

        conditions = [
            self.raw_data['hour'].between(0, 8),
            self.raw_data['hour'].between(8, 9),
            self.raw_data['hour'].between(9, 13),
            self.raw_data['hour'].between(13, 17),
            self.raw_data['hour'].between(17, 22),
            self.raw_data['hour'].between(22, 23)
        ]

        choices = [
            'asian',
            'asian/london',
            'london',
            'london/NY',
            'NY',
            'Closing'
        ]

        self.raw_data['sessions'] = np.select(conditions, choices, default='off')

    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ")

        # collect and store tick data
        recent_tick = pd.to_datetime(time).replace(tzinfo=None)
        df = pd.DataFrame({self.instrument: (ask + bid) / 2},
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df]) # new with pd.concat()

        # if a time longer than the bar_lenght has elapsed between last full bar and the most recent tick
        if recent_tick.floor('min') - self.last_bar >= self.bar_length:
            self.resample_and_join()
            bullish_signal = self.bullish_obs(len(self.raw_data) -1)
            bearish_signal = self.bearish_obs(len(self.raw_data) -1)

            if bullish_signal or bearish_signal:
                self.execute_trades()

            if len(self.raw_data) - self.trade_created_at > 20 and self.position == 1:
                print('going neutral')
                self.execute_trades('neutral')

    def resample_and_join(self):
        candle_data = pd.DataFrame({
            'c': self.tick_data['XAU_USD'].iloc[-1],
            'o': self.tick_data['XAU_USD'].iloc[0],
            'h': self.tick_data['XAU_USD'].max(),
            'l': self.tick_data['XAU_USD'].min(),
            'hour': self.tick_data.index.hour[-1]
        }, index=[self.tick_data.index[-1]])   # scalar index, not full range

        self.raw_data = pd.concat([self.raw_data, candle_data])
        self.tick_data = pd.DataFrame(columns=self.tick_data.columns)  # clear — no overlap
        self.dataset_structure()
        self.last_bar = self.raw_data.index[-1].floor('min')

    def bullish_obs(self, i, displacement_mult=2.0, take_trade = True):
        """
        df: DataFrame with ['o', 'h', 'l', 'c', 'ATR_14']
        type: either bullish or bearish
        displacement_mult: How much stronger the move must be than the OB candle to count.
        forward_window: How many candles to look ahead for return after a hit.
        """
        curr = self.raw_data.iloc[i]
        prev = self.raw_data.iloc[i-1]

        if (curr['c'] > curr['o'] and (curr['body']) > (prev['body']) * displacement_mult and \
                prev['body'] < prev['ATR_14']):
            if prev['c'] < prev['o']:
                self.active_zones.append({
                    'type': 'Bullish',
                    'top': prev['o'],
                    'bottom': prev['c'],
                    'created_at': i,
                    'created_time': self.raw_data['time'].iloc[i],
                    'status': 'Active'
                })

        for zone in self.active_zones:
            if zone['status'] != 'Active': continue

            # Check for INVALIDATION (Body Close through zone)
            if zone['type'] == 'Bullish' and curr['c'] < zone['bottom']:
                zone['status'] = 'Invalidated'
                continue
            hit = False
            if zone['type'] == 'Bullish':
                # Low crosses mid of order block
                if curr['l'] <= ((zone['top'] + zone['bottom'])/2) and i - zone['created_at'] > 10:
                    hit = True
            if hit and take_trade:
                self.signal = {
                    'status': 'active',
                    'position': 1,
                    'tp': prev['c'] + 100,
                    'sl': 25,
                    'entry_type': "market",
                    'ml_prob':0.6,
                    'i': i
                }
                zone['status'] = 'Mitigated'  # Mark as done

                return True

        return False

    def bearish_obs(self, i, displacement_mult=2.0, take_trade = True):
        """
            data: data with ['o', 'h', 'l', 'c', 'ATR_14']
            type: either bullish or bearish
            displacement_mult: How much stronger the move must be than the OB candle to count.
            forward_window: How many candles to look ahead for return after a hit.
        """
        curr = self.raw_data.iloc[i]
        prev = self.raw_data.iloc[i-1]

        if curr['c'] < curr['o'] and (curr['body']) > (prev['body']) * displacement_mult:
            if prev['c'] > prev['o']:
                self.active_zones.append({
                    'type': 'Bearish',
                    'top': prev['c'],
                    'bottom': prev['o'],
                    'created_time': self.raw_data['time'].iloc[i],
                    'created_at': i,
                    'status': 'Active'
                })

        # --- 2. INSPECT ACTIVE ZONES ---
        for zone in self.active_zones:
            if zone['status'] != 'Active': continue
            if zone['type'] == 'Bearish' and curr['c'] > zone['top']:
                zone['status'] = 'Invalidated'
                continue

            hit = False
            if zone['type'] == 'Bearish':  # Bearish
                if curr['h'] >= zone['bottom'] >= curr['c'] and i - zone['created_at'] > 10:
                    hit = True

            if hit and take_trade:
                self.signal = {
                    'status': 'active',
                    'position': -1,
                    'tp': prev['c'] - 100,
                    'sl': 25,
                    'entry_type': "market",
                    'ml_prob':0.6,
                    'i': i
                }
                zone['status'] = 'Mitigated'  # Mark as done

                return True

        return False

    def execute_trades(self, position = 'long'):
        signal = self.signal
        self.update_position()

        if signal['status'] == 'active' and signal['position'] == 1:
            if self.position == 0:
                self.units = round(self.riskable/signal['sl'])
                order = self.create_order(self.instrument,
                                          units=self.units,
                                          sl_distance= signal['sl'],
                                          tp_price= signal['tp'],
                                          suppress = True,
                                          ret = True)
                if order:
                    self.trade_created_at = signal['i']
                    self.position = 1
                    self.report_trade(order, 'long')


            if self.position == -1:
                order = self.create_order(self.instrument,
                                          units=self.units * -2,
                                          sl_distance= signal['sl'],
                                          tp_price= signal['tp'],
                                          suppress = True,
                                          ret = True)
                if order:
                    self.trade_created_at = signal['i']
                    self.position = 1
                    self.report_trade(order, 'long')


        if signal['status'] == 'active' and signal['position'] == -1:
            if self.position == 0:
                self.units = round(self.riskable/signal['sl'])
                order = self.create_order(self.instrument,
                                          units= self.units * -1,
                                          sl_distance= signal['sl'],
                                          tp_price= signal['tp'],
                                          suppress = True,
                                          ret = True)
                if order:
                    self.trade_created_at = signal['i']
                    self.position = -1
                    self.report_trade(order, 'short')

            if self.position == 1:
                order = self.create_order(self.instrument,
                                          units=self.units * -2,
                                          sl_distance= signal['sl'],
                                          tp_price= signal['tp'],
                                          suppress = True,
                                          ret = True)
                if order:
                    self.trade_created_at = signal['i']
                    self.position = -1
                    self.report_trade(order, 'short')


        if position == 'neutral':
            order = self.create_order(self.instrument,
                                      units= -self.units,
                                      sl_distance= signal['sl'],
                                      tp_price= signal['tp'],
                                      suppress = True,
                                      ret = True)
            if order:
                self.position = 0
                self.units = 0
                self.trade_created_at = 0
                self.report_trade(order, position)

    def update_position(self):
        positions = self.get_positions()

        instrument_data = next((p for p in positions if p['instrument'] == self.instrument), None)

        if instrument_data:
            net = int(float(instrument_data['long']['units'])) + int(float(instrument_data['short']['units']))

            if net > 0:
                self.position = 1  # Buy
            elif net < 0:
                self.position = -1 # Sell
            else:
                self.position = 0  # Flat
                self.units = 0
                self.trade_created_at = 0
        else:
            self.position = 0
            self.units = 0
            self.trade_created_at = 0

    def report_trade(self, order, going):  # NEW
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

print('starting')
if __name__ == "__main__":
    trader = ConTrader('oanda.cfg', 'XAU_USD', bar_length='1min', risk_percentage=2)

    print("Fetching historical M1 foundation data...")
    trader.get_most_recent(days=1)

    print("Entering live streaming loop with auto-reconnect fallback...")

    while True:
        try:
            # This triggers OANDA's live data feed
            trader.stream_data(trader.instrument)

        except Exception as e:
            print(f"\n[!] Stream disconnected or timed out: {e}")
            print("[*] Reconnecting to stream-fxpractice in 5 seconds...")

            time.sleep(5)

            try:
                trader.summary = trader.get_account_summary()
                trader.balance = trader.summary['balance']
                trader.riskable = float(trader.balance) * (2 / 100)
            except Exception:
                pass

            print("[*] Resuming stream session...")
            continue