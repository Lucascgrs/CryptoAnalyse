import requests
import pandas as pd
import time
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import random
import backoff


class CryptoAnalyzer:
    def __init__(self, base_dir: str = 'crypto_analysis_results', api_key: str = None):
        self.base_dir = base_dir
        self.api_base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        self.rate_limit_sleep = 3.0
        self.data_cache = {}
        self.fear_greed_data = None
        self.funding_rates = {}

    def get_headers(self):
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['x-cg-pro-api-key'] = self.api_key
        return headers

    @backoff.on_exception(backoff.expo,
                          requests.exceptions.RequestException,
                          max_tries=5,
                          giveup=lambda e: e.response is not None and e.response.status_code != 429)
    def make_api_request(self, url, params=None):
        headers = self.get_headers()
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            time.sleep(retry_after + random.uniform(0.5, 2.0))
            raise requests.exceptions.RequestException(response=response)

        response.raise_for_status()
        return response.json()

    def get_fear_greed_index(self, days=30):
        try:
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                fg_data = []
                for item in data.get('data', []):
                    fg_data.append({
                        'timestamp': datetime.fromtimestamp(int(item.get('timestamp'))).strftime('%Y-%m-%d'),
                        'value': int(item.get('value')),
                        'value_classification': item.get('value_classification'),
                    })

                self.fear_greed_data = pd.DataFrame(fg_data)
                return self.fear_greed_data
            return None
        except Exception:
            return None

    def get_funding_rates(self, crypto_symbol):
        try:
            symbol = crypto_symbol.upper() + "USDT"
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": 100}

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                funding_df = pd.DataFrame(data)
                if not funding_df.empty:
                    funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
                    funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)
                    self.funding_rates[crypto_symbol] = funding_df
                    return funding_df
            return None
        except Exception:
            return None

    def get_top_cryptos_by_market_cap(self, limit: int = 50) -> List[Dict]:
        try:
            url = f"{self.api_base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': 'false'
            }

            result = self.make_api_request(url, params)
            formatted_result = [
                {
                    "id": coin["id"],
                    "symbol": coin["symbol"],
                    "name": coin["name"],
                    "market_cap": coin.get("market_cap"),
                    "volume": coin.get("total_volume"),
                    "price": coin.get("current_price"),
                    "ath": coin.get("ath"),
                    "ath_change_percentage": coin.get("ath_change_percentage")
                }
                for coin in result
            ]

            return formatted_result
        except requests.exceptions.RequestException:
            return []

    def get_crypto_list(self) -> List[Dict]:
        try:
            url = f"{self.api_base_url}/coins/list"
            return self.make_api_request(url)
        except requests.exceptions.RequestException:
            return []

    def get_crypto_data(self, crypto_id: str) -> Optional[Dict]:
        try:
            url = f"{self.api_base_url}/coins/{crypto_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'true',
                'sparkline': 'true'
            }

            data = self.make_api_request(url, params)
            market_data = data.get('market_data', {})

            crypto_data = {
                'id': data['id'],
                'symbol': data['symbol'],
                'name': data['name'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price_usd': market_data.get('current_price', {}).get('usd'),
                'market_cap_usd': market_data.get('market_cap', {}).get('usd'),
                'volume_24h_usd': market_data.get('total_volume', {}).get('usd'),
                'price_change_24h': market_data.get('price_change_24h'),
                'price_change_percentage_24h': market_data.get('price_change_percentage_24h'),
                'high_24h_usd': market_data.get('high_24h', {}).get('usd'),
                'low_24h_usd': market_data.get('low_24h', {}).get('usd'),
                'ath_usd': market_data.get('ath', {}).get('usd'),
                'ath_date': market_data.get('ath_date', {}).get('usd'),
                'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd'),
            }

            return crypto_data
        except requests.exceptions.RequestException:
            return None

    def parse_time_interval(self, interval: str) -> Tuple[str, int]:
        interval_mapping = {
            '1m': ('minutely', 1), '3m': ('minutely', 1), '5m': ('minutely', 1),
            '15m': ('minutely', 1), '30m': ('minutely', 1), '1h': ('hourly', 90),
            '2h': ('hourly', 90), '4h': ('hourly', 90), '6h': ('hourly', 90),
            '12h': ('hourly', 90), '1d': ('daily', 365), '3d': ('daily', 365),
            '1w': ('daily', 365),
        }

        return interval_mapping.get(interval, interval_mapping['1d'])

    def get_historical_market_data(self, crypto_id: str, interval: str = '1d', days: int = 60) -> Optional[pd.DataFrame]:
        try:
            api_interval, max_days = self.parse_time_interval(interval)
            days = min(days, max_days)

            url = f"{self.api_base_url}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': api_interval
            }

            data = self.make_api_request(url, params)
            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            total_volumes = data.get('total_volumes', [])

            if not prices:
                return None

            historical_data = []
            for i in range(len(prices)):
                timestamp_ms = prices[i][0]
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

                entry = {
                    'id': crypto_id,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'current_price_usd': prices[i][1],
                    'market_cap_usd': market_caps[i][1] if i < len(market_caps) else None,
                    'volume_24h_usd': total_volumes[i][1] if i < len(total_volumes) else None
                }
                historical_data.append(entry)

            return pd.DataFrame(historical_data)
        except Exception:
            return None

    def get_crypto_dir(self, crypto_id: str) -> str:
        crypto_dir = os.path.join(self.base_dir, crypto_id)
        os.makedirs(crypto_dir, exist_ok=True)
        return crypto_dir

    def collect_historical_data(self, crypto_id: str, symbol: str = None, interval: str = '1d', days: int = 60) -> bool:
        try:
            df = self.get_historical_market_data(crypto_id, interval, days)
            if df is None or df.empty:
                return False

            current_data = self.get_crypto_data(crypto_id)
            if current_data:
                current_df = pd.DataFrame([current_data])
                latest_timestamp = pd.to_datetime(df['timestamp']).max() if 'timestamp' in df.columns else None
                current_timestamp = pd.to_datetime(current_data['timestamp'])

                if latest_timestamp is None or current_timestamp.date() != latest_timestamp.date():
                    df = pd.concat([df, current_df], ignore_index=True)

            # Nettoyage et tri des données
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.drop(columns=['date'])
                df = df.sort_values('timestamp')

            # Calcul de l'ATH historique
            df['historical_ath'] = df['current_price_usd'].expanding().max()
            df['pct_from_ath'] = (df['current_price_usd'] / df['historical_ath'] - 1) * 100

            # ATH officiel de CoinGecko
            if current_data and 'ath_usd' in current_data and current_data['ath_usd'] is not None:
                df['official_ath_usd'] = current_data['ath_usd']
                df['pct_from_official_ath'] = (df['current_price_usd'] / current_data['ath_usd'] - 1) * 100

            # Calcul des variations de volume
            if 'volume_24h_usd' in df.columns:
                df['volume_change_24h'] = df['volume_24h_usd'].pct_change() * 100

                if len(df) >= 20:
                    df['volume_sma20'] = df['volume_24h_usd'].rolling(window=20).mean()
                    df['volume_ratio'] = df['volume_24h_usd'] / df['volume_sma20']
                    df['volume_anomaly'] = np.where(df['volume_ratio'] > 2, 1, 0)

            # Ajouter le Fear & Greed Index
            if self.fear_greed_data is not None and 'timestamp' in df.columns:
                df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

                fg_temp = self.fear_greed_data.rename(columns={
                    'timestamp': 'fg_date',
                    'value': 'fear_greed_value',
                    'value_classification': 'fear_greed_label'
                })

                df = pd.merge(
                    df,
                    fg_temp[['fg_date', 'fear_greed_value', 'fear_greed_label']],
                    left_on='date_str',
                    right_on='fg_date',
                    how='left'
                )

                df = df.drop(['date_str', 'fg_date'], axis=1, errors='ignore')

            # Ajouter les funding rates
            if symbol and symbol in self.funding_rates:
                funding_df = self.funding_rates[symbol]
                if not funding_df.empty:
                    funding_df['date_str'] = funding_df['fundingTime'].dt.strftime('%Y-%m-%d')

                    funding_daily = funding_df.groupby('date_str').agg({
                        'fundingRate': ['mean', 'max', 'min']
                    }).reset_index()

                    funding_daily.columns = [
                        'date_str', 'funding_rate_avg', 'funding_rate_max', 'funding_rate_min'
                    ]

                    df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                    df = pd.merge(df, funding_daily, on='date_str', how='left')
                    df = df.drop('date_str', axis=1, errors='ignore')

            self.data_cache[f"{crypto_id}_{interval}"] = df
            return True

        except Exception:
            return False

    def get_technical_indicators(self, crypto_id: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        try:
            import ta
        except ImportError:
            return None

        cache_key = f"{crypto_id}_{interval}"
        df = self.data_cache.get(cache_key)

        if df is None or df.empty:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.dropna(subset=['current_price_usd'])

        if len(df) < 14:
            return None

        # Moyennes mobiles
        if len(df) >= 20:
            df['MA20'] = df['current_price_usd'].rolling(window=20).mean()
            df['MA50'] = df['current_price_usd'].rolling(window=50).mean() if len(df) >= 50 else None
            df['MA200'] = df['current_price_usd'].rolling(window=200).mean() if len(df) >= 200 else None

        # RSI
        if len(df) >= 14:
            df['RSI'] = ta.momentum.RSIIndicator(close=df['current_price_usd'], window=14).rsi()

        # MACD
        if len(df) >= 26:
            macd = ta.trend.MACD(
                close=df['current_price_usd'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()

        # Bollinger Bands
        if len(df) >= 20:
            bollinger = ta.volatility.BollingerBands(close=df['current_price_usd'], window=20, window_dev=2)
            df['BB_high'] = bollinger.bollinger_hband()
            df['BB_low'] = bollinger.bollinger_lband()
            df['BB_mid'] = bollinger.bollinger_mavg()
            df['BB_pct'] = bollinger.bollinger_pband()

            df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']

            rolling_mean = df['BB_width'].rolling(window=20).mean()
            rolling_std = df['BB_width'].rolling(window=20).std()
            df['BB_squeeze'] = (df['BB_width'] - rolling_mean) / rolling_std

            df['BB_extreme_squeeze'] = 0
            if len(df) >= 60:
                threshold = df['BB_width'].quantile(0.10)
                df.loc[df['BB_width'] < threshold, 'BB_extreme_squeeze'] = 1

        return df


class CryptoTradingSignals:
    def __init__(self, analyzer: CryptoAnalyzer):
        self.analyzer = analyzer

    def generate_signals(self, crypto_id: str, interval: str = '1d'):
        df = self.analyzer.get_technical_indicators(crypto_id, interval)

        if df is None or len(df) < 30:
            return None

        signals = df.copy()
        for col in ['signal', 'signal_ma_cross', 'signal_rsi', 'signal_macd', 'signal_bb', 'signal_squeeze',
                    'signal_volume', 'signal_fg', 'signal_funding']:
            signals[col] = 0

        # Signal de croisement de moyennes mobiles (MA)
        if 'MA20' in signals.columns and 'MA50' in signals.columns:
            valid_data = signals['MA20'].notnull() & signals['MA50'].notnull()

            signals.loc[
                valid_data &
                (signals['MA20'] > signals['MA50']) &
                (signals['MA20'].shift(1) <= signals['MA50'].shift(1)),
                'signal_ma_cross'
            ] = 1

            signals.loc[
                valid_data &
                (signals['MA20'] < signals['MA50']) &
                (signals['MA20'].shift(1) >= signals['MA50'].shift(1)),
                'signal_ma_cross'
            ] = -1

        # Signal RSI
        if 'RSI' in signals.columns:
            valid_rsi = signals['RSI'].notnull()

            signals.loc[
                valid_rsi &
                (signals['RSI'] > 30) &
                (signals['RSI'].shift(1) <= 30),
                'signal_rsi'
            ] = 1

            signals.loc[
                valid_rsi &
                (signals['RSI'] < 70) &
                (signals['RSI'].shift(1) >= 70),
                'signal_rsi'
            ] = -1

        # Signal MACD
        if all(col in signals.columns and signals[col].notnull().any() for col in ['MACD', 'MACD_signal']):
            valid_macd = signals['MACD'].notnull() & signals['MACD_signal'].notnull()

            signals.loc[
                valid_macd &
                (signals['MACD'] > signals['MACD_signal']) &
                (signals['MACD'].shift(1) <= signals['MACD_signal'].shift(1)),
                'signal_macd'
            ] = 1

            signals.loc[
                valid_macd &
                (signals['MACD'] < signals['MACD_signal']) &
                (signals['MACD'].shift(1) >= signals['MACD_signal'].shift(1)),
                'signal_macd'
            ] = -1

        # Signal Bollinger Bands
        if all(col in signals.columns for col in ['current_price_usd', 'BB_low', 'BB_high']):
            valid_bb = signals['BB_low'].notnull() & signals['BB_high'].notnull()

            signals.loc[
                valid_bb &
                (signals['current_price_usd'] <= signals['BB_low']) &
                (signals['current_price_usd'].shift(1) > signals['BB_low'].shift(1)),
                'signal_bb'
            ] = 1

            signals.loc[
                valid_bb &
                (signals['current_price_usd'] >= signals['BB_high']) &
                (signals['current_price_usd'].shift(1) < signals['BB_high'].shift(1)),
                'signal_bb'
            ] = -1

        # Signal de resserrement des bandes de Bollinger
        if 'BB_extreme_squeeze' in signals.columns:
            signals.loc[signals['BB_extreme_squeeze'] == 1, 'signal_squeeze'] = 1

        # Signaux basés sur le volume
        if 'volume_anomaly' in signals.columns:
            signals.loc[signals['volume_anomaly'] == 1, 'signal_volume'] = 1

        # Signaux basés sur le Fear & Greed Index
        if 'fear_greed_value' in signals.columns:
            valid_fg = signals['fear_greed_value'].notnull()
            signals.loc[valid_fg & (signals['fear_greed_value'] < 20), 'signal_fg'] = 1
            signals.loc[valid_fg & (signals['fear_greed_value'] > 80), 'signal_fg'] = -1

        # Signaux basés sur le Funding Rate
        if 'funding_rate_avg' in signals.columns:
            valid_fr = signals['funding_rate_avg'].notnull()
            signals.loc[valid_fr & (signals['funding_rate_avg'] < -0.01), 'signal_funding'] = 1
            signals.loc[valid_fr & (signals['funding_rate_avg'] > 0.01), 'signal_funding'] = -1

        # Combiner tous les signaux
        signal_cols = [col for col in signals.columns if col.startswith('signal_')]
        if signal_cols:
            available_signals = len([col for col in signal_cols if not signals[col].isna().all()])
            if available_signals > 0:
                signals['signal'] = signals[signal_cols].sum(axis=1)
                signals['signal_strength'] = signals['signal'] / available_signals

        return signals

    def plot_signals(self, signals_df, crypto_id, interval):
        if signals_df is None or 'signal_strength' not in signals_df.columns:
            return

        crypto_dir = self.analyzer.get_crypto_dir(crypto_id)

        # 1. Graphique principal d'analyse technique
        self._plot_technical_analysis(signals_df, crypto_id, interval, crypto_dir)

        # 2. Graphique dédié au funding rate si disponible
        if 'funding_rate_avg' in signals_df.columns and signals_df['funding_rate_avg'].notnull().any():
            self._plot_funding_rates(signals_df, crypto_id, interval, crypto_dir)

    def _plot_technical_analysis(self, signals_df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 12))

        # Graphique prix
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(signals_df['timestamp'], signals_df['current_price_usd'], label='Prix', color='blue')

        # Moyennes mobiles
        for ma, color in [('MA20', 'orange'), ('MA50', 'red'), ('MA200', 'purple')]:
            if ma in signals_df.columns and not signals_df[ma].isna().all():
                ax1.plot(signals_df['timestamp'], signals_df[ma], label=ma, color=color, alpha=0.7)

        # Bandes de Bollinger
        if all(col in signals_df.columns for col in ['BB_high', 'BB_low', 'BB_mid']):
            ax1.fill_between(
                signals_df['timestamp'],
                signals_df['BB_high'],
                signals_df['BB_low'],
                color='gray',
                alpha=0.2,
                label='Bandes de Bollinger'
            )
            ax1.plot(signals_df['timestamp'], signals_df['BB_high'], '--', color='gray', alpha=0.7)
            ax1.plot(signals_df['timestamp'], signals_df['BB_low'], '--', color='gray', alpha=0.7)

        # ATH
        if 'historical_ath' in signals_df.columns:
            ax1.plot(signals_df['timestamp'], signals_df['historical_ath'],
                     linestyle='--', color='darkgreen', label='ATH historique', alpha=0.5)

        if 'official_ath_usd' in signals_df.columns and not pd.isna(signals_df['official_ath_usd'].iloc[0]):
            ath_official = signals_df['official_ath_usd'].iloc[0]
            ax1.axhline(y=ath_official, color='green', linestyle='-.',
                        label=f'ATH officiel: ${ath_official:.2f}', alpha=0.5)

        # Signaux
        buy_signals = signals_df[signals_df['signal_strength'] >= 0.5]
        sell_signals = signals_df[signals_df['signal_strength'] <= -0.5]
        squeeze_signals = signals_df[
            signals_df['signal_squeeze'] > 0] if 'signal_squeeze' in signals_df.columns else pd.DataFrame()

        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['current_price_usd'],
                        marker='^', color='green', s=100, label='Signal Achat')

        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['current_price_usd'],
                        marker='v', color='red', s=100, label='Signal Vente')

        if not squeeze_signals.empty:
            ax1.scatter(squeeze_signals['timestamp'], squeeze_signals['current_price_usd'],
                        marker='*', color='purple', s=120, label='Resserrement BB')

        # Titre avec % ATH
        if 'pct_from_ath' in signals_df.columns:
            current_pct = signals_df['pct_from_ath'].iloc[-1]
            title = f'{crypto_id.upper()} - Signaux de Trading ({interval}) | {current_pct:.1f}% de l\'ATH historique'

            if 'pct_from_official_ath' in signals_df.columns:
                official_pct = signals_df['pct_from_official_ath'].iloc[-1]
                title += f' | {official_pct:.1f}% de l\'ATH officiel'

            ax1.set_title(title)
        else:
            ax1.set_title(f'{crypto_id.upper()} - Signaux de Trading ({interval})')

        ax1.set_ylabel('Prix (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # RSI
        if 'RSI' in signals_df.columns and not signals_df['RSI'].isna().all():
            ax2 = plt.subplot(4, 1, 2, sharex=ax1)
            ax2.plot(signals_df['timestamp'], signals_df['RSI'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax2.fill_between(signals_df['timestamp'], 70, 100, color='red', alpha=0.1)
            ax2.fill_between(signals_df['timestamp'], 0, 30, color='green', alpha=0.1)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True)

        # Volume
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)
        ax3.bar(signals_df['timestamp'], signals_df['volume_24h_usd'], color='blue', alpha=0.6, label='Volume')

        if 'volume_anomaly' in signals_df.columns:
            volume_anomalies = signals_df[signals_df['volume_anomaly'] == 1]
            if not volume_anomalies.empty:
                ax3.bar(volume_anomalies['timestamp'], volume_anomalies['volume_24h_usd'],
                        color='red', alpha=0.7, label='Anomalie Volume')

        ax3.set_ylabel('Volume (USD)')
        ax3.legend()
        ax3.grid(True)

        # MACD
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)

        if all(col in signals_df.columns for col in ['MACD', 'MACD_signal']):
            ax4.plot(signals_df['timestamp'], signals_df['MACD'], label='MACD', color='blue')
            ax4.plot(signals_df['timestamp'], signals_df['MACD_signal'], label='Signal', color='orange')

            if 'MACD_diff' in signals_df.columns:
                diff = signals_df['MACD_diff'].fillna(0)
                ax4.bar(signals_df['timestamp'], diff,
                        color=np.where(diff > 0, 'g', 'r'), alpha=0.5)

        # Axe secondaire pour BB_width
        if 'BB_width' in signals_df.columns:
            ax4b = ax4.twinx()
            ax4b.plot(signals_df['timestamp'], signals_df['BB_width'], color='purple',
                      linestyle='--', label='BB Width', alpha=0.7)

            if 'BB_extreme_squeeze' in signals_df.columns:
                squeeze_points = signals_df[signals_df['BB_extreme_squeeze'] == 1]
                if not squeeze_points.empty:
                    ax4b.scatter(squeeze_points['timestamp'], squeeze_points['BB_width'],
                                 marker='*', color='red', s=80, label='Squeeze')

            ax4b.set_ylabel('BB Width', color='purple')
            ax4b.tick_params(axis='y', colors='purple')

            lines, labels = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4b.get_legend_handles_labels()
            ax4.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax4.legend()

        ax4.set_ylabel('MACD')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_trading_signals_{interval}.png"))
        plt.close()

        # Graphique Fear & Greed séparé
        if 'fear_greed_value' in signals_df.columns and not signals_df['fear_greed_value'].isna().all():
            self._plot_fear_greed(signals_df, crypto_id, interval, crypto_dir)

    def _plot_fear_greed(self, signals_df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        color_map = np.where(signals_df['fear_greed_value'] < 50, 'green', 'red')
        ax.bar(signals_df['timestamp'], signals_df['fear_greed_value'],
               color=color_map, alpha=0.5, label='Fear & Greed')

        ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Peur Extrême (20)')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Avidité Extrême (80)')

        ax.set_ylim(0, 100)
        ax.set_ylabel('Fear & Greed Index')

        for i, row in signals_df.loc[signals_df['signal_fg'] != 0].iterrows():
            color = 'green' if row['signal_fg'] > 0 else 'red'
            label = 'Acheter (Peur)' if row['signal_fg'] > 0 else 'Vendre (Avidité)'
            ax.scatter(row['timestamp'], row['fear_greed_value'],
                       marker='o' if row['signal_fg'] > 0 else 'x',
                       color=color, s=100, label=label if i == 0 else "")

        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(f'{crypto_id.upper()} - Fear & Greed Index ({interval})')

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_fear_greed_{interval}.png"))
        plt.close()

    def _plot_funding_rates(self, signals_df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        # Convertir en pourcentage pour plus de lisibilité
        ax.plot(signals_df['timestamp'], signals_df['funding_rate_avg'] * 100,
                color='blue', label='Funding Rate (%)', marker='o', markersize=3)

        # Afficher les min/max si disponibles
        if 'funding_rate_min' in signals_df.columns and 'funding_rate_max' in signals_df.columns:
            ax.fill_between(
                signals_df['timestamp'],
                signals_df['funding_rate_min'] * 100,
                signals_df['funding_rate_max'] * 100,
                color='blue', alpha=0.2, label='Min-Max Range'
            )

        # Lignes de référence
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.5, label='Élevé (0.1%)')
        ax.axhline(y=-0.1, color='green', linestyle=':', alpha=0.5, label='Négatif (-0.1%)')

        # Signaux funding rate
        for i, row in signals_df.loc[signals_df['signal_funding'] != 0].iterrows():
            color = 'green' if row['signal_funding'] > 0 else 'red'
            label = 'Acheter (FR négatif)' if row['signal_funding'] > 0 else 'Vendre (FR positif)'
            ax.scatter(row['timestamp'], row['funding_rate_avg'] * 100,
                       marker='o' if row['signal_funding'] > 0 else 'x',
                       color=color, s=100, label=label if i == 0 else "")

        ax.set_ylabel('Funding Rate (%)')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(f'{crypto_id.upper()} - Funding Rates ({interval})')

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_funding_rates_{interval}.png"))
        plt.close()

    def export_signals_to_excel(self, signals_df, crypto_id, interval):
        if signals_df is None or signals_df.empty:
            return

        crypto_dir = self.analyzer.get_crypto_dir(crypto_id)
        filename = os.path.join(crypto_dir, f"{crypto_id}_trading_signals_{interval}.xlsx")
        signals_df.to_excel(filename, index=False)

    def export_all_signals_to_excel(self, all_signals, interval):
        if not all_signals:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.analyzer.base_dir, f"all_trading_signals_{interval}_{timestamp}.xlsx")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for crypto_id, signals_df in all_signals.items():
                if signals_df is not None and not signals_df.empty:
                    sheet_name = crypto_id[:31]  # Excel limite à 31 caractères
                    signals_df.to_excel(writer, sheet_name=sheet_name, index=False)

    def create_detailed_summary(self, all_signals, interval):
        if not all_signals:
            return pd.DataFrame()

        summary_data = []

        for crypto_id, signals_df in all_signals.items():
            if signals_df is None or signals_df.empty:
                continue

            last_signal = signals_df.iloc[-1]

            crypto_info = {
                'crypto_id': crypto_id,
                'name': signals_df['name'].iloc[0] if 'name' in signals_df.columns else crypto_id,
                'symbol': signals_df['symbol'].iloc[0].upper() if 'symbol' in signals_df.columns else crypto_id,
                'price_usd': last_signal.get('current_price_usd'),
                'market_cap_usd': last_signal.get('market_cap_usd'),
                'volume_24h_usd': last_signal.get('volume_24h_usd'),
                'timestamp': last_signal.get('timestamp'),
                'interval': interval
            }

            # ATH
            if 'historical_ath' in last_signal.index:
                crypto_info['historical_ath'] = last_signal.get('historical_ath')
            if 'pct_from_ath' in last_signal.index:
                crypto_info['pct_from_ath'] = last_signal.get('pct_from_ath')
            if 'official_ath_usd' in last_signal.index:
                crypto_info['official_ath_usd'] = last_signal.get('official_ath_usd')
            if 'pct_from_official_ath' in last_signal.index:
                crypto_info['pct_from_official_ath'] = last_signal.get('pct_from_official_ath')

            # Volume
            if 'volume_change_24h' in last_signal.index:
                crypto_info['volume_change_24h'] = last_signal.get('volume_change_24h')
            if 'volume_ratio' in last_signal.index:
                crypto_info['volume_ratio'] = last_signal.get('volume_ratio')
            if 'volume_anomaly' in last_signal.index:
                crypto_info['volume_anomaly'] = last_signal.get('volume_anomaly')

            # Fear & Greed
            if 'fear_greed_value' in last_signal.index:
                crypto_info['fear_greed_value'] = last_signal.get('fear_greed_value')
                crypto_info['fear_greed_label'] = last_signal.get('fear_greed_label')

            # Funding Rate
            if 'funding_rate_avg' in last_signal.index:
                crypto_info['funding_rate_avg'] = last_signal.get('funding_rate_avg')
            if 'funding_rate_max' in last_signal.index:
                crypto_info['funding_rate_max'] = last_signal.get('funding_rate_max')
            if 'funding_rate_min' in last_signal.index:
                crypto_info['funding_rate_min'] = last_signal.get('funding_rate_min')

            # Signaux
            signal_cols = [col for col in last_signal.index if col.startswith('signal_')]
            for col in signal_cols:
                crypto_info[col] = last_signal.get(col, 0)

            crypto_info['signal_strength'] = last_signal.get('signal_strength', 0)

            # Bollinger Bands
            if 'BB_width' in last_signal.index:
                crypto_info['BB_width'] = last_signal.get('BB_width')
            if 'BB_squeeze' in last_signal.index:
                crypto_info['BB_squeeze'] = last_signal.get('BB_squeeze')
            if 'BB_extreme_squeeze' in last_signal.index:
                crypto_info['BB_extreme_squeeze'] = last_signal.get('BB_extreme_squeeze')

            summary_data.append(crypto_info)

        if not summary_data:
            return pd.DataFrame()

        summary_df = pd.DataFrame(summary_data)

        # Organiser les colonnes
        base_cols = ['crypto_id', 'name', 'symbol', 'price_usd',
                     'historical_ath', 'pct_from_ath', 'official_ath_usd', 'pct_from_official_ath',
                     'market_cap_usd', 'volume_24h_usd', 'volume_change_24h', 'volume_ratio',
                     'fear_greed_value', 'fear_greed_label',
                     'funding_rate_avg', 'funding_rate_max', 'funding_rate_min',
                     'timestamp', 'interval']
        bb_cols = [col for col in summary_df.columns if col.startswith('BB_')]
        signal_cols = [col for col in summary_df.columns if col.startswith('signal_')]
        final_cols = base_cols + bb_cols + signal_cols

        existing_cols = [col for col in final_cols if col in summary_df.columns]
        return summary_df[existing_cols]


def main(interval='1d', num_cryptos=10, historical_days=90, output_dir='crypto_analysis_results', api_key=None):
    os.makedirs(output_dir, exist_ok=True)

    analyzer = CryptoAnalyzer(base_dir=output_dir, api_key=api_key)
    signals_generator = CryptoTradingSignals(analyzer)

    analyzer.get_fear_greed_index(days=historical_days)

    crypto_list = analyzer.get_top_cryptos_by_market_cap(limit=num_cryptos)

    if not crypto_list:
        all_cryptos = analyzer.get_crypto_list()
        crypto_list = all_cryptos[:num_cryptos] if all_cryptos else []

    if not crypto_list:
        return

    all_signals = {}

    for idx, crypto in enumerate(crypto_list, 1):
        crypto_id = crypto['id']
        symbol = crypto.get('symbol', '')
        name = crypto.get('name', crypto_id)

        try:
            if symbol:
                analyzer.get_funding_rates(symbol)

            historical_success = analyzer.collect_historical_data(crypto_id, symbol, interval, historical_days)

            if historical_success:
                signals = signals_generator.generate_signals(crypto_id, interval)

                if signals is not None and not signals.empty:
                    signals['name'] = name
                    signals['symbol'] = symbol
                    all_signals[crypto_id] = signals
                    signals_generator.plot_signals(signals, crypto_id, interval)
                    signals_generator.export_signals_to_excel(signals, crypto_id, interval)

        except Exception:
            import traceback
            traceback.print_exc()

        if idx < len(crypto_list):
            time.sleep(analyzer.rate_limit_sleep * random.uniform(1.0, 1.5))

    if all_signals:
        signals_generator.export_all_signals_to_excel(all_signals, interval)
        summary_df = signals_generator.create_detailed_summary(all_signals, interval)

        if not summary_df.empty:
            summary_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = os.path.join(output_dir, f"crypto_signals_summary_{interval}_{summary_timestamp}.xlsx")
            summary_df.to_excel(summary_filename, index=False)


if __name__ == "__main__":
    INTERVALS = ['1d']
    NUM_CRYPTOS = 2
    HISTORICAL_DAYS = 90
    OUTPUT_DIR = 'crypto_analysis_results'
    API_KEY = None

    try:
        for interval in INTERVALS:
            main(interval=interval,
                 num_cryptos=NUM_CRYPTOS,
                 historical_days=HISTORICAL_DAYS,
                 output_dir=OUTPUT_DIR,
                 api_key=API_KEY)

            if interval != INTERVALS[-1]:
                time.sleep(5)
    except Exception:
        import traceback

        traceback.print_exc()