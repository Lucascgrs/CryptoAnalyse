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

    def get_historical_market_data(self, crypto_id: str, interval: str = '1d', days: int = 60) -> Optional[
        pd.DataFrame]:
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

                # Ajouter la tendance du Fear & Greed
                df['fear_greed_change'] = df['fear_greed_value'].diff()
                df['fear_greed_trend'] = np.sign(df['fear_greed_change'])

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

                    # Ajouter la tendance du funding rate
                    funding_daily['funding_rate_prev'] = funding_daily['funding_rate_avg'].shift(1)
                    funding_daily['funding_rate_change'] = funding_daily['funding_rate_avg'] - funding_daily[
                        'funding_rate_prev']
                    funding_daily['funding_rate_trend'] = np.sign(funding_daily['funding_rate_change'])
                    funding_daily = funding_daily.drop('funding_rate_prev', axis=1)

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

        # Calcul des rendements pour les tendances
        df['price_change_1d'] = df['current_price_usd'].pct_change(1)
        df['price_change_3d'] = df['current_price_usd'].pct_change(3)
        df['price_change_7d'] = df['current_price_usd'].pct_change(7)

        # Calcul des tendances
        df['trend_1d'] = np.sign(df['price_change_1d'])
        df['trend_3d'] = np.sign(df['price_change_3d'])
        df['trend_7d'] = np.sign(df['price_change_7d'])

        # Moyennes mobiles
        if len(df) >= 20:
            df['MA20'] = df['current_price_usd'].rolling(window=20).mean()
            df['MA50'] = df['current_price_usd'].rolling(window=50).mean() if len(df) >= 50 else None
            df['MA200'] = df['current_price_usd'].rolling(window=200).mean() if len(df) >= 200 else None

            # Distance aux moyennes mobiles (en %)
            df['dist_MA20'] = (df['current_price_usd'] / df['MA20'] - 1) * 100
            df['MA20_trend'] = np.sign(df['MA20'].diff())

            if 'MA50' in df.columns and df['MA50'].notnull().any():
                df['dist_MA50'] = (df['current_price_usd'] / df['MA50'] - 1) * 100
                df['MA50_trend'] = np.sign(df['MA50'].diff())

            if 'MA200' in df.columns and df['MA200'].notnull().any():
                df['dist_MA200'] = (df['current_price_usd'] / df['MA200'] - 1) * 100
                df['MA200_trend'] = np.sign(df['MA200'].diff())

        # RSI
        if len(df) >= 14:
            df['RSI'] = ta.momentum.RSIIndicator(close=df['current_price_usd'], window=14).rsi()
            df['RSI_change'] = df['RSI'].diff()
            df['RSI_trend'] = np.sign(df['RSI_change'])

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
            df['MACD_trend'] = np.sign(df['MACD_diff'])

        # Bollinger Bands
        if len(df) >= 20:
            bollinger = ta.volatility.BollingerBands(close=df['current_price_usd'], window=20, window_dev=2)
            df['BB_high'] = bollinger.bollinger_hband()
            df['BB_low'] = bollinger.bollinger_lband()
            df['BB_mid'] = bollinger.bollinger_mavg()
            df['BB_pct'] = bollinger.bollinger_pband()  # Position dans les bandes (0-1)

            # Largeur des bandes - mesure de volatilité et resserrement
            df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
            df['BB_width_trend'] = np.sign(df['BB_width'].diff())

            # Détection des resserrements (historiquement bas)
            if len(df) >= 60:
                df['BB_width_20d_avg'] = df['BB_width'].rolling(window=20).mean()
                df['BB_width_20d_std'] = df['BB_width'].rolling(window=20).std()
                df['BB_width_zscore'] = (df['BB_width'] - df['BB_width_20d_avg']) / df['BB_width_20d_std']

                # Calcul du percentile du resserrement actuel (plus bas = plus resserré)
                df['BB_squeeze'] = np.where(df['BB_width'] < df['BB_width'].quantile(0.10), 1, 0)

        # Autres indicateurs techniques utiles
        # Stochastique
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(
                high=df['current_price_usd'].rolling(14).max(),
                low=df['current_price_usd'].rolling(14).min(),
                close=df['current_price_usd'],
                window=14,
                smooth_window=3
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['stoch_trend'] = np.sign(df['stoch_k'].diff())

        # Momentum
        df['momentum_1d'] = df['current_price_usd'] / df['current_price_usd'].shift(1) - 1
        df['momentum_7d'] = df['current_price_usd'] / df['current_price_usd'].shift(7) - 1
        df['momentum_14d'] = df['current_price_usd'] / df['current_price_usd'].shift(14) - 1

        return df


class CryptoDataVisualizer:
    def __init__(self, analyzer: CryptoAnalyzer):
        self.analyzer = analyzer

    def visualize_data(self, df, crypto_id, interval):
        if df is None:
            return

        crypto_dir = self.analyzer.get_crypto_dir(crypto_id)

        # Visualisations principales
        self._plot_price_and_volume(df, crypto_id, interval, crypto_dir)
        self._plot_technical_indicators(df, crypto_id, interval, crypto_dir)

        # Visualisations spécifiques si les données sont disponibles
        if 'fear_greed_value' in df.columns and df['fear_greed_value'].notnull().any():
            self._plot_fear_greed(df, crypto_id, interval, crypto_dir)

        if 'funding_rate_avg' in df.columns and df['funding_rate_avg'].notnull().any():
            self._plot_funding_rates(df, crypto_id, interval, crypto_dir)

    def _plot_price_and_volume(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 10))

        # Graphique des prix
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df['timestamp'], df['current_price_usd'], label='Prix', color='blue')

        # Moyennes mobiles
        for ma, color in [('MA20', 'orange'), ('MA50', 'red'), ('MA200', 'purple')]:
            if ma in df.columns and not df[ma].isna().all():
                ax1.plot(df['timestamp'], df[ma], label=ma, color=color, alpha=0.7)

        # Bandes de Bollinger
        if all(col in df.columns for col in ['BB_high', 'BB_low', 'BB_mid']):
            ax1.fill_between(
                df['timestamp'],
                df['BB_high'],
                df['BB_low'],
                color='gray',
                alpha=0.2,
                label='Bandes de Bollinger'
            )
            ax1.plot(df['timestamp'], df['BB_high'], '--', color='gray', alpha=0.7)
            ax1.plot(df['timestamp'], df['BB_low'], '--', color='gray', alpha=0.7)

        # ATH
        if 'historical_ath' in df.columns:
            ax1.plot(df['timestamp'], df['historical_ath'],
                     linestyle='--', color='darkgreen', label='ATH historique', alpha=0.5)

        if 'official_ath_usd' in df.columns and not pd.isna(df['official_ath_usd'].iloc[0]):
            ath_official = df['official_ath_usd'].iloc[0]
            ax1.axhline(y=ath_official, color='green', linestyle='-.',
                        label=f'ATH officiel: ${ath_official:.2f}', alpha=0.5)

        # Titre avec % ATH
        if 'pct_from_ath' in df.columns:
            current_pct = df['pct_from_ath'].iloc[-1]
            title = f'{crypto_id.upper()} - Prix et Volume ({interval}) | {current_pct:.1f}% de l\'ATH historique'

            if 'pct_from_official_ath' in df.columns:
                official_pct = df['pct_from_official_ath'].iloc[-1]
                title += f' | {official_pct:.1f}% de l\'ATH officiel'

            ax1.set_title(title)
        else:
            ax1.set_title(f'{crypto_id.upper()} - Prix et Volume ({interval})')

        ax1.set_ylabel('Prix (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Graphique des volumes
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.bar(df['timestamp'], df['volume_24h_usd'], color='blue', alpha=0.6, label='Volume')

        # Marquer les anomalies de volume
        if 'volume_anomaly' in df.columns:
            volume_anomalies = df[df['volume_anomaly'] == 1]
            if not volume_anomalies.empty:
                ax2.bar(volume_anomalies['timestamp'], volume_anomalies['volume_24h_usd'],
                        color='red', alpha=0.7, label='Anomalie Volume (>2x moyenne 20j)')

        # Moyenne mobile du volume
        if 'volume_sma20' in df.columns:
            ax2.plot(df['timestamp'], df['volume_sma20'],
                     color='black', linestyle='--', label='Volume SMA(20)')

        ax2.set_ylabel('Volume (USD)')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_price_volume_{interval}.png"))
        plt.close()

    def _plot_technical_indicators(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 16))

        # RSI
        ax1 = plt.subplot(4, 1, 1)
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            ax1.plot(df['timestamp'], df['RSI'], label='RSI', color='blue')
            ax1.axhline(y=70, color='r', linestyle='-', alpha=0.3, label='Surachat (70)')
            ax1.axhline(y=30, color='g', linestyle='-', alpha=0.3, label='Survente (30)')
            ax1.axhline(y=50, color='black', linestyle='--', alpha=0.2)
            ax1.fill_between(df['timestamp'], 70, 100, color='red', alpha=0.1)
            ax1.fill_between(df['timestamp'], 0, 30, color='green', alpha=0.1)
            ax1.set_ylim(0, 100)

            # Annotate current value and trend
            current_rsi = df['RSI'].iloc[-1]
            current_trend = "↑" if df['RSI_trend'].iloc[-1] > 0 else "↓" if df['RSI_trend'].iloc[-1] < 0 else "→"
            ax1.text(0.02, 0.95, f"RSI: {current_rsi:.1f} {current_trend}",
                     transform=ax1.transAxes, fontsize=10, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax1.set_title(f'{crypto_id.upper()} - Indicateurs Techniques ({interval})')
        ax1.set_ylabel('RSI')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # MACD
        ax2 = plt.subplot(4, 1, 2, sharex=ax1)
        if all(col in df.columns for col in ['MACD', 'MACD_signal']):
            ax2.plot(df['timestamp'], df['MACD'], label='MACD', color='blue')
            ax2.plot(df['timestamp'], df['MACD_signal'], label='Signal', color='orange')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)

            if 'MACD_diff' in df.columns:
                # Histogramme de différence MACD
                diff = df['MACD_diff'].fillna(0)
                ax2.bar(df['timestamp'], diff,
                        color=np.where(diff > 0, 'g', 'r'), alpha=0.5, label='MACD-Signal')

                # Annotate current values
                current_macd = df['MACD'].iloc[-1]
                current_signal = df['MACD_signal'].iloc[-1]
                current_diff = df['MACD_diff'].iloc[-1]
                current_trend = "↑" if df['MACD_trend'].iloc[-1] > 0 else "↓" if df['MACD_trend'].iloc[-1] < 0 else "→"

                ax2.text(0.02, 0.95,
                         f"MACD: {current_macd:.4f}, Signal: {current_signal:.4f}, Diff: {current_diff:.4f} {current_trend}",
                         transform=ax2.transAxes, fontsize=10, va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        # Bollinger Bands Width
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)
        if 'BB_width' in df.columns:
            ax3.plot(df['timestamp'], df['BB_width'], color='purple', label='BB Width')

            if 'BB_squeeze' in df.columns:
                squeeze_points = df[df['BB_squeeze'] == 1]
                if not squeeze_points.empty:
                    ax3.scatter(squeeze_points['timestamp'], squeeze_points['BB_width'],
                                marker='*', color='red', s=80, label='Resserrement <10%')

            if 'BB_width_zscore' in df.columns:
                ax3b = ax3.twinx()
                ax3b.plot(df['timestamp'], df['BB_width_zscore'],
                          color='blue', linestyle='--', alpha=0.7, label='Z-Score')
                ax3b.set_ylabel('Z-Score', color='blue')
                ax3b.tick_params(axis='y', colors='blue')

                # Combiner les légendes
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3b.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                # Annotate current values
                current_width = df['BB_width'].iloc[-1]
                current_zscore = df['BB_width_zscore'].iloc[-1]
                current_trend = "↑" if df['BB_width_trend'].iloc[-1] > 0 else "↓" if df['BB_width_trend'].iloc[
                                                                                         -1] < 0 else "→"

                ax3.text(0.02, 0.95,
                         f"BB Width: {current_width:.4f} {current_trend}, Z-Score: {current_zscore:.2f}",
                         transform=ax3.transAxes, fontsize=10, va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax3.legend(loc='upper right')

        ax3.set_ylabel('BB Width')
        ax3.grid(True)

        # Stochastic Oscillator
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)
        if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            ax4.plot(df['timestamp'], df['stoch_k'], label='%K', color='blue')
            ax4.plot(df['timestamp'], df['stoch_d'], label='%D', color='red')
            ax4.axhline(y=80, color='r', linestyle='-', alpha=0.3, label='Surachat (80)')
            ax4.axhline(y=20, color='g', linestyle='-', alpha=0.3, label='Survente (20)')
            ax4.fill_between(df['timestamp'], 80, 100, color='red', alpha=0.1)
            ax4.fill_between(df['timestamp'], 0, 20, color='green', alpha=0.1)
            ax4.set_ylim(0, 100)

            # Annotate current values
            current_k = df['stoch_k'].iloc[-1]
            current_d = df['stoch_d'].iloc[-1]
            current_trend = "↑" if df['stoch_trend'].iloc[-1] > 0 else "↓" if df['stoch_trend'].iloc[-1] < 0 else "→"

            ax4.text(0.02, 0.95,
                     f"Stoch %K: {current_k:.1f}, %D: {current_d:.1f} {current_trend}",
                     transform=ax4.transAxes, fontsize=10, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax4.set_ylabel('Stochastic')
        ax4.legend(loc='upper right')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_indicators_{interval}.png"))
        plt.close()

    def _plot_fear_greed(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        # Définir les couleurs en fonction du niveau
        def get_fg_color(value):
            if pd.isna(value):
                return 'gray'
            elif value <= 20:  # Peur extrême
                return 'darkgreen'
            elif value <= 40:  # Peur
                return 'lightgreen'
            elif value <= 60:  # Neutre
                return 'yellow'
            elif value <= 80:  # Avidité
                return 'orange'
            else:  # Avidité extrême
                return 'red'

        # Créer une liste de couleurs pour chaque barre
        colors = [get_fg_color(val) for val in df['fear_greed_value']]

        # Graphique en barres avec couleurs dynamiques
        ax.bar(df['timestamp'], df['fear_greed_value'], color=colors)

        # Lignes de référence
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Peur Extrême (<20)')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Avidité Extrême (>80)')
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Neutre (50)')

        ax.set_ylim(0, 100)
        ax.set_ylabel('Fear & Greed Index')
        ax.set_title(f'{crypto_id.upper()} - Fear & Greed Index ({interval})')

        # Annotate current value and trend
        if not df['fear_greed_value'].isna().all():
            current_fg = df['fear_greed_value'].iloc[-1]
            current_label = df['fear_greed_label'].iloc[-1] if 'fear_greed_label' in df.columns else ""
            current_trend = "↑" if df['fear_greed_trend'].iloc[-1] > 0 else "↓" if df['fear_greed_trend'].iloc[
                                                                                       -1] < 0 else "→"

            ax.text(0.02, 0.95,
                    f"Fear & Greed: {current_fg} - {current_label} {current_trend}",
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_fear_greed_{interval}.png"))
        plt.close()

    def _plot_funding_rates(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        # Convertir en pourcentage pour plus de lisibilité
        ax.plot(df['timestamp'], df['funding_rate_avg'] * 100,
                color='blue', label='Funding Rate (%)', marker='o', markersize=3)

        # Afficher les min/max si disponibles
        if 'funding_rate_min' in df.columns and 'funding_rate_max' in df.columns:
            ax.fill_between(
                df['timestamp'],
                df['funding_rate_min'] * 100,
                df['funding_rate_max'] * 100,
                color='blue', alpha=0.2, label='Min-Max Range'
            )

        # Lignes de référence
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Neutre (0%)')
        ax.axhline(y=0.03, color='red', linestyle=':', alpha=0.5, label='Élevé (0.03%)')
        ax.axhline(y=-0.03, color='green', linestyle=':', alpha=0.5, label='Négatif (-0.03%)')

        # Annotate current value and trend
        if not df['funding_rate_avg'].isna().all():
            current_fr = df['funding_rate_avg'].iloc[-1] * 100
            current_trend = "↑" if df['funding_rate_trend'].iloc[-1] > 0 else "↓" if df['funding_rate_trend'].iloc[
                                                                                         -1] < 0 else "→"

            ax.text(0.02, 0.95,
                    f"Funding Rate: {current_fr:.4f}% {current_trend}",
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_ylabel('Funding Rate (%)')
        ax.set_title(f'{crypto_id.upper()} - Funding Rates ({interval})')
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_funding_rates_{interval}.png"))
        plt.close()

    def export_data_to_excel(self, df, crypto_id, interval):
        if df is None or df.empty:
            return

        crypto_dir = self.analyzer.get_crypto_dir(crypto_id)
        filename = os.path.join(crypto_dir, f"{crypto_id}_data_{interval}.xlsx")
        df.to_excel(filename, index=False)

    def export_all_data_to_excel(self, all_data, interval):
        if not all_data:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.analyzer.base_dir, f"all_crypto_data_{interval}_{timestamp}.xlsx")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for crypto_id, data_df in all_data.items():
                if data_df is not None and not data_df.empty:
                    sheet_name = crypto_id[:31]  # Excel limite à 31 caractères
                    data_df.to_excel(writer, sheet_name=sheet_name, index=False)

    def create_detailed_summary(self, all_data, interval):
        if not all_data:
            return pd.DataFrame()

        summary_data = []

        for crypto_id, df in all_data.items():
            if df is None or df.empty:
                continue

            # Dernière ligne de données
            last_data = df.iloc[-1]

            crypto_info = {
                'crypto_id': crypto_id,
                'name': df['name'].iloc[0] if 'name' in df.columns else crypto_id,
                'symbol': df['symbol'].iloc[0].upper() if 'symbol' in df.columns else crypto_id,
                'price_usd': last_data.get('current_price_usd'),
                'market_cap_usd': last_data.get('market_cap_usd'),
                'volume_24h_usd': last_data.get('volume_24h_usd'),
                'timestamp': last_data.get('timestamp'),
                'interval': interval
            }

            # ATH
            if 'historical_ath' in last_data.index:
                crypto_info['historical_ath'] = last_data.get('historical_ath')
            if 'pct_from_ath' in last_data.index:
                crypto_info['pct_from_ath'] = last_data.get('pct_from_ath')
            if 'official_ath_usd' in last_data.index:
                crypto_info['official_ath_usd'] = last_data.get('official_ath_usd')
            if 'pct_from_official_ath' in last_data.index:
                crypto_info['pct_from_official_ath'] = last_data.get('pct_from_official_ath')

            # Volume
            if 'volume_change_24h' in last_data.index:
                crypto_info['volume_change_24h'] = last_data.get('volume_change_24h')
            if 'volume_ratio' in last_data.index:
                crypto_info['volume_ratio'] = last_data.get('volume_ratio')
            if 'volume_anomaly' in last_data.index:
                crypto_info['volume_anomaly'] = last_data.get('volume_anomaly')

            # Fear & Greed
            if 'fear_greed_value' in last_data.index:
                crypto_info['fear_greed_value'] = last_data.get('fear_greed_value')
                crypto_info['fear_greed_label'] = last_data.get('fear_greed_label')
                if 'fear_greed_trend' in last_data.index:
                    crypto_info['fear_greed_trend'] = last_data.get('fear_greed_trend')

            # Funding Rate
            if 'funding_rate_avg' in last_data.index:
                crypto_info['funding_rate_avg'] = last_data.get('funding_rate_avg')
            if 'funding_rate_max' in last_data.index:
                crypto_info['funding_rate_max'] = last_data.get('funding_rate_max')
            if 'funding_rate_min' in last_data.index:
                crypto_info['funding_rate_min'] = last_data.get('funding_rate_min')
            if 'funding_rate_trend' in last_data.index:
                crypto_info['funding_rate_trend'] = last_data.get('funding_rate_trend')

            # Indicateurs techniques
            tech_indicators = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_width', 'BB_squeeze',
                               'stoch_k', 'stoch_d']

            for indicator in tech_indicators:
                if indicator in last_data.index:
                    crypto_info[indicator] = last_data.get(indicator)

            # Tendances
            trend_indicators = ['trend_1d', 'trend_3d', 'trend_7d', 'RSI_trend', 'MACD_trend',
                                'BB_width_trend', 'stoch_trend']

            for trend in trend_indicators:
                if trend in last_data.index:
                    crypto_info[trend] = last_data.get(trend)

            summary_data.append(crypto_info)

        if not summary_data:
            return pd.DataFrame()

        summary_df = pd.DataFrame(summary_data)

        # Organiser les colonnes
        base_cols = ['crypto_id', 'name', 'symbol', 'price_usd',
                     'historical_ath', 'pct_from_ath', 'official_ath_usd', 'pct_from_official_ath',
                     'market_cap_usd', 'volume_24h_usd', 'volume_change_24h', 'volume_ratio',
                     'fear_greed_value', 'fear_greed_label', 'fear_greed_trend',
                     'funding_rate_avg', 'funding_rate_max', 'funding_rate_min', 'funding_rate_trend',
                     'timestamp', 'interval']

        tech_cols = [col for col in summary_df.columns if col in tech_indicators]
        trend_cols = [col for col in summary_df.columns if col in trend_indicators]

        final_cols = base_cols + tech_cols + trend_cols

        existing_cols = [col for col in final_cols if col in summary_df.columns]
        return summary_df[existing_cols]


def main(interval='1d', num_cryptos=10, historical_days=90, output_dir='crypto_analysis_results', api_key=None):
    os.makedirs(output_dir, exist_ok=True)

    analyzer = CryptoAnalyzer(base_dir=output_dir, api_key=api_key)
    visualizer = CryptoDataVisualizer(analyzer)

    analyzer.get_fear_greed_index(days=historical_days)

    crypto_list = analyzer.get_top_cryptos_by_market_cap(limit=num_cryptos)

    if not crypto_list:
        all_cryptos = analyzer.get_crypto_list()
        crypto_list = all_cryptos[:num_cryptos] if all_cryptos else []

    if not crypto_list:
        return

    all_data = {}

    for idx, crypto in enumerate(crypto_list, 1):
        crypto_id = crypto['id']
        symbol = crypto.get('symbol', '')
        name = crypto.get('name', crypto_id)

        try:
            if symbol:
                analyzer.get_funding_rates(symbol)

            historical_success = analyzer.collect_historical_data(crypto_id, symbol, interval, historical_days)

            if historical_success:
                data = analyzer.get_technical_indicators(crypto_id, interval)

                if data is not None and not data.empty:
                    data['name'] = name
                    data['symbol'] = symbol
                    all_data[crypto_id] = data
                    visualizer.visualize_data(data, crypto_id, interval)
                    visualizer.export_data_to_excel(data, crypto_id, interval)

                    # Afficher un résumé des dernières données
                    last_data = data.iloc[-1]

                    print(f"\n{name} ({symbol}):")
                    print(f"  Prix: ${last_data['current_price_usd']:.2f}")

                    if 'pct_from_ath' in last_data:
                        print(f"  ATH: {last_data['pct_from_ath']:.1f}% du maximum historique")

                    if 'RSI' in last_data:
                        rsi_status = last_data['RSI']
                        rsi_trend = "↑" if last_data['RSI_trend'] > 0 else "↓" if last_data['RSI_trend'] < 0 else "→"
                        print(f"  RSI: {rsi_status:.1f} {rsi_trend}")

                    if 'BB_width' in last_data:
                        bb_status = last_data['BB_width']
                        bb_trend = "↑" if last_data['BB_width_trend'] > 0 else "↓" if last_data[
                                                                                          'BB_width_trend'] < 0 else "→"
                        if 'BB_squeeze' in last_data and last_data['BB_squeeze'] == 1:
                            bb_info = " (Resserrement significatif)"
                        else:
                            bb_info = ""
                        print(f"  BB Width: {bb_status:.4f} {bb_trend}{bb_info}")

                    if 'fear_greed_value' in last_data and not pd.isna(last_data['fear_greed_value']):
                        fg = int(last_data['fear_greed_value'])
                        fg_label = last_data.get('fear_greed_label', '')
                        fg_trend = "↑" if last_data['fear_greed_trend'] > 0 else "↓" if last_data[
                                                                                            'fear_greed_trend'] < 0 else "→"
                        print(f"  Fear & Greed: {fg} - {fg_label} {fg_trend}")

                    if 'funding_rate_avg' in last_data and not pd.isna(last_data['funding_rate_avg']):
                        fr = last_data['funding_rate_avg'] * 100
                        fr_trend = "↑" if last_data['funding_rate_trend'] > 0 else "↓" if last_data[
                                                                                              'funding_rate_trend'] < 0 else "→"
                        print(f"  Funding Rate: {fr:.4f}% {fr_trend}")

        except Exception:
            import traceback
            traceback.print_exc()

        if idx < len(crypto_list):
            time.sleep(analyzer.rate_limit_sleep * random.uniform(1.0, 1.5))

    if all_data:
        visualizer.export_all_data_to_excel(all_data, interval)
        summary_df = visualizer.create_detailed_summary(all_data, interval)

        if not summary_df.empty:
            summary_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = os.path.join(output_dir, f"crypto_data_summary_{interval}_{summary_timestamp}.xlsx")
            summary_df.to_excel(summary_filename, index=False)
            print(f"\nRésumé détaillé exporté: {summary_filename}")


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