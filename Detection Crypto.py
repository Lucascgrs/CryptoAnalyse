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
    """
    Classe principale pour récupérer, stocker et analyser les données de cryptomonnaies
    """

    def __init__(self, base_dir: str = 'crypto_analysis_results', api_key: str = None):
        self.base_dir = base_dir
        self.api_base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        self.rate_limit_sleep = 3.0
        self.data_cache = {}

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
            print(f"Limite de taux atteinte. Attente de {retry_after} secondes.")
            time.sleep(retry_after + random.uniform(0.5, 2.0))
            raise requests.exceptions.RequestException(response=response)

        if response.status_code != 200:
            print(f"Erreur API ({response.status_code}): {url}")

        response.raise_for_status()
        return response.json()

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

            # Reformater pour correspondre au format utilisé par le reste du code
            formatted_result = [
                {
                    "id": coin["id"],
                    "symbol": coin["symbol"],
                    "name": coin["name"],
                    "market_cap": coin.get("market_cap"),
                    "volume": coin.get("total_volume"),
                    "price": coin.get("current_price")
                }
                for coin in result
            ]

            print(f"Liste de {len(formatted_result)} cryptomonnaies triées par market cap récupérée")
            return formatted_result
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des crypto par market cap: {e}")
            return []

    def get_crypto_list(self) -> List[Dict]:
        try:
            url = f"{self.api_base_url}/coins/list"
            result = self.make_api_request(url)
            print(f"Liste de {len(result)} cryptomonnaies récupérée avec succès")
            return result
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération de la liste des cryptomonnaies: {e}")
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

            # Extraire les caractéristiques pertinentes
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
                'low_24h_usd': market_data.get('low_24h', {}).get('usd')
            }

            print(f"Données récupérées avec succès pour {crypto_id}")
            return crypto_data

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des données pour {crypto_id}: {e}")
            return None

    def parse_time_interval(self, interval: str) -> Tuple[str, int]:
        # Mapping des intervalles vers les valeurs d'API
        interval_mapping = {
            # Minutes
            '1m': ('minutely', 1),  # 1 jour max pour les données par minute
            '3m': ('minutely', 1),
            '5m': ('minutely', 1),
            '15m': ('minutely', 1),
            '30m': ('minutely', 1),
            # Heures
            '1h': ('hourly', 90),  # 90 jours max pour les données par heure
            '2h': ('hourly', 90),
            '4h': ('hourly', 90),
            '6h': ('hourly', 90),
            '12h': ('hourly', 90),
            # Jours et semaines
            '1d': ('daily', 365),  # 365 jours max pour les données quotidiennes
            '3d': ('daily', 365),
            '1w': ('daily', 365),
        }

        if interval not in interval_mapping:
            print(f"Intervalle non reconnu: {interval}, utilisation de '1d' par défaut")
            return interval_mapping['1d']

        return interval_mapping[interval]

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

            print(f"Récupération des données historiques pour {crypto_id} (intervalle: {interval}, jours: {days})...")
            data = self.make_api_request(url, params)

            # Extraire les prix, capitalisation et volumes
            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            total_volumes = data.get('total_volumes', [])

            if not prices:
                print(f"Aucune donnée historique de prix pour {crypto_id}")
                return None

            # Créer un DataFrame à partir des données historiques
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

            df = pd.DataFrame(historical_data)
            print(f"Données historiques récupérées pour {crypto_id}: {len(df)} entrées")
            return df

        except Exception as e:
            print(f"Erreur lors de la récupération des données historiques pour {crypto_id}: {e}")
            return None

    def get_crypto_dir(self, crypto_id: str) -> str:
        crypto_dir = os.path.join(self.base_dir, crypto_id)
        os.makedirs(crypto_dir, exist_ok=True)
        return crypto_dir

    def collect_historical_data(self, crypto_id: str, interval: str = '1d', days: int = 60) -> bool:
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

            # Nettoyage des données
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.drop(columns=['date'])
                df = df.sort_values('timestamp')

            self.data_cache[f"{crypto_id}_{interval}"] = df
            print(f"Données historiques collectées pour {crypto_id} (intervalle: {interval}): {len(df)} entrées")
            return True

        except Exception as e:
            print(f"Erreur lors de la collecte des données historiques pour {crypto_id}: {e}")
            return False

    def get_technical_indicators(self, crypto_id: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        try:
            # Vérifier si le module TA est installé
            try:
                import ta
            except ImportError:
                print("Module 'ta' non installé. Installez-le avec 'pip install ta'")
                return None

            # Récupérer les données depuis le cache
            cache_key = f"{crypto_id}_{interval}"
            df = self.data_cache.get(cache_key)

            if df is None or df.empty:
                print(f"Aucune donnée trouvée dans le cache pour {crypto_id} (intervalle: {interval})")
                return None

            # Préparation des données
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df = df.dropna(subset=['current_price_usd'])

            if len(df) < 14:
                print(f"Données insuffisantes pour calculer les indicateurs de {crypto_id}")
                return None

            # Calcul des indicateurs techniques
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

                # Ajouter l'indicateur de resserrement des bandes
                # Plus cette valeur est faible, plus les bandes sont resserrées
                df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']

                # Calculer le z-score du resserrement (normalisation)
                # Un z-score négatif élevé indique un resserrement important par rapport à la moyenne historique
                rolling_mean = df['BB_width'].rolling(window=20).mean()
                rolling_std = df['BB_width'].rolling(window=20).std()
                df['BB_squeeze'] = (df['BB_width'] - rolling_mean) / rolling_std

                # Indicateur binaire de resserrement extrême (percentile 10)
                df['BB_extreme_squeeze'] = 0
                if len(df) >= 60:  # Besoin d'assez de données historiques
                    threshold = df['BB_width'].quantile(0.10)
                    df.loc[df['BB_width'] < threshold, 'BB_extreme_squeeze'] = 1

            print(f"Indicateurs techniques calculés pour {crypto_id} (intervalle: {interval})")
            return df

        except Exception as e:
            print(f"Erreur lors du calcul des indicateurs techniques pour {crypto_id}: {e}")
            return None


class CryptoTradingSignals:
    """
    Classe pour générer et visualiser des signaux de trading à partir des données de cryptomonnaies
    """

    def __init__(self, analyzer: CryptoAnalyzer):
        self.analyzer = analyzer

    def generate_signals(self, crypto_id: str, interval: str = '1d'):
        # Obtenir les données avec indicateurs techniques
        df = self.analyzer.get_technical_indicators(crypto_id, interval)

        if df is None or len(df) < 30:
            print(f"Données insuffisantes pour générer des signaux pour {crypto_id}")
            return None

        # Initialiser les signaux
        signals = df.copy()
        for col in ['signal', 'signal_ma_cross', 'signal_rsi', 'signal_macd', 'signal_bb', 'signal_squeeze']:
            signals[col] = 0

        # 1. Signal de croisement de moyennes mobiles (MA)
        if 'MA20' in signals.columns and 'MA50' in signals.columns:
            # Filtrer les valeurs null pour éviter les erreurs de comparaison
            valid_data = signals['MA20'].notnull() & signals['MA50'].notnull()

            # Signal d'achat: MA20 croise MA50 par le haut
            signals.loc[
                valid_data &
                (signals['MA20'] > signals['MA50']) &
                (signals['MA20'].shift(1) <= signals['MA50'].shift(1)),
                'signal_ma_cross'
            ] = 1

            # Signal de vente: MA20 croise MA50 par le bas
            signals.loc[
                valid_data &
                (signals['MA20'] < signals['MA50']) &
                (signals['MA20'].shift(1) >= signals['MA50'].shift(1)),
                'signal_ma_cross'
            ] = -1

        # 2. Signal RSI
        if 'RSI' in signals.columns:
            valid_rsi = signals['RSI'].notnull()

            # Signal d'achat: RSI remonte au-dessus de 30 (survente)
            signals.loc[
                valid_rsi &
                (signals['RSI'] > 30) &
                (signals['RSI'].shift(1) <= 30),
                'signal_rsi'
            ] = 1

            # Signal de vente: RSI descend sous 70 (surachat)
            signals.loc[
                valid_rsi &
                (signals['RSI'] < 70) &
                (signals['RSI'].shift(1) >= 70),
                'signal_rsi'
            ] = -1

        # 3. Signal MACD
        if all(col in signals.columns and signals[col].notnull().any() for col in ['MACD', 'MACD_signal']):
            valid_macd = signals['MACD'].notnull() & signals['MACD_signal'].notnull()

            # Signal d'achat: MACD croise la ligne de signal par le haut
            signals.loc[
                valid_macd &
                (signals['MACD'] > signals['MACD_signal']) &
                (signals['MACD'].shift(1) <= signals['MACD_signal'].shift(1)),
                'signal_macd'
            ] = 1

            # Signal de vente: MACD croise la ligne de signal par le bas
            signals.loc[
                valid_macd &
                (signals['MACD'] < signals['MACD_signal']) &
                (signals['MACD'].shift(1) >= signals['MACD_signal'].shift(1)),
                'signal_macd'
            ] = -1

        # 4. Signal Bollinger Bands
        if all(col in signals.columns for col in ['current_price_usd', 'BB_low', 'BB_high']):
            valid_bb = signals['BB_low'].notnull() & signals['BB_high'].notnull()

            # Signal d'achat: le prix touche la bande inférieure
            signals.loc[
                valid_bb &
                (signals['current_price_usd'] <= signals['BB_low']) &
                (signals['current_price_usd'].shift(1) > signals['BB_low'].shift(1)),
                'signal_bb'
            ] = 1

            # Signal de vente: le prix touche la bande supérieure
            signals.loc[
                valid_bb &
                (signals['current_price_usd'] >= signals['BB_high']) &
                (signals['current_price_usd'].shift(1) < signals['BB_high'].shift(1)),
                'signal_bb'
            ] = -1

        # 5. Signal de resserrement des bandes de Bollinger
        if 'BB_extreme_squeeze' in signals.columns:
            # Signal d'alerte de resserrement extrême
            signals.loc[
                signals['BB_extreme_squeeze'] == 1,
                'signal_squeeze'
            ] = 1

        # Combiner tous les signaux
        signal_cols = [col for col in signals.columns if col.startswith('signal_')]
        if signal_cols:
            # Compter les signaux disponibles et calculer la force moyenne
            available_signals = len([col for col in signal_cols if not signals[col].isna().all()])

            if available_signals > 0:
                signals['signal'] = signals[signal_cols].sum(axis=1)
                signals['signal_strength'] = signals['signal'] / available_signals

                # Décisions finales
                signals['decision'] = 'HOLD'
                signals.loc[signals['signal_strength'] >= 0.5, 'decision'] = 'BUY'
                signals.loc[signals['signal_strength'] <= -0.5, 'decision'] = 'SELL'

        return signals

    def plot_signals(self, signals_df, crypto_id, interval):
        if signals_df is None or 'decision' not in signals_df.columns:
            print(f"Données de signaux invalides pour {crypto_id}")
            return

        # Créer le répertoire pour la crypto
        crypto_dir = self.analyzer.get_crypto_dir(crypto_id)

        plt.figure(figsize=(14, 10))

        # Graphique principal: prix, moyennes mobiles et bandes de Bollinger
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(signals_df['timestamp'], signals_df['current_price_usd'], label='Prix', color='blue')

        # Ajouter les moyennes mobiles
        for ma, color in [('MA20', 'orange'), ('MA50', 'red'), ('MA200', 'purple')]:
            if ma in signals_df.columns and not signals_df[ma].isna().all():
                ax1.plot(signals_df['timestamp'], signals_df[ma], label=ma, color=color, alpha=0.7)

        # Ajouter les bandes de Bollinger
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

        # Marquer les signaux d'achat et de vente
        buy_signals = signals_df[signals_df['decision'] == 'BUY']
        sell_signals = signals_df[signals_df['decision'] == 'SELL']
        squeeze_signals = signals_df[signals_df['signal_squeeze'] > 0]

        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['current_price_usd'],
                        marker='^', color='green', s=100, label='Achat')

        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['current_price_usd'],
                        marker='v', color='red', s=100, label='Vente')

        if not squeeze_signals.empty:
            ax1.scatter(squeeze_signals['timestamp'], squeeze_signals['current_price_usd'],
                        marker='*', color='purple', s=120, label='Resserrement BB')

        ax1.set_title(f'{crypto_id.upper()} - Signaux de Trading ({interval})')
        ax1.set_ylabel('Prix (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Sous-graphique pour le RSI
        if 'RSI' in signals_df.columns and not signals_df['RSI'].isna().all():
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(signals_df['timestamp'], signals_df['RSI'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax2.fill_between(signals_df['timestamp'], 70, 100, color='red', alpha=0.1)
            ax2.fill_between(signals_df['timestamp'], 0, 30, color='green', alpha=0.1)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True)

        # Sous-graphique combiné pour le MACD et la largeur des bandes de Bollinger
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)

        # Afficher le MACD
        if all(col in signals_df.columns for col in ['MACD', 'MACD_signal']):
            ax3.plot(signals_df['timestamp'], signals_df['MACD'], label='MACD', color='blue')
            ax3.plot(signals_df['timestamp'], signals_df['MACD_signal'], label='Signal', color='orange')

            # Histogramme de différence MACD
            if 'MACD_diff' in signals_df.columns:
                diff = signals_df['MACD_diff'].fillna(0)
                ax3.bar(signals_df['timestamp'], diff,
                        color=np.where(diff > 0, 'g', 'r'), alpha=0.5)

        # Créer un deuxième axe pour la largeur des bandes de Bollinger
        if 'BB_width' in signals_df.columns:
            ax4 = ax3.twinx()
            ax4.plot(signals_df['timestamp'], signals_df['BB_width'], color='purple',
                     linestyle='--', label='BB Width', alpha=0.7)

            # Marquer les points de resserrement extrême
            if 'BB_extreme_squeeze' in signals_df.columns:
                squeeze_points = signals_df[signals_df['BB_extreme_squeeze'] == 1]
                if not squeeze_points.empty:
                    ax4.scatter(squeeze_points['timestamp'], squeeze_points['BB_width'],
                                marker='*', color='red', s=80, label='Squeeze')

            ax4.set_ylabel('BB Width', color='purple')
            ax4.tick_params(axis='y', colors='purple')

            # Ajouter les légendes des deux axes
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax4.get_legend_handles_labels()
            ax3.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax3.legend()

        ax3.set_ylabel('MACD')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax3.grid(True)

        plt.tight_layout()

        # Enregistrer l'image dans le répertoire de la crypto
        filename = os.path.join(crypto_dir, f"{crypto_id}_trading_signals_{interval}.png")
        plt.savefig(filename)
        print(f"Graphique des signaux enregistré sous '{filename}'")
        plt.close()

    def export_signals_to_excel(self, signals_df, crypto_id, interval):
        if signals_df is None or signals_df.empty:
            return

        # Créer le répertoire pour la crypto si nécessaire
        crypto_dir = self.analyzer.get_crypto_dir(crypto_id)

        filename = os.path.join(crypto_dir, f"{crypto_id}_trading_signals_{interval}.xlsx")
        signals_df.to_excel(filename, index=False)
        print(f"Signaux de trading exportés vers '{filename}'")

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

        print(f"Tous les signaux de trading exportés vers '{filename}'")

    def create_detailed_summary(self, all_signals, interval):
        if not all_signals:
            return pd.DataFrame()

        summary_data = []

        for crypto_id, signals_df in all_signals.items():
            if signals_df is None or signals_df.empty:
                continue

            # Obtenir la dernière entrée
            last_signal = signals_df.iloc[-1]

            # Informations de base
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

            # Ajouter les signaux individuels
            signal_cols = [col for col in last_signal.index if col.startswith('signal_')]
            for col in signal_cols:
                crypto_info[col] = last_signal.get(col, 0)

            # Ajouter le signal global et la décision
            crypto_info['signal_strength'] = last_signal.get('signal_strength', 0)
            crypto_info['decision'] = last_signal.get('decision', 'UNKNOWN')

            # Ajouter les indicateurs de resserrement des bandes
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
        base_cols = ['crypto_id', 'name', 'symbol', 'price_usd', 'market_cap_usd',
                     'volume_24h_usd', 'timestamp', 'interval']
        bb_cols = [col for col in summary_df.columns if col.startswith('BB_')]
        signal_cols = [col for col in summary_df.columns if col.startswith('signal_')]
        final_cols = base_cols + bb_cols + signal_cols + ['decision']

        # Réorganiser les colonnes (uniquement celles qui existent)
        existing_cols = [col for col in final_cols if col in summary_df.columns]
        summary_df = summary_df[existing_cols]

        return summary_df


def main(interval='1d', num_cryptos=10, historical_days=90, output_dir='crypto_analysis_results', api_key=None):
    """
    Analyse des cryptomonnaies et génération de signaux de trading
    """
    print(f"\n===== ANALYSE DE CRYPTOMONNAIES (Intervalle: {interval}) =====\n")

    # Préparation du répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Initialisation des classes
    analyzer = CryptoAnalyzer(base_dir=output_dir, api_key=api_key)
    signals_generator = CryptoTradingSignals(analyzer)

    # Récupérer les cryptos par capitalisation de marché
    crypto_list = analyzer.get_top_cryptos_by_market_cap(limit=num_cryptos)

    if not crypto_list:
        print("Impossible de récupérer les cryptomonnaies par capitalisation. Utilisation de la liste générale.")
        all_cryptos = analyzer.get_crypto_list()
        crypto_list = all_cryptos[:num_cryptos] if all_cryptos else []

    if not crypto_list:
        print("Aucune cryptomonnaie disponible pour l'analyse.")
        return

    print(f"1. Analyse de {len(crypto_list)} cryptomonnaies avec intervalle {interval}")

    # Stockage des résultats
    all_signals = {}

    # Traitement des cryptomonnaies
    for idx, crypto in enumerate(crypto_list, 1):
        crypto_id = crypto['id']
        symbol = crypto.get('symbol', '')
        name = crypto.get('name', crypto_id)

        print(f"\nTraitement de {idx}/{len(crypto_list)}: {name} ({symbol})...")

        try:
            # 1. Collecte des données historiques
            historical_success = analyzer.collect_historical_data(crypto_id, interval, historical_days)

            if historical_success:
                # 2. Génération des indicateurs et signaux
                signals = signals_generator.generate_signals(crypto_id, interval)

                if signals is not None and not signals.empty:
                    # Ajouter les métadonnées
                    signals['name'] = name
                    signals['symbol'] = symbol

                    # Stocker les signaux
                    all_signals[crypto_id] = signals

                    # Visualisation et export
                    signals_generator.plot_signals(signals, crypto_id, interval)
                    signals_generator.export_signals_to_excel(signals, crypto_id, interval)

                    # Affichage du signal actuel
                    last_signal = signals.iloc[-1]

                    # Information sur le resserrement des bandes de Bollinger
                    bb_info = ""
                    if 'BB_width' in last_signal.index and not pd.isna(last_signal['BB_width']):
                        bb_info = f" | BB Width: {last_signal['BB_width']:.4f}"
                        if 'BB_extreme_squeeze' in last_signal.index and last_signal['BB_extreme_squeeze'] == 1:
                            bb_info += " (RESSERREMENT IMPORTANT)"

                    print(
                        f"  Signal actuel: {last_signal.get('decision', 'UNKNOWN')} (force: {last_signal.get('signal_strength', 0):.2f}){bb_info}")
            else:
                print(f"  Échec de la collecte des données historiques")

        except Exception as e:
            print(f"  Erreur lors du traitement de {crypto_id}: {e}")
            import traceback
            traceback.print_exc()

        # Pause entre les cryptos
        if idx < len(crypto_list):
            wait_time = analyzer.rate_limit_sleep * random.uniform(1.0, 1.5)
            time.sleep(wait_time)

    # Export de tous les signaux
    if all_signals:
        print("\n3. Exportation des signaux...")
        signals_generator.export_all_signals_to_excel(all_signals, interval)

        # Création du résumé détaillé
        print("\n4. Création du résumé détaillé...")
        summary_df = signals_generator.create_detailed_summary(all_signals, interval)

        if not summary_df.empty:
            # Export du résumé
            summary_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = os.path.join(output_dir, f"crypto_signals_summary_{interval}_{summary_timestamp}.xlsx")
            summary_df.to_excel(summary_filename, index=False)
            print(f"Résumé détaillé exporté vers {summary_filename}")

    print(f"\n===== ANALYSE TERMINÉE (Intervalle: {interval}) =====")


if __name__ == "__main__":
    # Configuration de l'analyse
    INTERVALS = ['1d']  # Intervalles disponibles: '1m', '5m', '15m', '1h', '4h', '1d', etc.
    NUM_CRYPTOS = 2  # Nombre de cryptomonnaies à analyser
    HISTORICAL_DAYS = 90  # Jours d'historique à récupérer
    OUTPUT_DIR = 'crypto_analysis_results'
    API_KEY = None  # Votre clé API CoinGecko Pro (optionnelle)

    try:
        for interval in INTERVALS:
            main(interval=interval, num_cryptos=NUM_CRYPTOS, historical_days=HISTORICAL_DAYS, output_dir=OUTPUT_DIR, api_key=API_KEY)

            if interval != INTERVALS[-1]:
                time.sleep(5)

    except Exception as e:
        print(f"Erreur fatale: {e}")
        import traceback

        traceback.print_exc()