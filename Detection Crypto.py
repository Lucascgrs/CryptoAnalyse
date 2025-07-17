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

    #btc dominance

    def get_altcoin_season_index(self, days=90):
        """
        Calcule l'Altcoin Season Index pour les derniers X jours
        en utilisant des données historiques de CryptoCompare
        """
        try:
            # 1. Récupérer l'historique de Bitcoin
            hist_url = "https://min-api.cryptocompare.com/data/v2/histoday"

            # Paramètres pour Bitcoin
            btc_params = {
                "fsym": "BTC",
                "tsym": "USD",
                "limit": days,  # Nombre de jours d'historique
                "extraParams": "CryptoAnalyzer"
            }

            btc_response = requests.get(hist_url, params=btc_params)
            if btc_response.status_code != 200:
                print(f"Erreur API pour l'historique BTC: {btc_response.status_code}")
                return self._get_current_altcoin_index_only()

            btc_data = btc_response.json().get('Data', {}).get('Data', [])
            if not btc_data:
                print("Données historiques BTC non disponibles")
                return self._get_current_altcoin_index_only()

            # 2. Récupérer l'historique des principales altcoins
            # Récupérer les top altcoins
            top_cryptos = self.get_top_cryptos_by_market_cap(limit=10)
            if not top_cryptos:
                print("Impossible de récupérer la liste des cryptomonnaies")
                return self._get_current_altcoin_index_only()

            # Sélectionner 3 altcoins représentatifs (pour limiter les requêtes)
            alt_symbols = ["ETH"]  # Toujours inclure ETH
            for coin in top_cryptos:
                if coin['symbol'].upper() not in ["BTC", "ETH", "USDT", "USDC"] and len(alt_symbols) < 3:
                    alt_symbols.append(coin['symbol'].upper())

            # Récupérer les données pour chaque altcoin
            alt_data = {}
            for symbol in alt_symbols:
                try:
                    alt_params = {
                        "fsym": symbol,
                        "tsym": "USD",
                        "limit": days,
                        "extraParams": "CryptoAnalyzer"
                    }

                    alt_response = requests.get(hist_url, params=alt_params)
                    if alt_response.status_code == 200:
                        alt_data[symbol] = alt_response.json().get('Data', {}).get('Data', [])
                        time.sleep(0.5)  # Pause pour respecter les limites d'API
                except Exception as e:
                    print(f"Erreur pour {symbol}: {e}")

            # 3. Calculer l'indice Altcoin Season pour chaque jour
            results = []
            window_size = 14  # Fenêtre glissante de 14 jours pour le calcul

            for i in range(window_size, len(btc_data)):
                date = datetime.fromtimestamp(btc_data[i]['time'])

                # Calculer la performance de BTC sur la fenêtre
                btc_start = btc_data[i - window_size]['close']
                btc_end = btc_data[i]['close']
                btc_perf = ((btc_end / btc_start) - 1) * 100

                # Calculer les performances des altcoins sur la même fenêtre
                alt_count = 0
                outperform_count = 0

                for symbol, data in alt_data.items():
                    if i < len(data) and i - window_size >= 0 and i - window_size < len(data):
                        alt_start = data[i - window_size]['close']
                        alt_end = data[i]['close']
                        alt_perf = ((alt_end / alt_start) - 1) * 100

                        alt_count += 1
                        if alt_perf > btc_perf:
                            outperform_count += 1

                if alt_count > 0:
                    outperform_pct = (outperform_count / alt_count) * 100
                    results.append({
                        'timestamp': date,
                        'altcoin_season_index': outperform_pct,
                        'btc_performance': btc_perf,
                        'sample_size': alt_count
                    })

            # Créer le DataFrame final
            if not results:
                print("Pas assez de données pour calculer l'historique de l'Altcoin Season Index")
                return self._get_current_altcoin_index_only()

            altcoin_df = pd.DataFrame(results)

            # Calculer la tendance
            altcoin_df['altcoin_index_change'] = altcoin_df['altcoin_season_index'].diff()
            altcoin_df['altcoin_index_trend'] = np.sign(altcoin_df['altcoin_index_change'])

            # Stocker les données
            self.altcoin_season_data = altcoin_df

            # Afficher l'indice actuel
            current_index = altcoin_df['altcoin_season_index'].iloc[-1]
            season = "Altcoin Season" if current_index > 75 else "Bitcoin Season" if current_index < 25 else "Neutral"
            print(f"Altcoin Season Index: {current_index:.1f} - {season}")

            return altcoin_df

        except Exception as e:
            print(f"Erreur lors du calcul de l'Altcoin Season Index historique: {e}")
            import traceback
            traceback.print_exc()
            return self._get_current_altcoin_index_only()

    def _get_current_altcoin_index_only(self):
        """
        Méthode pour obtenir uniquement l'indice actuel comme solution de repli
        """
        try:
            # Simplification de la méthode originale pour juste obtenir l'indice actuel
            base_url = "https://min-api.cryptocompare.com/data/pricemultifull"
            alt_symbols = ["ETH", "BNB", "SOL", "XRP", "ADA"]

            altcoin_list = ",".join(alt_symbols)

            params = {
                "fsyms": f"BTC,{altcoin_list}",
                "tsyms": "USD",
                "extraParams": "CryptoAnalyzer"
            }

            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                print(f"Erreur API: {response.status_code}")
                return None

            data = response.json()
            raw_data = data.get('RAW', {})

            if not raw_data or 'BTC' not in raw_data:
                return None

            btc_change_24h = raw_data['BTC']['USD'].get('CHANGEPCT24HOUR', 0)

            outperforming_count = 0
            valid_alts = 0

            for alt in alt_symbols:
                if alt in raw_data and 'USD' in raw_data[alt]:
                    change_24h = raw_data[alt]['USD'].get('CHANGEPCT24HOUR', 0)
                    valid_alts += 1

                    if change_24h > btc_change_24h:
                        outperforming_count += 1

            if valid_alts == 0:
                return None

            outperform_percentage = (outperforming_count / valid_alts) * 100

            today = datetime.now()
            altcoin_df = pd.DataFrame([{
                'timestamp': today,
                'altcoin_season_index': outperform_percentage,
                'btc_performance': btc_change_24h,
                'sample_size': valid_alts,
                'altcoin_index_trend': 0
            }])

            self.altcoin_season_data = altcoin_df

            season = "Altcoin Season" if outperform_percentage > 75 else "Bitcoin Season" if outperform_percentage < 25 else "Neutral"
            print(f"Altcoin Season Index (actuel uniquement): {outperform_percentage:.1f} - {season}")

            return altcoin_df
        except Exception as e:
            print(f"Erreur lors du calcul de l'indice Altcoin Season actuel: {e}")
            return None

    def get_long_short_ratio(self, symbol='BTCUSDT'):
        """
        Récupère le ratio long/short des positions ouvertes sur Binance Futures
        """
        try:
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': '1d',
                'limit': 30  # 30 jours d'historique
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                # Transformer en DataFrame
                ratio_df = pd.DataFrame(data)
                ratio_df['timestamp'] = pd.to_datetime(ratio_df['timestamp'], unit='ms')
                ratio_df['long_short_ratio'] = ratio_df['longShortRatio'].astype(float)
                ratio_df['long_account'] = ratio_df['longAccount'].astype(float)
                ratio_df['short_account'] = ratio_df['shortAccount'].astype(float)

                # Calculer la tendance
                ratio_df['long_short_change'] = ratio_df['long_short_ratio'].diff()
                ratio_df['long_short_trend'] = np.sign(ratio_df['long_short_change'])

                # Stocker les données
                symbol_key = symbol.lower()
                if not hasattr(self, 'long_short_data'):
                    self.long_short_data = {}
                self.long_short_data[symbol_key] = ratio_df

                return ratio_df

            return None
        except Exception as e:
            print(f"Erreur lors de la récupération du ratio long/short: {e}")
            return None

    def get_position_amounts(self, symbol='BTCUSDT'):
        """
        Récupère le montant des positions longues et courtes via le volume d'achat/vente des takers
        ainsi que le volume total des positions ouvertes par direction
        """
        try:
            # Récupérer le ratio de volume taker achat/vente
            url_taker = "https://fapi.binance.com/futures/data/takerlongshortRatio"
            params_taker = {
                'symbol': symbol,
                'period': '1d',
                'limit': 30  # 30 jours d'historique
            }

            # Récupérer la répartition des positions ouvertes
            url_positions = "https://fapi.binance.com/futures/data/openInterestHist"
            params_positions = {
                'symbol': symbol,
                'period': '1d',
                'limit': 30
            }

            response_taker = requests.get(url_taker, params=params_taker)
            response_positions = requests.get(url_positions, params=params_positions)

            if response_taker.status_code == 200 and response_positions.status_code == 200:
                taker_data = response_taker.json()
                position_data = response_positions.json()

                # Transformer les données taker en DataFrame
                taker_df = pd.DataFrame(taker_data)
                taker_df['timestamp'] = pd.to_datetime(taker_df['timestamp'], unit='ms')
                taker_df['buy_sell_ratio'] = taker_df['buySellRatio'].astype(float)
                taker_df['buy_volume'] = taker_df['buyVol'].astype(float)
                taker_df['sell_volume'] = taker_df['sellVol'].astype(float)

                # Calculer la tendance
                taker_df['volume_ratio_change'] = taker_df['buy_sell_ratio'].diff()
                taker_df['volume_ratio_trend'] = np.sign(taker_df['volume_ratio_change'])

                # Transformer les données de positions en DataFrame
                pos_df = pd.DataFrame(position_data)
                pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], unit='ms')
                pos_df['open_interest'] = pos_df['sumOpenInterest'].astype(float)

                # Combiner les deux dataframes
                result_df = pd.merge(taker_df, pos_df[['timestamp', 'open_interest']], on='timestamp', how='left')

                # Calculer les montants estimés de positions longues et courtes
                # Note: C'est une approximation basée sur le ratio buy/sell et l'open interest total
                result_df['long_value_ratio'] = result_df['buy_sell_ratio'] / (1 + result_df['buy_sell_ratio'])
                result_df['short_value_ratio'] = 1 - result_df['long_value_ratio']

                result_df['long_position_value'] = result_df['open_interest'] * result_df['long_value_ratio']
                result_df['short_position_value'] = result_df['open_interest'] * result_df['short_value_ratio']

                # Stocker les données
                symbol_key = symbol.lower()
                if not hasattr(self, 'position_amount_data'):
                    self.position_amount_data = {}
                self.position_amount_data[symbol_key] = result_df

                return result_df

            return None
        except Exception as e:
            print(f"Erreur lors de la récupération des montants de positions: {e}")
            return None

    def get_put_call_ratio(self, currency='BTC'):
        """
        Récupère le ratio put/call des options crypto (Deribit)
        Note: Cette API peut nécessiter une authentification
        """
        try:
            # Exemple avec l'API publique Deribit
            url = f"https://www.deribit.com/api/v2/public/get_put_call_ratio?currency={currency}"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json().get('result', [])

                # Transformer en DataFrame
                pc_ratio_df = pd.DataFrame(data)
                pc_ratio_df['timestamp'] = pd.to_datetime(pc_ratio_df['timestamp'], unit='ms')
                pc_ratio_df['put_call_ratio'] = pc_ratio_df['put_call_ratio'].astype(float)

                # Calculer la tendance
                pc_ratio_df['put_call_change'] = pc_ratio_df['put_call_ratio'].diff()
                pc_ratio_df['put_call_trend'] = np.sign(pc_ratio_df['put_call_change'])

                # Stocker les données
                if not hasattr(self, 'put_call_data'):
                    self.put_call_data = {}
                self.put_call_data[currency.lower()] = pc_ratio_df

                return pc_ratio_df

            return None
        except Exception as e:
            print(f"Erreur lors de la récupération du ratio put/call: {e}")
            return None

    def get_open_interest(self, symbol='BTCUSDT'):
        """
        Récupère l'Open Interest et son évolution depuis Binance Futures
        """
        try:
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = {
                'symbol': symbol,
                'period': '1d',
                'limit': 30  # 30 jours d'historique
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                # Transformer en DataFrame
                oi_df = pd.DataFrame(data)
                oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df['open_interest'] = oi_df['sumOpenInterest'].astype(float)
                oi_df['open_interest_value'] = oi_df['sumOpenInterestValue'].astype(float)

                # Calculer la tendance
                oi_df['open_interest_change'] = oi_df['open_interest'].diff()
                oi_df['open_interest_trend'] = np.sign(oi_df['open_interest_change'])

                # Calculer le changement en pourcentage
                oi_df['open_interest_pct_change'] = oi_df['open_interest'].pct_change() * 100

                # Stocker les données
                symbol_key = symbol.lower()
                if not hasattr(self, 'open_interest_data'):
                    self.open_interest_data = {}
                self.open_interest_data[symbol_key] = oi_df

                return oi_df

            return None
        except Exception as e:
            print(f"Erreur lors de la récupération de l'Open Interest: {e}")
            return None

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
        """
        Collecte les données historiques et intègre tous les indicateurs disponibles
        """
        try:
            # Récupérer les données historiques de base
            df = self.get_historical_market_data(crypto_id, interval, days)
            if df is None or df.empty:
                return False

            # Ajouter les données actuelles si nécessaire
            current_data = self.get_crypto_data(crypto_id)
            if current_data:
                current_df = pd.DataFrame([current_data])
                latest_timestamp = pd.to_datetime(df['timestamp']).max() if 'timestamp' in df.columns else None
                current_timestamp = pd.to_datetime(current_data['timestamp'])

                if latest_timestamp is None or current_timestamp.date() != latest_timestamp.date():
                    df = pd.concat([df, current_df], ignore_index=True)

            # Nettoyage et formatage des données
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.drop(columns=['date'])
                df = df.sort_values('timestamp')

            # ==== MÉTRIQUES DE BASE ====

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

            # ==== INDICATEUR FEAR & GREED ====

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

                # Tendance du Fear & Greed
                df['fear_greed_change'] = df['fear_greed_value'].diff()
                df['fear_greed_trend'] = np.sign(df['fear_greed_change'])

                df = df.drop(['date_str', 'fg_date'], axis=1, errors='ignore')

            # ==== FUNDING RATES ====

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

                    # Tendance du funding rate
                    funding_daily['funding_rate_prev'] = funding_daily['funding_rate_avg'].shift(1)
                    funding_daily['funding_rate_change'] = funding_daily['funding_rate_avg'] - funding_daily[
                        'funding_rate_prev']
                    funding_daily['funding_rate_trend'] = np.sign(funding_daily['funding_rate_change'])
                    funding_daily = funding_daily.drop('funding_rate_prev', axis=1)

                    df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                    df = pd.merge(df, funding_daily, on='date_str', how='left')
                    df = df.drop('date_str', axis=1, errors='ignore')

            # ==== NOUVEAUX INDICATEURS ====

            # 1. Bitcoin Dominance - VERSION ROBUSTE
            if hasattr(self, 'btc_dominance_data') and self.btc_dominance_data is not None:
                try:
                    df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                    dom_df = self.btc_dominance_data.copy()
                    dom_df['date_str'] = pd.to_datetime(dom_df['timestamp']).dt.strftime('%Y-%m-%d')

                    # Vérifier les colonnes disponibles
                    available_cols = ['date_str']
                    for col in ['btc_dominance', 'btc_dominance_trend']:
                        if col in dom_df.columns:
                            available_cols.append(col)

                    if len(available_cols) > 1:  # Au moins une colonne de données en plus de date_str
                        df = pd.merge(
                            df,
                            dom_df[available_cols],
                            on='date_str',
                            how='left'
                        )

                    df = df.drop('date_str', axis=1, errors='ignore')
                except Exception as e:
                    print(f"Erreur lors de l'ajout de la dominance Bitcoin: {e}")

            # 2. Altcoin Season Index - VERSION ROBUSTE
            if hasattr(self, 'altcoin_season_data') and self.altcoin_season_data is not None:
                try:
                    df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                    alt_df = self.altcoin_season_data.copy()
                    alt_df['date_str'] = pd.to_datetime(alt_df['timestamp']).dt.strftime('%Y-%m-%d')

                    # Vérifier les colonnes disponibles
                    available_cols = ['date_str']
                    for col in ['altcoin_season_index', 'altcoin_index_trend']:
                        if col in alt_df.columns:
                            available_cols.append(col)

                    if len(available_cols) > 1:  # Au moins une colonne de données en plus de date_str
                        df = pd.merge(
                            df,
                            alt_df[available_cols],
                            on='date_str',
                            how='left'
                        )

                    df = df.drop('date_str', axis=1, errors='ignore')
                except Exception as e:
                    print(f"Erreur lors de l'ajout de l'Altcoin Season Index: {e}")

            # 3. Long/Short Ratio
            symbol_futures = f"{symbol.upper()}USDT" if symbol else None
            if symbol_futures and hasattr(self, 'long_short_data') and symbol_futures.lower() in self.long_short_data:
                df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                ls_df = self.long_short_data[symbol_futures.lower()].copy()
                ls_df['date_str'] = pd.to_datetime(ls_df['timestamp']).dt.strftime('%Y-%m-%d')

                df = pd.merge(
                    df,
                    ls_df[['date_str', 'long_short_ratio', 'long_short_trend', 'long_account', 'short_account']],
                    left_on='date_str',
                    right_on='date_str',
                    how='left'
                )

                df = df.drop('date_str', axis=1, errors='ignore')

            # 3.1 Montants des positions longues/courtes
            if symbol_futures and hasattr(self, 'position_amount_data') and symbol_futures.lower() in self.position_amount_data:
                df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                pos_df = self.position_amount_data[symbol_futures.lower()].copy()
                pos_df['date_str'] = pd.to_datetime(pos_df['timestamp']).dt.strftime('%Y-%m-%d')

                df = pd.merge(
                    df,
                    pos_df[['date_str', 'buy_sell_ratio', 'buy_volume', 'sell_volume',
                            'volume_ratio_trend', 'long_position_value', 'short_position_value']],
                    left_on='date_str',
                    right_on='date_str',
                    how='left'
                )

                # Calculer le ratio de valeur long/short
                df['position_value_ratio'] = df['long_position_value'] / df['short_position_value']

                df = df.drop('date_str', axis=1, errors='ignore')

            # 4. Put/Call Ratio
            crypto_symbol = symbol.upper().replace('USDT', '') if symbol else None
            if crypto_symbol and hasattr(self, 'put_call_data') and crypto_symbol.lower() in self.put_call_data:
                df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                pc_df = self.put_call_data[crypto_symbol.lower()].copy()
                pc_df['date_str'] = pd.to_datetime(pc_df['timestamp']).dt.strftime('%Y-%m-%d')

                df = pd.merge(
                    df,
                    pc_df[['date_str', 'put_call_ratio', 'put_call_trend']],
                    left_on='date_str',
                    right_on='date_str',
                    how='left'
                )

                df = df.drop('date_str', axis=1, errors='ignore')

            # 5. Open Interest
            if symbol_futures and hasattr(self,
                                          'open_interest_data') and symbol_futures.lower() in self.open_interest_data:
                df['date_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                oi_df = self.open_interest_data[symbol_futures.lower()].copy()
                oi_df['date_str'] = pd.to_datetime(oi_df['timestamp']).dt.strftime('%Y-%m-%d')

                df = pd.merge(
                    df,
                    oi_df[['date_str', 'open_interest', 'open_interest_trend', 'open_interest_pct_change']],
                    left_on='date_str',
                    right_on='date_str',
                    how='left'
                )

                df = df.drop('date_str', axis=1, errors='ignore')

            # Stockage dans le cache
            self.data_cache[f"{crypto_id}_{interval}"] = df
            return True

        except Exception as e:
            print(f"Erreur lors de la collecte des données historiques pour {crypto_id}: {e}")
            import traceback
            traceback.print_exc()
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

        # Visualisations existantes
        self._plot_price_and_volume(df, crypto_id, interval, crypto_dir)
        self._plot_technical_indicators(df, crypto_id, interval, crypto_dir)

        if 'fear_greed_value' in df.columns and df['fear_greed_value'].notnull().any():
            self._plot_fear_greed(df, crypto_id, interval, crypto_dir)

        if 'funding_rate_avg' in df.columns and df['funding_rate_avg'].notnull().any():
            self._plot_funding_rates(df, crypto_id, interval, crypto_dir)

        # NOUVELLES VISUALISATIONS

        # Visualisation de la dominance Bitcoin
        if 'btc_dominance' in df.columns and df['btc_dominance'].notnull().any():
            self._plot_btc_dominance(df, crypto_id, interval, crypto_dir)

        # Visualisation du Altcoin Season Index
        if 'altcoin_season_index' in df.columns and df['altcoin_season_index'].notnull().any():
            self._plot_altcoin_season(df, crypto_id, interval, crypto_dir)

        # Visualisation du ratio Long/Short
        if 'long_short_ratio' in df.columns and df['long_short_ratio'].notnull().any():
            self._plot_long_short_ratio(df, crypto_id, interval, crypto_dir)

        # Visualisation du ratio Put/Call
        if 'put_call_ratio' in df.columns and df['put_call_ratio'].notnull().any():
            self._plot_put_call_ratio(df, crypto_id, interval, crypto_dir)

        # Visualisation de l'Open Interest
        if 'open_interest' in df.columns and df['open_interest'].notnull().any():
            self._plot_open_interest(df, crypto_id, interval, crypto_dir)

        # Visualisation des montants de positions long/short
        if all(col in df.columns for col in ['buy_sell_ratio', 'long_position_value', 'short_position_value']):
            self._plot_position_amounts(df, crypto_id, interval, crypto_dir)

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

    def _plot_btc_dominance(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        # Vérifier si nous avons une série temporelle ou un point unique
        if len(df) <= 1:
            # Pour un seul point, utiliser un graphique en barre
            current_date = df['timestamp'].iloc[-1]
            dominance = df['btc_dominance'].iloc[-1]

            ax.bar(current_date, dominance, color='orange', width=5, label='Bitcoin Dominance (%)')
            ax.set_title(f'Bitcoin Dominance - Valeur actuelle: {dominance:.2f}%')

            # Lignes de référence plus visibles
            ax.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Dominance élevée (60%)')
            ax.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='Seuil 50%')
            ax.axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Dominance faible (40%)')

            # Étiquette de valeur sur la barre
            ax.text(current_date, dominance + 2, f"{dominance:.2f}%",
                    ha='center', va='bottom', fontweight='bold')
        else:
            # Pour une série temporelle, utiliser une ligne
            ax.plot(df['timestamp'], df['btc_dominance'], color='orange', marker='o', label='Bitcoin Dominance (%)')

            # Lignes de référence
            ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Dominance élevée (60%)')
            ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Seuil 50%')
            ax.axhline(y=40, color='green', linestyle='--', alpha=0.5, label='Dominance faible (40%)')

        # CORRECTION: Fixer l'échelle y entre 0 et 100
        ax.set_ylim(0, 100)

        # Annoter la valeur actuelle
        if not df['btc_dominance'].isna().all():
            current_dominance = df['btc_dominance'].iloc[-1]
            current_trend = "↑" if df['btc_dominance_trend'].iloc[-1] > 0 else "↓" if df['btc_dominance_trend'].iloc[
                                                                                          -1] < 0 else "→"

            ax.text(0.02, 0.95,
                    f"BTC Dominance: {current_dominance:.2f}% {current_trend}",
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_ylabel('Bitcoin Dominance (%)')
        ax.set_title(f'Bitcoin Dominance ({interval})')
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"btc_dominance_{interval}.png"))
        plt.close()

    def _plot_altcoin_season(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        # Définir les zones de couleur pour le fond
        ax.axhspan(0, 25, color='blue', alpha=0.1, label='Bitcoin Season (<25)')
        ax.axhspan(25, 75, color='gray', alpha=0.1, label='Neutral (25-75)')
        ax.axhspan(75, 100, color='green', alpha=0.1, label='Altcoin Season (>75)')

        # Vérifier si nous avons une série temporelle ou un point unique
        if len(df) <= 1:
            # Pour un seul point, utiliser un graphique en barre ou un scatter plot
            current_date = df['timestamp'].iloc[-1]
            altcoin_index = df['altcoin_season_index'].iloc[-1]

            # Couleur basée sur la valeur
            color = 'green' if altcoin_index > 75 else 'blue' if altcoin_index < 25 else 'gray'

            # Utiliser un scatter plot avec un point plus gros
            ax.scatter(current_date, altcoin_index, s=200, color=color,
                       label=f'Valeur actuelle: {altcoin_index:.1f}', zorder=5)

            # Ajouter un texte pour la valeur
            ax.text(current_date, altcoin_index + 5, f"{altcoin_index:.1f}",
                    ha='center', va='bottom', fontweight='bold')

            # Titre explicatif
            season = "Altcoin Season" if altcoin_index > 75 else "Bitcoin Season" if altcoin_index < 25 else "Neutral"
            ax.set_title(f'Altcoin Season Index - {season} ({altcoin_index:.1f})')
        else:
            # Pour une série temporelle, utiliser une ligne
            ax.plot(df['timestamp'], df['altcoin_season_index'], color='purple', marker='o',
                    label='Altcoin Season Index')
            ax.set_title(f'Altcoin Season Index ({interval})')

        # Lignes de référence
        ax.axhline(y=75, color='green', linestyle='--', alpha=0.7, label='Seuil Altcoin Season (75)')
        ax.axhline(y=25, color='blue', linestyle='--', alpha=0.7, label='Seuil Bitcoin Season (25)')

        # Annoter la valeur actuelle et la tendance
        if not df['altcoin_season_index'].isna().all():
            current_index = df['altcoin_season_index'].iloc[-1]
            current_trend = "↑" if df['altcoin_index_trend'].iloc[-1] > 0 else "↓" if df['altcoin_index_trend'].iloc[
                                                                                          -1] < 0 else "→"

            season_label = "Altcoin Season" if current_index > 75 else "Bitcoin Season" if current_index < 25 else "Neutral"

            ax.text(0.02, 0.95,
                    f"Altcoin Season Index: {current_index:.1f} - {season_label} {current_trend}",
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_ylabel('Altcoin Season Index')
        ax.legend(loc='upper right')
        ax.grid(True)

        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"altcoin_season_{interval}.png"))
        plt.close()

    def _plot_long_short_ratio(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))

        ax1 = plt.subplot(1, 1, 1)

        # Tracer le ratio long/short
        ax1.plot(df['timestamp'], df['long_short_ratio'], color='blue', label='Long/Short Ratio')

        # Ligne de référence à 1 (équilibre entre longs et shorts)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Équilibre (1.0)')

        # Zones de couleur pour les ratios extrêmes
        ax1.fill_between(df['timestamp'], 0, 0.75, color='red', alpha=0.1, label='Shorts Dominants (<0.75)')
        ax1.fill_between(df['timestamp'], 1.25, 5, color='green', alpha=0.1, label='Longs Dominants (>1.25)')

        # Ajouter un graphique secondaire pour les pourcentages
        if 'long_account' in df.columns and 'short_account' in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df['timestamp'], df['long_account'] * 100, color='green', linestyle='--', label='% Longs',
                     alpha=0.7)
            ax2.plot(df['timestamp'], df['short_account'] * 100, color='red', linestyle='--', label='% Shorts',
                     alpha=0.7)
            ax2.set_ylabel('Pourcentage (%)')

            # Combiner les légendes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')

        # Annoter la valeur actuelle et la tendance
        if not df['long_short_ratio'].isna().all():
            current_ratio = df['long_short_ratio'].iloc[-1]
            current_trend = "↑" if df['long_short_trend'].iloc[-1] > 0 else "↓" if df['long_short_trend'].iloc[
                                                                                       -1] < 0 else "→"

            ax1.text(0.02, 0.95,
                     f"Long/Short Ratio: {current_ratio:.2f} {current_trend}",
                     transform=ax1.transAxes, fontsize=12, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax1.set_ylabel('Long/Short Ratio')
        ax1.set_title(f'{crypto_id.upper()} - Long/Short Ratio ({interval})')
        ax1.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_long_short_{interval}.png"))
        plt.close()

    def _plot_put_call_ratio(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

        # Tracer le ratio put/call
        ax.plot(df['timestamp'], df['put_call_ratio'], color='purple', label='Put/Call Ratio')

        # Ligne de référence à 1 (équilibre entre puts et calls)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Équilibre (1.0)')

        # Zones pour les ratios extrêmes
        ax.fill_between(df['timestamp'], 0, 0.7, color='green', alpha=0.1, label='Optimisme (<0.7)')
        ax.fill_between(df['timestamp'], 1.3, 3, color='red', alpha=0.1, label='Pessimisme (>1.3)')

        # Annoter la valeur actuelle et la tendance
        if not df['put_call_ratio'].isna().all():
            current_ratio = df['put_call_ratio'].iloc[-1]
            current_trend = "↑" if df['put_call_trend'].iloc[-1] > 0 else "↓" if df['put_call_trend'].iloc[
                                                                                     -1] < 0 else "→"

            sentiment = "Pessimiste" if current_ratio > 1.3 else "Optimiste" if current_ratio < 0.7 else "Neutre"

            ax.text(0.02, 0.95,
                    f"Put/Call Ratio: {current_ratio:.2f} - {sentiment} {current_trend}",
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_ylabel('Put/Call Ratio')
        ax.set_title(f'{crypto_id.upper()} - Put/Call Ratio ({interval})')
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_put_call_{interval}.png"))
        plt.close()

    def _plot_open_interest(self, df, crypto_id, interval, crypto_dir):
        plt.figure(figsize=(14, 8))

        # Graphique principal pour l'OI total
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df['timestamp'], df['open_interest'], color='blue', label='Open Interest')

        # Annoter la valeur actuelle et la tendance
        if not df['open_interest'].isna().all():
            current_oi = df['open_interest'].iloc[-1]
            current_trend = "↑" if df['open_interest_trend'].iloc[-1] > 0 else "↓" if df['open_interest_trend'].iloc[
                                                                                          -1] < 0 else "→"

            ax1.text(0.02, 0.95,
                     f"Open Interest: {current_oi:,.0f} {current_trend}",
                     transform=ax1.transAxes, fontsize=12, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax1.set_ylabel('Open Interest')
        ax1.set_title(f'{crypto_id.upper()} - Open Interest ({interval})')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Graphique secondaire pour le % de changement
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.bar(df['timestamp'], df['open_interest_pct_change'],
                color=np.where(df['open_interest_pct_change'] >= 0, 'green', 'red'),
                label='% Changement')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Annoter le changement actuel
        if not df['open_interest_pct_change'].isna().all():
            current_change = df['open_interest_pct_change'].iloc[-1]
            ax2.text(0.02, 0.95,
                     f"Changement: {current_change:.2f}%",
                     transform=ax2.transAxes, fontsize=12, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax2.set_ylabel('Changement (%)')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_open_interest_{interval}.png"))
        plt.close()

    def _plot_position_amounts(self, df, crypto_id, interval, crypto_dir):
        """
        Visualise les montants des positions longues et courtes
        """
        plt.figure(figsize=(14, 10))

        # Premier graphique - Ratio achat/vente des takers
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df['timestamp'], df['buy_sell_ratio'], color='blue', label='Ratio Achat/Vente')
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Équilibre (1.0)')

        ax1.fill_between(df['timestamp'], 0, 0.8, color='red', alpha=0.1, label='Ventes dominantes (<0.8)')
        ax1.fill_between(df['timestamp'], 1.2, 2, color='green', alpha=0.1, label='Achats dominants (>1.2)')

        # Annoter la valeur actuelle
        if not df['buy_sell_ratio'].isna().all():
            current_ratio = df['buy_sell_ratio'].iloc[-1]
            current_trend = "↑" if df['volume_ratio_trend'].iloc[-1] > 0 else "↓" if df['volume_ratio_trend'].iloc[
                                                                                         -1] < 0 else "→"

            ax1.text(0.02, 0.95,
                     f"Ratio Achat/Vente: {current_ratio:.2f} {current_trend}",
                     transform=ax1.transAxes, fontsize=12, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax1.set_ylabel('Ratio Achat/Vente')
        ax1.set_title(f'{crypto_id.upper()} - Activité des traders ({interval})')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Deuxième graphique - Volumes d'achat et de vente
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.bar(df['timestamp'], df['buy_volume'], color='green', alpha=0.6, label='Volume Achat')
        ax2.bar(df['timestamp'], -df['sell_volume'], color='red', alpha=0.6, label='Volume Vente')

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Annoter les valeurs actuelles
        if not df['buy_volume'].isna().all():
            current_buy = df['buy_volume'].iloc[-1]
            current_sell = df['sell_volume'].iloc[-1]

            ax2.text(0.02, 0.95,
                     f"Achat: {current_buy:,.0f}, Vente: {current_sell:,.0f}",
                     transform=ax2.transAxes, fontsize=12, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax2.set_ylabel('Volume')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        # Troisième graphique - Valeur des positions
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)

        if all(col in df.columns for col in ['long_position_value', 'short_position_value']):
            ax3.plot(df['timestamp'], df['long_position_value'], color='green', label='Valeur Positions Longues')
            ax3.plot(df['timestamp'], df['short_position_value'], color='red', label='Valeur Positions Courtes')

            # Afficher le ratio sur un axe secondaire
            if 'position_value_ratio' in df.columns:
                ax3b = ax3.twinx()
                ax3b.plot(df['timestamp'], df['position_value_ratio'], color='blue', linestyle='--',
                          label='Ratio Long/Short (valeur)', alpha=0.7)
                ax3b.axhline(y=1, color='black', linestyle=':', alpha=0.3)
                ax3b.set_ylabel('Ratio Long/Short', color='blue')
                ax3b.tick_params(axis='y', colors='blue')

                # Combiner les légendes
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3b.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                # Annoter les valeurs actuelles
                if not df['position_value_ratio'].isna().all():
                    current_long = df['long_position_value'].iloc[-1]
                    current_short = df['short_position_value'].iloc[-1]
                    current_ratio = df['position_value_ratio'].iloc[-1]

                    ax3.text(0.02, 0.95,
                             f"Long: {current_long:,.0f}, Short: {current_short:,.0f}, Ratio: {current_ratio:.2f}",
                             transform=ax3.transAxes, fontsize=12, va='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax3.legend(loc='upper right')

        ax3.set_ylabel('Valeur des positions')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(crypto_dir, f"{crypto_id}_position_amounts_{interval}.png"))
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

    # Récupérer les données de marché globales
    analyzer.get_fear_greed_index(days=historical_days)
    analyzer.get_bitcoin_dominance(days=historical_days)

    # Essayer de récupérer l'Altcoin Season Index
    try:
        analyzer.get_altcoin_season_index()
    except:
        print("Impossible de récupérer l'Altcoin Season Index")

    # Récupérer les données BTC spécifiques (pour le marché global)
    try:
        analyzer.get_long_short_ratio('BTCUSDT')
        analyzer.get_position_amounts('BTCUSDT')
        analyzer.get_put_call_ratio('BTC')
        analyzer.get_open_interest('BTCUSDT')
    except:
        print("Impossible de récupérer certains indicateurs pour BTC")

    # Liste des cryptomonnaies à analyser
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
            # Récupérer les données spécifiques à cette crypto
            if symbol:
                analyzer.get_funding_rates(symbol)

                # Récupérer les indicateurs avancés pour cette crypto si ce n'est pas BTC
                if symbol.upper() != 'BTC' and symbol.upper() != 'BTCUSDT':
                    try:
                        symbol_futures = f"{symbol.upper()}USDT"
                        analyzer.get_long_short_ratio(symbol_futures)
                        analyzer.get_position_amounts(symbol_futures)
                        analyzer.get_open_interest(symbol_futures)
                        analyzer.get_put_call_ratio(symbol.upper())
                    except:
                        pass  # Ne pas interrompre si certains indicateurs ne sont pas disponibles

            historical_success = analyzer.collect_historical_data(crypto_id, symbol, interval, historical_days)

            if historical_success:
                data = analyzer.get_technical_indicators(crypto_id, interval)

                if data is not None and not data.empty:
                    data['name'] = name
                    data['symbol'] = symbol
                    all_data[crypto_id] = data
                    visualizer.visualize_data(data, crypto_id, interval)
                    visualizer.export_data_to_excel(data, crypto_id, interval)

                    # Affichage du résumé des indicateurs...
                    last_data = data.iloc[-1]

                    print(f"\n{name} ({symbol}):")
                    print(f"  Prix: ${last_data['current_price_usd']:.2f}")

                    if 'pct_from_ath' in last_data:
                        print(f"  ATH: {last_data['pct_from_ath']:.1f}% du maximum historique")

                    # Afficher les indicateurs techniques standard

                    # NOUVEAUX INDICATEURS
                    if 'btc_dominance' in last_data and not pd.isna(last_data['btc_dominance']):
                        dom = last_data['btc_dominance']
                        dom_trend = "↑" if last_data['btc_dominance_trend'] > 0 else "↓" if last_data[
                                                                                                'btc_dominance_trend'] < 0 else "→"
                        print(f"  BTC Dominance: {dom:.1f}% {dom_trend}")

                    if 'altcoin_season_index' in last_data and not pd.isna(last_data['altcoin_season_index']):
                        asi = last_data['altcoin_season_index']
                        asi_trend = "↑" if last_data['altcoin_index_trend'] > 0 else "↓" if last_data[
                                                                                                'altcoin_index_trend'] < 0 else "→"
                        season = "Altcoin Season" if asi > 75 else "Bitcoin Season" if asi < 25 else "Neutral"
                        print(f"  Altcoin Season: {asi:.0f} - {season} {asi_trend}")

                    if 'long_short_ratio' in last_data and not pd.isna(last_data['long_short_ratio']):
                        lsr = last_data['long_short_ratio']
                        lsr_trend = "↑" if last_data['long_short_trend'] > 0 else "↓" if last_data[
                                                                                             'long_short_trend'] < 0 else "→"
                        print(f"  Long/Short Ratio: {lsr:.2f} {lsr_trend}")

                    if 'put_call_ratio' in last_data and not pd.isna(last_data['put_call_ratio']):
                        pcr = last_data['put_call_ratio']
                        pcr_trend = "↑" if last_data['put_call_trend'] > 0 else "↓" if last_data[
                                                                                           'put_call_trend'] < 0 else "→"
                        sentiment = "Pessimiste" if pcr > 1.3 else "Optimiste" if pcr < 0.7 else "Neutre"
                        print(f"  Put/Call Ratio: {pcr:.2f} - {sentiment} {pcr_trend}")

                    if 'open_interest' in last_data and not pd.isna(last_data['open_interest']):
                        oi = last_data['open_interest']
                        oi_trend = "↑" if last_data['open_interest_trend'] > 0 else "↓" if last_data[
                                                                                               'open_interest_trend'] < 0 else "→"
                        oi_change = last_data.get('open_interest_pct_change', 0)
                        print(f"  Open Interest: {oi:,.0f} {oi_trend} ({oi_change:.1f}%)")

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