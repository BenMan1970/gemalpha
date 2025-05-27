import streamlit as st
import pandas as pd
import numpy as np
import talib
import requests
import time
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="Scanner Confluence Forex Premium", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium")
st.markdown("*Filtrage automatique 5-6 étoiles sur les paires forex et XAU/USD (via Alpha Vantage)*")

# Liste des paires supportées par Alpha Vantage
FOREX_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP", "AUD/JPY", "EUR/AUD", "EUR/CHF", "AUD/NZD",
    "NZD/JPY", "GBP/AUD", "GBP/CAD", "EUR/NZD", "AUD/CAD", "GBP/CHF", "CAD/CHF",
    "EUR/CAD", "AUD/CHF", "NZD/CAD", "NZD/CHF", "XAU/USD"
]

# Paramètres par défaut
DEFAULT_PARAMS = {
    'hma_length': 20,
    'adx_threshold': 20,
    'rsi_length': 10,
    'adx_length': 14,
    'ichimoku_length': 9,
    'smoothed_ha_len1': 10,
    'smoothed_ha_len2': 10
}

@st.cache_data(ttl=3600)
def get_forex_data_alpha_vantage(symbol: str, interval: str = "60min", apikey: str = "") -> pd.DataFrame:
    base_url = "https://www.alphavantage.co/query"
    from_symbol, to_symbol = symbol.split("/")
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "apikey": apikey,
        "outputsize": "compact"
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    if f"Time Series FX ({interval})" not in data:
        return None
    df = pd.DataFrame(data[f"Time Series FX ({interval})"]).T
    df.columns = ["Open", "High", "Low", "Close"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def calculate_confluence_signals(df, params):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    hma = talib.WMA(close, timeperiod=params['hma_length'])
    rsi = talib.RSI(close, timeperiod=params['rsi_length'])
    adx = talib.ADX(high, low, close, timeperiod=params['adx_length'])

    kijun_sen = (df['High'].rolling(params['ichimoku_length']).max() + df['Low'].rolling(params['ichimoku_length']).min()) / 2

    signal_ha = 1 if close[-1] > hma[-1] else -1
    signal_rsi = 1 if rsi[-1] > 50 else -1
    signal_adx = 1 if adx[-1] > params['adx_threshold'] else 0
    signal_ichimoku = 1 if close[-1] > kijun_sen.iloc[-1] else -1

    total = signal_ha + signal_rsi + signal_adx + signal_ichimoku
    stars = 3 + signal_adx

    direction = "BUY" if total >= 3 else ("SELL" if total <= -3 else "WAIT")
    return {
        "direction": direction,
        "confluence": stars,
        "signals": {
            "HeikenAshi": signal_ha,
            "RSI": signal_rsi,
            "ADX": signal_adx,
            "Ichimoku": signal_ichimoku
        }
    }

def main():
    st.sidebar.header("⚙️ Paramètres")
    api_key = st.sidebar.text_input("🔑 Clé API Alpha Vantage", type="password", value="demo")
    selected_pairs = st.sidebar.multiselect("📈 Paires à scanner", FOREX_PAIRS, default=["EUR/USD", "GBP/USD", "XAU/USD"])

    st.subheader("📊 Résultats")
    with st.spinner("Chargement des données..."):
        results = []
        for i, pair in enumerate(selected_pairs):
            if i != 0 and i % 5 == 0:
                time.sleep(60)  # Respecter la limite gratuite Alpha Vantage

            df = get_forex_data_alpha_vantage(pair, apikey=api_key)
            if df is not None:
                signals = calculate_confluence_signals(df, DEFAULT_PARAMS)
                if signals:
                    results.append({
                        "Pair": pair,
                        "Direction": signals["direction"],
                        "Confluence": f"{signals['confluence']} ⭐",
                        "Signaux": signals["signals"]
                    })

        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)
        else:
            st.info("Aucune donnée ou signal détecté.")

if __name__ == "__main__":
    main()

