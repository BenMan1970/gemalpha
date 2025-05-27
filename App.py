import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback 

from alpha_vantage.foreignexchange import ForeignExchange

st.set_page_config(page_title="Scanner Confluence Forex (AlphaVantage)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Alpha Vantage)")
st.markdown("*Utilisation de l'API Alpha Vantage pour les données de marché*")

AV_API_KEY = None
fx = None 
try:
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except KeyError:
    st.error("Erreur: Secret 'ALPHA_VANTAGE_API_KEY' non défini. Configurez vos secrets.")
    st.stop()

if AV_API_KEY:
    try:
        fx = ForeignExchange(key=AV_API_KEY, output_format='pandas')
        st.sidebar.success("Client Alpha Vantage initialisé.")
    except Exception as e:
        st.error(f"Erreur initialisation client Alpha Vantage: {e}")
        st.sidebar.error("Échec initialisation Alpha Vantage.")
else:
    st.error("Clé API Alpha Vantage non disponible.")

FOREX_PAIRS_AV = [
    ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), ('USD', 'CHF'),
    ('AUD', 'USD'), ('USD', 'CAD'), ('NZD', 'USD'), ('EUR', 'JPY'),
    ('GBP', 'JPY'), ('EUR', 'GBP')
]

# --- Mapping pour les Timeframes Alpha Vantage ---
# Utilisé pour convertir une chaîne (hashable) en objet TimeFrame (potentiellement non hashable comme arg direct de @st.cache_data)
TIMEFRAME_MAP_ALPACA = { # Renommé pour clarté, même si utilisé pour AV ici
    "1Min": "1min", "5Min": "5min", "15Min": "15min", 
    "30Min": "30min", "60min": "60min" 
    # Alpha Vantage utilise des strings pour les intervalles intraday
    # Si vous utilisez fx.get_currency_exchange_daily, etc., pas besoin d'intervalle
}


def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()
def hull_ma_pine(dc, p=20):
    hl=int(p/2); sl=int(np.sqrt(p))
    wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    diff=2*wma1-wma2; return diff.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
def rsi_pine(po4,p=10): d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0);ag=rma(g,p);al=rma(l,p);rs=ag/al.replace(0,1e-9);rsi=100-(100/(1+rs));return rsi.fillna(50)
def adx_pine(h,l,c,p=14):
    tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1));tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
    um=h.diff();dm=l.shift(1)-l
    pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index);mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
    satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
    dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden);return rma(dx,p).fillna(0)
def heiken_ashi_pine(dfo):
    ha=pd.DataFrame(index=dfo.index)
    if dfo.empty:ha['HA_Open']=pd.Series(dtype=float);ha['HA_Close']=pd.Series(dtype=float);return ha['HA_Open'],ha['HA_Close']
    ha['HA_Close']=(dfo['Open']+dfo['High']+dfo['Low']+dfo['Close'])/4;ha['HA_Open']=np.nan
    if not dfo.empty:
        ha.iloc[0,ha.columns.get_loc('HA_Open')]=(dfo['Open'].iloc[0]+dfo['Close'].iloc[0])/2
        for i in range(1,len(dfo)):ha.iloc[i,ha.columns.get_loc('HA_Open')]=(ha.iloc[i-1,ha.columns.get_loc('HA_Open')]+ha.iloc[i-1,ha.columns.get_loc('HA_Close')])/2
    return ha['HA_Open'],ha['HA_Close']
def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10):
    eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
    hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index)
    hao_i,hac_i=heiken_ashi_pine(hai);sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc
def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high)<min_len_req or len(df_low)<min_len_req or len(df_close)<min_len_req:print(f"Ichi:Data<({len(df_close)}) vs req {min_len_req}.");return 0
    ts=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2;ks=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
    sa=(ts+ks)/2;sb=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2
    if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):print("Ichi:NaN close/spans.");return 0
    ccl=df_close.iloc[-1];cssa=sa.iloc[-1];cssb=sb.iloc[-1];ctn=max(cssa,cssb);cbn=min(cssa,cssb);sig=0
    if ccl>ctn:sig=1
    elif ccl<cbn:sig=-1
    return sig

@st.cache_data(ttl=3600)
def get_data_av(from_currency: str, to_currency: str, av_interval_str: str = '60min', output_size_av: str = 'compact'):
    global fx 
    if fx is None: st.error("FATAL: Client Alpha Vantage non initialisé."); print("FATAL: Client AV non initialisé."); return None
    pair_str = f"{from_currency}/{to_currency}"
    print(f"\n--- Début get_data_av: from={from_currency}, to={to_currency}, interval={av_interval_str}, size={output_size_av} ---")
    try:
        # Alpha Vantage utilise directement la chaîne pour l'intervalle intraday
        data_df, meta_data = fx.get_currency_exchange_intraday(from_symbol=from_currency, to_symbol=to_currency, interval=av_interval_str, outputsize=output_size_av)
        print(f"Données brutes AV reçues pour {pair_str}. Meta: {meta_data}")
        data_df.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close'}, inplace=True)
        if data_df.index.tz is not None: data_df.index = data_df.index.tz_convert('UTC'); print(f"Index {pair_str} converti UTC.")
        else: data_df.index = data_df.index.tz_localize('UTC'); print(f"Index {pair_str} localisé UTC (supposition).")
        data_df = data_df.iloc[::-1] 
        if data_df.empty or len(data_df) < 55: st.warning(f"Données AV<55 {pair_str} ({len(data_df)})."); print(f"Données AV<55 {pair_str} ({len(data_df)})."); return None
        cols_num = ['Open','High','Low','Close']; 
        for col in cols_num:
            if col in data_df.columns: data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        pdf = data_df.dropna(subset=cols_num)
        print(f"Données {pair_str} OK. Retour {len(pdf)}l.\n--- Fin get_data_av {pair_str} ---\n"); return pdf
    except ValueError as ve:
        st.error(f"Erreur AV (ValueError) {pair_str}: {ve}"); print(f"ERREUR AV (ValueError) {pair_str}:\n{str(ve)}")
        if "Invalid API call" in str(ve) or "API key" in str(ve): st.error("Problème clé API AV ou format appel.")
        elif "call frequency" in str(ve): st.warning(f"Limite taux AV atteinte {pair_str}. Réessayez.")
        return None
    except Exception as e: st.error(f"Erreur inattendue get_data_av {pair_str}: {type(e).__name__}"); st.exception(e); print(f"ERREUR INATTENDUE get_data_av {pair_str}:\n{traceback.format_exc()}"); return None

# --- Fonction calculate_all_signals_pine (CORRIGÉE pour indentation et syntaxe) ---
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60:
        print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes).")
        return None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"calculate_all_signals: Colonnes OHLC manquantes.")
        return None
    
    close = data['Close']; high = data['High']; low = data['Low']; open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    bull_confluences, bear_confluences, signal_details_pine = 0, 0, {}

    try: # 1. HMA
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]; hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A"
    except Exception as e: signal_details_pine['HMA'] = "ErrHMA"; print(f"Erreur HMA: {e}")
    try: # 2. RSI
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >=1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]; signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"
    except Exception as e: signal_details_pine['RSI'] = "ErrRSI"; signal_details_pine['RSI_val'] = "N/A"; print(f"Erreur RSI: {e}")
    try: # 3. ADX
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]; signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"
    except Exception as e: signal_details_pine['ADX'] = "ErrADX"; signal_details_pine['ADX_val'] = "N/A"; print(f"Erreur ADX: {e}")
    try: # 4. Heiken Ashi
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >=1 and len(ha_close) >=1 and not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = "ErrHA"; print(f"Erreur HA: {e}")
    try: # 5. Smoothed Heiken Ashi
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >=1 and len(sha_close) >=1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = "ErrSHA"; print(f"Erreur SHA: {e}")
    try: # 6. Ichimoku
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        elif ichimoku_signal_val == 0 and (len(data) < max(9,26,52) or (len(data) > 0 and pd.isna(data['Close'].iloc[-1]))): signal_details_pine['Ichi'] = "N/D"
        else: signal_details_pine['Ichi'] = "─"
    except Exception as e: signal_details_pine['Ichi'] = "ErrIchi"; print(f"Erreur Ichi: {e}")
    confluence_value=max(bull_confluences,bear_confluences)
    direction="NEUTRE"
    if bull_confluences > bear_confluences: direction="HAUSSIER"
    elif bear_confluences > bull_confluences: direction="BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0: direction="CONFLIT"
    return{'confluence_P':confluence_value,'direction_P':direction,'bull_P':bull_confluences,'bear_P':bear_confluences,
            'rsi_P':signal_details_pine.get('RSI_val',"N/A"),'adx_P':signal_details_pine.get('ADX_val',"N/A"),
            'signals_P':signal_details_pine}

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

col1,col2=st.columns([1,3])
with col1:
    st.subheader("⚙️ Paramètres");min_conf=st.selectbox("Confluence min (0-6)",options=[0,1,2,3,4,5,6],index=3,format_func=lambda x:f"{x} (confluence)")
    show_all=st.checkbox("Voir toutes les paires (ignorer filtre)");scan_dis_av = fx is None;scan_tip_av="Client AV non initialisé." if scan_dis_av else "Lancer scan (AV)"
    scan_btn=st.button("🔍 Scanner (Données Alpha Vantage H1)",type="primary",use_container_width=True,disabled=scan_dis_av,help=scan_tip_av)
with col2:
    if scan_btn:
        st.info(f"🔄 Scan en cours (Alpha Vantage H1)...");pr_res=[];pb=st.progress(0);stx=st.empty()
        for i,(from_s,to_s) in enumerate(FOREX_PAIRS_AV):
            pnd=f"{from_s}{to_s}";cp=(i+1)/len(FOREX_PAIRS_AV);pb.progress(cp);stx.text(f"Analyse (AV H1):{pnd}({i+1}/{len(FOREX_PAIRS_AV)})")
            d_h1_av=get_data_av(from_s,to_s,av_interval_str="60min",output_size_av='compact') # Utiliser av_interval_str
            if d_h1_av is not None:
                sigs=calculate_all_signals_pine(d_h1_av)
                if sigs:strs=get_stars_pine(sigs['confluence_P']);rd={'Paire':pnd,'Direction':sigs['direction_P'],'Conf. (0-6)':sigs['confluence_P'],'Étoiles':strs,'RSI':sigs['rsi_P'],'ADX':sigs['adx_P'],'Bull':sigs['bull_P'],'Bear':sigs['bear_P'],'details':sigs['signals_P']};pr_res.append(rd)
                else:pr_res.append({'Paire':pnd,'Direction':'ERREUR CALCUL','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Calcul signaux (AV) échoué'}})
            else:pr_res.append({'Paire':pnd,'Direction':'ERREUR DONNÉES AV','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Données AV non dispo/symb invalide(logs serveur)'}})
            print(f"Pause de 13 secondes pour limite de taux AV...");time.sleep(13)
        pb.empty();stx.empty()
        if pr_res:
            dfa=pd.DataFrame(pr_res);dfd=dfa[dfa['Conf. (0-6)']>=min_conf].copy()if not show_all else dfa.copy()
            if not show_all:st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (Alpha Vantage).")
            else:st.info(f"🔍 Affichage des {len(dfd)} paires (Alpha Vantage).")
            if not dfd.empty:
                dfds=dfd.sort_values('Conf. (0-6)',ascending=False);vcs=[c for c in['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear']if c in dfds.columns]
                st.dataframe(dfds[vcs],use_container_width=True,hide_index=True)
                with st.expander("📊 Détails des signaux (Alpha Vantage)"):
                    for _,r in dfds.iterrows():
                        sm=r.get('details',{});
                        if not isinstance(sm,dict):sm={'Info':'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc=st.columns(6);so=['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx,sk in enumerate(so):dc[idx].metric(label=sk,value=sm.get(sk,"N/P"))
                        st.divider()
            else:st.warning(f"❌ Aucune paire avec critères filtrage (Alpha Vantage). Vérifiez erreurs données/symbole.")
        else:st.error("❌ Aucune paire traitée (Alpha Vantage). Vérifiez logs serveur.")
with st.expander("ℹ️ Comment ça marche (Logique Pine Script avec Données Alpha Vantage)"):
    st.markdown("""**6 Signaux Confluence:** HMA(20),RSI(10),ADX(14)>=20,HA(Simple),SHA(10,10),Ichi(9,26,52).**Comptage & Étoiles:**Pine.**Source:**Alpha Vantage API.""")
st.caption("Scanner H1 (Alpha Vantage). Multi-TF non actif. Attention aux limites de taux de l'API Alpha Vantage.")
