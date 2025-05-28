# 장기투자
# ===============================================================
#  KOSPI 과열 지수 계산기 (0~100) – Relative Extremes & Signed Weights
#  최초 : 2025-05-27  /  개정 : 2025-05-28
# ===============================================================

# 1) 설치 ------------------------------------------------------------
!pip install yfinance pandas_datareader beautifulsoup4 plotly ta scipy pykrx --quiet

# 2) 라이브러리 -------------------------------------------------------
import pandas as pd, numpy as np, datetime, os, pickle, warnings
import yfinance as yf
from ta.momentum   import RSIIndicator
from ta.trend      import MACD
from ta.volatility import BollingerBands
from scipy.stats   import percentileofscore
from pykrx        import stock
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 3) 기간 & 파라미터 --------------------------------------------------
START_DATE, END_DATE = "2010-01-01", datetime.date.today().strftime("%Y-%m-%d")
print(f"[기간] {START_DATE} ~ {END_DATE}")
PD_END = pd.to_datetime(END_DATE)
PYKRX_START, PYKRX_END = PD_END.strftime("%Y%m%d"), PD_END.strftime("%Y%m%d")
ROLL_W, CORR_W, STEP_D = 504, 1008, 40

# 4) 캐싱 ------------------------------------------------------------
def _save(obj, path):  pickle.dump(obj, open(path,"wb"))
def _load(path):       return pickle.load(open(path,"rb")) if os.path.exists(path) else None
def _valid(df):        return isinstance(df, pd.DataFrame) and not df.empty

# 5) 데이터 수집 ------------------------------------------------------
TICKERS = {"KOSPI":"^KS11","SP500":"^GSPC","VIX":"^VIX",
           "USDKRW":"KRW=X","US10Y":"^TNX","US2Y":"^FVX"}

def fetch_price():
    c=_load("price.pkl")
    if _valid(c) and c.index.max() >= PD_END:
        return c[list(TICKERS.keys())]
    d = yf.download(list(TICKERS.values()), start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    if isinstance(d, pd.Series): d=d.to_frame()
    colmap={v:k for k,v in TICKERS.items()}
    d.columns=[colmap.get(col if isinstance(col,str) else col[0],col) for col in d.columns]
    d=d[list(TICKERS.keys())].ffill().bfill()
    _save(d,"price.pkl"); return d

def fetch_per_pbr():
    c=_load("perpbr.pkl")
    if _valid(c): return c
    df=stock.get_index_fundamental(PYKRX_START,PYKRX_END,"1001")
    df=df.rename_axis("Date").reset_index()
    df["Date"]=pd.to_datetime(df["Date"])
    df=df[["Date","PER","PBR"]].set_index("Date").replace(0,np.nan).ffill()
    _save(df,"perpbr.pkl"); return df

def fetch_flow():
    c=_load("flow.pkl")
    if _valid(c): return c
    df=stock.get_market_trading_value_by_date(PYKRX_START,PYKRX_END,"KOSPI")
    df=df.rename_axis("Date").reset_index(); df["Date"]=pd.to_datetime(df["Date"])
    f_col=next(col for col in df.columns if "외국인" in col)
    i_col=next(col for col in df.columns if "기관"  in col)
    df=df[["Date",f_col,i_col]].rename(columns={f_col:"Foreign",i_col:"Institution"}).set_index("Date").ffill()
    _save(df,"flow.pkl"); return df

price, perpbr, flow = fetch_price(), fetch_per_pbr(), fetch_flow()
kospi = price["KOSPI"].dropna()

# 6) 지표 계산 --------------------------------------------------------
rsi       = RSIIndicator(kospi,14).rsi()
macd_diff = MACD(kospi).macd_diff()
bb_pct    = ((kospi - BollingerBands(kospi).bollinger_lband()) /
            (BollingerBands(kospi).bollinger_hband() - BollingerBands(kospi).bollinger_lband()))*100
mom20     = kospi.pct_change(20)*100
ma50_pct  = (kospi/kospi.rolling(50).mean()-1)*100
ret20     = kospi.pct_change(20)*100
sp_ret20  = price["SP500"].pct_change(20)*100
vol20     = kospi.pct_change().rolling(20).std()*np.sqrt(252)*100
sp_vol20  = price["SP500"].pct_change().rolling(20).std()*np.sqrt(252)*100
vol_ratio = vol20 / sp_vol20.replace(0,np.nan)
macro     = pd.DataFrame({
    "USDKRW_20d_change": price["USDKRW"].pct_change(20)*100,
    "Yield_Spread_10Y_2Y": price["US10Y"] - price["US2Y"]
}, index=kospi.index)

panel = pd.DataFrame({
    "RSI":rsi, "MACD_diff":macd_diff, "BB_Position":bb_pct, "Momentum_20d":mom20,
    "Price_to_MA50":ma50_pct, "PER":perpbr["PER"], "PBR":perpbr["PBR"],
    "VIX":price["VIX"], "KOSPI_vol_20d":vol20, "KOSPI_SP_vol_ratio":vol_ratio,
    "KOSPI_20d_ret":ret20, "KOSPI_SP_ret_diff":ret20-sp_ret20,
    "USDKRW_20d_change":macro["USDKRW_20d_change"],
    "Yield_Spread_10Y_2Y":macro["Yield_Spread_10Y_2Y"],
    "Foreign_NetBuy_20d_sum":flow["Foreign"].rolling(20).sum(),
    "Institution_NetBuy_20d_sum":flow["Institution"].rolling(20).sum()
})

# 7) 0-100 스케일 함수 ---------------------------------------------------
def roll_pct(s,w,inv=False):
    arr=s.to_numpy(); out=np.full_like(arr,np.nan,dtype=float)
    for i in range(len(arr)):
        if i<w-1 or np.isnan(arr[i]): continue
        win=arr[i-w+1:i+1][~np.isnan(arr[i-w+1:i+1])]
        if len(win):
            out[i] = (100-percentileofscore(win,arr[i])) if inv else percentileofscore(win,arr[i])
    return pd.Series(out,index=s.index)

hi_list = ["RSI","BB_Position","Momentum_20d","Price_to_MA50","PER","PBR","VIX",
           "KOSPI_vol_20d","KOSPI_SP_vol_ratio","KOSPI_20d_ret","KOSPI_SP_ret_diff","USDKRW_20d_change"]
lo_list = ["Yield_Spread_10Y_2Y","Foreign_NetBuy_20d_sum","Institution_NetBuy_20d_sum"]

Score = pd.DataFrame(index=panel.index)
for c in hi_list: Score[c] = roll_pct(panel[c], ROLL_W)
for c in lo_list: Score[c] = roll_pct(panel[c], ROLL_W, inv=True)

# MACD 점수
macd_abs_max = macd_diff.abs().rolling(ROLL_W).max()
Score["MACD_Score"] = ((macd_diff/macd_abs_max).fillna(0)*50+50).clip(0,100)
Score = Score.ffill()

# 8) 시장 위치 백분위 ---------------------------------------------------
mkt_pct = roll_pct(kospi, CORR_W).clip(0,100)

# 9) 동적 부호 포함 가중치 계산 --------------------------------------------
weights = pd.DataFrame(index=Score.index, columns=Score.columns, dtype=float)

def safe_corr(a,b):
    df = pd.concat([a,b], axis=1).dropna()
    if len(df)<30 or df.nunique().min()<2: return 0
    return df.corr().iloc[0,1]

for idx in range(CORR_W, len(Score), STEP_D):
    sub, tgt = Score.iloc[idx-CORR_W:idx], mkt_pct.iloc[idx-CORR_W:idx]
    corrs = sub.apply(lambda col: safe_corr(col, tgt))
    if corrs.abs().sum()==0: corrs[:] = 1/len(corrs)
    else: corrs = corrs / corrs.abs().sum()
    weights.loc[Score.index[idx]] = corrs

weights = weights.ffill().bfill()

# 9-1) ▶ 가중치 설정 결과 확인 ----------------------------------------
# 가중치 변화 시각화
fig_w = px.line(weights, title="Indicator Weights Over Time")
fig_w.update_layout(xaxis_title="Date", yaxis_title="Weight")
fig_w.show()

# 10) RAW & 최종 지수 계산 --------------------------------------------
raw_idx   = (Score * weights).sum(axis=1)
risk_idx  = roll_pct(raw_idx, CORR_W).clip(0,100)

result    = Score.copy()
result["RAW_IDX"]        = raw_idx
result["KOSPI_RISK_IDX"] = risk_idx

# 11) 시각화 ----------------------------------------------------------
fig = go.Figure([
    go.Scatter(x=result.index, y=result["KOSPI_RISK_IDX"], name="Risk Index", line=dict(color="red")),
    go.Scatter(x=kospi.index,      y=kospi,                name="KOSPI",    yaxis="y2", line=dict(color="blue", dash="dot"))
])
fig.add_hline(y=80,   line_dash="dash", line_color="rgba(255,0,0,0.6)")
fig.add_hline(y=20,   line_dash="dash", line_color="rgba(0,128,0,0.6)")
fig.update_layout(
    title="KOSPI Risk Index (0-100) vs KOSPI",
    yaxis = dict(title="Risk (0-100)", range=[0,100]),
    yaxis2= dict(title="KOSPI", overlaying="y", side="right"),
    legend= dict(orientation="h", y=1.02)
)
fig.show()

# 12) 최신 요약 -------------------------------------------------------
today  = result.index[-1].strftime("%Y-%m-%d")
latest = result["KOSPI_RISK_IDX"].iloc[-1]
lvl    = "매우 높음" if latest>=80 else "높음" if latest>=60 else "중립" if latest>=40 else "낮음" if latest>=20 else "매우 낮음"
print(f"\n[요약 {today}] KOSPI_RISK_IDX = {latest:.1f}/100 → {lvl}")

# 13) 저장 -----------------------------------------------------------
result.to_csv("kospi_risk_index_dynamic.csv")
print("✔ CSV 저장 완료")
