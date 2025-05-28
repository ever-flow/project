# 필요한 라이브러리 가져오기
import pandas as pd, numpy as np, datetime, os, pickle, warnings, math
import yfinance as yf
from ta.momentum   import RSIIndicator
from ta.trend      import MACD
from ta.volatility import BollingerBands
from scipy.stats   import percentileofscore
from pykrx        import stock
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 글로벌 파라미터 설정
START_DATE, END_DATE = "1995-01-01", datetime.date.today().strftime("%Y-%m-%d")
ROLL_W   = 63                       # 랭킹용 롤링 윈도우 (3개월)
CORR_W   = 126                      # 상관계수 학습 구간 (반년)
STEP_D   = 40                        # 가중치 재계산 주기 (≈2개월)
HALF_LF  = 15                       # EWM 반감기 (15일)

print(f"[기간] {START_DATE} ~ {END_DATE}")

PD_END      = pd.to_datetime(END_DATE)
PYKRX_START = pd.to_datetime(START_DATE).strftime("%Y%m%d")
PYKRX_END   = PD_END.strftime("%Y%m%d")

# 캐싱 유틸리티 함수
def _save(obj, path):
    # 객체를 파일에 저장
    pickle.dump(obj, open(path, "wb"))
def _load(path):
    # 파일에서 객체 로드, 실패 시 None 반환
    try: return pickle.load(open(path, "rb"))
    except: return None
def _valid(df):
    # DataFrame 유효성 검증
    return isinstance(df, pd.DataFrame) and not df.empty

# 데이터 수집
TICKERS = {
    "KOSPI":"^KS11", "SP500":"^GSPC", "VIX":"^VIX",
    "USDKRW":"KRW=X", "US10Y":"^TNX", "US2Y":"^FVX"
}

def fetch_price():
    # 가격 데이터 가져오기 (캐시 활용)
    cache = _load("price.pkl")
    if _valid(cache) and cache.index.max() >= PD_END:
        return cache[list(TICKERS.keys())]
    df = yf.download(list(TICKERS.values()), start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.columns = [{v:k for k,v in TICKERS.items()}.get(col if isinstance(col,str) else col[0], col) for col in df.columns]
    df = df[list(TICKERS.keys())].ffill().bfill()
    _save(df, "price.pkl")
    return df

def fetch_per_pbr():
    # PER/PBR 데이터 가져오기 (캐시 활용)
    cache = _load("perpbr.pkl")
    if _valid(cache):
        return cache
    df = stock.get_index_fundamental(PYKRX_START, PYKRX_END, "1001")
    df = df.rename_axis("Date").reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date","PER","PBR"]].set_index("Date").replace(0, np.nan).ffill()
    _save(df, "perpbr.pkl")
    return df

def fetch_flow():
    # 매매 동향 데이터 가져오기 (캐시 활용)
    cache = _load("flow.pkl")
    if _valid(cache):
        return cache
    df = stock.get_market_trading_value_by_date(PYKRX_START, PYKRX_END, "KOSPI")
    df = df.rename_axis("Date").reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    f_col = next(col for col in df.columns if "외국인" in col)
    i_col = next(col for col in df.columns if "기관"   in col)
    p_cols = [col for col in df.columns if "개인" in col]
    p_col  = p_cols[0] if p_cols else None
    rename_map = {f_col:"Foreign", i_col:"Institution"}
    cols = ["Date", f_col, i_col]
    if p_col:
        rename_map[p_col] = "Individual"
        cols.append(p_col)
    df = df[cols].rename(columns=rename_map).set_index("Date").ffill()
    _save(df, "flow.pkl")
    return df

# 데이터 가져오기
price, perpbr, flow = fetch_price(), fetch_per_pbr(), fetch_flow()
kospi = price["KOSPI"].dropna()

# 지표 계산
rsi       = RSIIndicator(kospi, 14).rsi()
macd_diff = MACD(kospi).macd_diff()
bb_pct    = ((kospi - BollingerBands(kospi).bollinger_lband()) / 
             (BollingerBands(kospi).bollinger_hband() - BollingerBands(kospi).bollinger_lband())) * 100
mom20     = kospi.pct_change(20) * 100
ma50_pct  = (kospi / kospi.rolling(50).mean() - 1) * 100
ret20     = kospi.pct_change(20) * 100
sp_ret20  = price["SP500"].pct_change(20) * 100
vol20     = kospi.pct_change().rolling(20).std() * math.sqrt(252) * 100
sp_vol20  = price["SP500"].pct_change().rolling(20).std() * math.sqrt(252) * 100
vol_ratio = vol20 / sp_vol20.replace(0, np.nan)
macro     = pd.DataFrame({
    "USDKRW_20d_change" : price["USDKRW"].pct_change(20)*100,
    "Yield_Spread_10Y_2Y": price["US10Y"] - price["US2Y"]
}, index=kospi.index)

panel = {
    "RSI"                   : rsi,
    "MACD_diff"             : macd_diff,
    "BB_Position"           : bb_pct,
    "Momentum_20d"          : mom20,
    "Price_to_MA50"         : ma50_pct,
    "PER"                   : perpbr["PER"],
    "PBR"                   : perpbr["PBR"],
    "VIX"                   : price["VIX"],
    "KOSPI_vol_20d"         : vol20,
    "KOSPI_SP_vol_ratio"    : vol_ratio,
    "KOSPI_20d_ret"         : ret20,
    "KOSPI_SP_ret_diff"     : ret20 - sp_ret20,
    "USDKRW_20d_change"     : macro["USDKRW_20d_change"],
    "Yield_Spread_10Y_2Y"   : macro["Yield_Spread_10Y_2Y"],
    "Foreign_NetBuy_20d"    : flow["Foreign"].rolling(20).sum(),
    "Institution_NetBuy_20d": flow["Institution"].rolling(20).sum()
}
if "Individual" in flow.columns:
    panel["Individual_NetBuy_20d"] = flow["Individual"].rolling(20).sum()
panel = pd.DataFrame(panel)

# 백분위 점수 계산
def roll_pct(s, w, inv=False):
    # 롤링 윈도우 기반 백분위 계산
    arr = s.to_numpy()
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        if i < w-1 or np.isnan(arr[i]):
            continue
        win = arr[i-w+1:i+1][~np.isnan(arr[i-w+1:i+1])]
        if win.size:
            pct = percentileofscore(win, arr[i])
            out[i] = 100 - pct if inv else pct
    return pd.Series(out, index=s.index)

hi = ["RSI","BB_Position","Momentum_20d","Price_to_MA50","PER","PBR","VIX",
      "KOSPI_vol_20d","KOSPI_SP_vol_ratio","KOSPI_20d_ret","KOSPI_SP_ret_diff","USDKRW_20d_change"]
lo = ["Yield_Spread_10Y_2Y","Foreign_NetBuy_20d","Institution_NetBuy_20d"] + \
     (["Individual_NetBuy_20d"] if "Individual_NetBuy_20d" in panel.columns else [])

Score = pd.DataFrame(index=panel.index)
for c in hi:
    Score[c] = roll_pct(panel[c], ROLL_W)
for c in lo:
    Score[c] = roll_pct(panel[c], ROLL_W, inv=True)

# MACD 스케일링
macd_abs_max = macd_diff.abs().rolling(ROLL_W).max()
Score["MACD_Score"] = ((macd_diff/macd_abs_max).fillna(0)*50 + 50).clip(0,100)
Score = Score.ffill()

# 시장 위치 백분위
mkt_pct = roll_pct(kospi, CORR_W).clip(0,100)

# EWM 상관계수 기반 가중치 계산
def ewm_corr(a, b, span):
    # 지수 이동 평균 상관계수 계산
    a_mean = a.ewm(span=span, adjust=False).mean()
    b_mean = b.ewm(span=span, adjust=False).mean()
    cov    = ((a - a_mean)*(b - b_mean)).ewm(span=span, adjust=False).mean()
    std_a  = ((a - a_mean)**2).ewm(span=span, adjust=False).mean().pow(0.5)
    std_b  = ((b - b_mean)**2).ewm(span=span, adjust=False).mean().pow(0.5)
    corr   = cov / (std_a * std_b)
    return corr.iloc[-1] if np.isfinite(corr.iloc[-1]) else 0

weights = pd.DataFrame(index=Score.index, columns=Score.columns, dtype=float)
for idx in range(CORR_W, len(Score), STEP_D):
    sub   = Score.iloc[idx-CORR_W: idx]
    tgt   = mkt_pct.iloc[idx-CORR_W: idx]
    corrs = sub.apply(lambda col: ewm_corr(col, tgt, HALF_LF*2))
    if corrs.abs().sum() == 0:
        corrs[:] = 1 / len(corrs)
    else:
        corrs = corrs / corrs.abs().sum()
    weights.loc[Score.index[idx]] = corrs
weights = weights.ffill().bfill()

# 원시 및 최종 리스크 지수
raw_idx   = (Score * weights).sum(axis=1)
risk_idx  = roll_pct(raw_idx, CORR_W).clip(0,100)

# 백테스팅
etf_tkr  = "069500.KS"  # KODEX 200 ETF
tlt_tkr  = "IEF"        # iShares 7-10 Year Treasury Bond ETF

# ETF 및 채권 데이터 다운로드
data = yf.download([etf_tkr, tlt_tkr], start=risk_idx.index.min(), end=risk_idx.index.max(), auto_adjust=True)["Close"]
weekly_dates   = risk_idx.asfreq("W-FRI").index
risk_weekly    = risk_idx.reindex(weekly_dates).ffill()
prices_weekly  = data.reindex(weekly_dates).ffill()
ret_weekly = prices_weekly.pct_change().dropna()

# 자산 배분: 리스크 지수 높을수록 ETF 비중 증가
weights_port = pd.DataFrame({
    etf_tkr:  risk_weekly / 100,
    tlt_tkr: (100 - risk_weekly) / 100
}, index=ret_weekly.index)

# 거래비용 계산 (0.2%)
trade_amount = (weights_port.shift(1) - weights_port).abs().sum(axis=1)
cost = trade_amount * 0.002
port_ret = (weights_port * ret_weekly).sum(axis=1) - cost

# 전략 성과 지표 계산
years = len(port_ret) / 52
cagr = (port_ret.add(1).prod()) ** (1/years) - 1
sharpe = port_ret.mean() / port_ret.std() * np.sqrt(52)
cum_ret = (1 + port_ret).cumprod()
rolling_max = cum_ret.cummax()
drawdown = (cum_ret - rolling_max) / rolling_max
mdd = drawdown.min()
downside_ret = port_ret[port_ret < 0]
downside_std = downside_ret.std() if len(downside_ret) > 0 else 0
sortino = port_ret.mean() / downside_std * np.sqrt(52) if downside_std > 0 else np.nan

print("\n전략 성과:")
print(f"CAGR:            {cagr:.2%}")
print(f"샤프 비율:       {sharpe:.2f}")
print(f"소르티노 비율:   {sortino:.2f}")
print(f"최대 낙폭:       {mdd:.2%}")

# 바이 앤 홀드 성과 계산
ret_bh = prices_weekly[etf_tkr].pct_change().dropna()
cagr_bh = (1 + ret_bh).prod() ** (1/years) - 1
sharpe_bh = ret_bh.mean() / ret_bh.std() * np.sqrt(52)
cum_ret_bh = (1 + ret_bh).cumprod()
rolling_max_bh = cum_ret_bh.cummax()
drawdown_bh = (cum_ret_bh - rolling_max_bh) / rolling_max_bh
mdd_bh = drawdown_bh.min()
downside_ret_bh = ret_bh[ret_bh < 0]
downside_std_bh = downside_ret_bh.std() if len(downside_ret_bh) > 0 else 0
sortino_bh = ret_bh.mean() / downside_std_bh * np.sqrt(52) if downside_std_bh > 0 else np.nan

print("\n바이 앤 홀드 성과:")
print(f"CAGR:            {cagr_bh:.2%}")
print(f"샤프 비율:       {sharpe_bh:.2f}")
print(f"소르티노 비율:   {sortino_bh:.2f}")
print(f"최대 낙폭:       {mdd_bh:.2%}")

# 시각화
# 1. 지표 가중치 시각화
fig_w = px.line(weights, title="지표별 가중치 (EWM)")
fig_w.update_layout(
    xaxis_title="날짜",
    yaxis_title="가중치",
    legend_title="지표"
)
fig_w.show()

# 2. 리스크 지수 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=risk_idx.index, y=risk_idx, name="리스크 지수", line=dict(color="red")))
fig.add_trace(go.Scatter(x=kospi.index, y=kospi, name="KOSPI", yaxis="y2", line=dict(color="blue", dash="dot")))
fig.add_shape(type="rect", x0=risk_idx.index[0], x1=risk_idx.index[-1], y0=80, y1=100, fillcolor="red", opacity=0.2, layer="below")
fig.add_shape(type="rect", x0=risk_idx.index[0], x1=risk_idx.index[-1], y0=0, y1=20, fillcolor="green", opacity=0.2, layer="below")
fig.update_layout(
    title="KOSPI 리스크 지수 (0-100) vs KOSPI",
    yaxis=dict(title="리스크 (0-100)", range=[0,100]),
    yaxis2=dict(title="KOSPI 지수", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.02),
    template="plotly_white"
)
fig.show()

# 3. 누적 수익률 비교 시각화
strategy_cum_ret = (1 + port_ret).cumprod()
bh_cum_ret = (1 + ret_bh).cumprod()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=strategy_cum_ret.index, y=strategy_cum_ret, name="전략", line=dict(color="orange")))
fig2.add_trace(go.Scatter(x=bh_cum_ret.index, y=bh_cum_ret, name="바이 앤 홀드", line=dict(color="green")))
fig2.update_layout(
    title="전략 vs 바이 앤 홀드 누적 수익률",
    yaxis_title="누적 수익률",
    xaxis_title="날짜",
    legend=dict(orientation="h", y=1.02),
    template="plotly_white"
)
fig2.show()

# 결과 저장
result = Score.copy()
result["RAW_IDX"] = raw_idx
result["KOSPI_RISK_IDX"] = risk_idx
result.to_csv("kospi_risk_index_dynamic_v2_1.csv")
print("✔ CSV 저장 완료")
