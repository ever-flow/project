# 2분 54초
# 조화평균
# 산업분류 변수 추가
# 이상치 코드로 제거

# !pip install optuna
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import optuna
import warnings
warnings.filterwarnings("ignore")

# 1) 산업 PER 기반 피처 계산 (조화평균 버전)
def calculate_industry_per_feature(df):
    import numpy as np

    industry_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5']
    industries = df[industry_cols].stack().dropna().unique()

    harm_means = {}
    for ind in industries:
        mask = df[industry_cols].apply(lambda r: ind in r.values, axis=1)
        sub = df.loc[mask & df['Net Income (LTM)'].gt(0) & df['Net Income (LTM)'].notna()]
        per_vals = sub['market_cap'] / sub['Net Income (LTM)']
        if len(per_vals) > 0:
            harm_means[ind] = len(per_vals) / (1.0 / per_vals).sum()
        else:
            harm_means[ind] = np.nan

    def feature(row):
        nets = row['Net Income (LTM)']
        if nets <= 0 or pd.isna(nets):
            return np.nan    # ← 수정: 0이 아닌 NaN 반환
        inds = [row[c] for c in industry_cols if pd.notna(row[c])]
        hms = [harm_means[i] for i in inds if not np.isnan(harm_means.get(i, np.nan))]
        if not hms:
            return np.nan    # ← 수정: 산업 정보 없으면 NaN
        avg_hm = np.mean(hms)
        return nets * avg_hm

    return df.apply(feature, axis=1)

# 1.5) 산업 EV/EBITDA 기반 피처 계산 (조화평균 버전)
def calculate_industry_EV_EBITDA_feature(df):
    import numpy as np

    industry_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5']
    industries = df[industry_cols].stack().dropna().unique()

    harm_means_ev = {}
    for ind in industries:
        mask = df[industry_cols].apply(lambda r: ind in r.values, axis=1)
        sub = df.loc[
            mask &
            df['EBITDA (LTM)'].gt(0) &
            df['EBITDA (LTM)'].notna() &
            df['Enterprise Value (FQ0)'].notna()
        ]
        ev_vals = sub['Enterprise Value (FQ0)'] / sub['EBITDA (LTM)']
        if len(ev_vals) > 0:
            harm_means_ev[ind] = len(ev_vals) / (1.0 / ev_vals).sum()
        else:
            harm_means_ev[ind] = np.nan

    def feature_ev(row):
        ebitda = row['EBITDA (LTM)']
        if ebitda <= 0 or pd.isna(ebitda):
            return np.nan   # ← 수정: 0이 아닌 NaN 반환
        inds = [row[c] for c in industry_cols if pd.notna(row[c])]
        hms = [harm_means_ev[i] for i in inds if not np.isnan(harm_means_ev.get(i, np.nan))]
        if not hms:
            return np.nan   # ← 수정: 산업 정보 없으면 NaN
        avg_hm = np.mean(hms)
        return ebitda * avg_hm

    return df.apply(feature_ev, axis=1)

# 2) 도메인·시간 기반 피처 엔지니어 클래스
class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Xt = self._transform(X.copy())
        self.cols_ = Xt.columns.tolist()
        print(f"\n사용 가능한 변수 리스트 ({len(self.cols_)}개):")
        for col in self.cols_:
            print(f"  {col}")
        return self

    def transform(self, X):
        Xt = self._transform(X.copy())
        return Xt.reindex(columns=self.cols_, fill_value=0)

    def _transform(self, X):
        X = X.drop('market', axis=1, errors='ignore')
        tcols = [c for c in X.columns if '(' in c and 'LTM' in c]
        tf = {}
        for c in tcols:
            base, per = c.split(' (')
            tf.setdefault(base, {})[per.rstrip(')')] = c

        if 'EBIT' in tf and 'Depreciation' in tf:
            for p, e in tf['EBIT'].items():
                d = tf['Depreciation'].get(p)
                if d:
                    X[f'EBITDA ({p})'] = X[e] + X[d]
            tf['EBITDA'] = {p: f'EBITDA ({p})' for p in tf['EBIT']}

        bases = ['Total Assets', 'Total Liabilities', 'Equity', 'Net Debt',
                 'Revenue', 'EBIT', 'EBITDA', 'Net Income', 'Net Income After Minority']

        def safe_div(a, b):
            return a.div(b.replace({0: np.nan}))

        for v in bases:
            periods = tf.get(v, {})
            seq = [p for p in ['LTM-3', 'LTM-2', 'LTM-1', 'LTM'] if p in periods]
            grs = []
            for i in range(1, len(seq)):
                prev, curr = seq[i-1], seq[i]
                r = safe_div(X[periods[curr]] - X[periods[prev]], X[periods[prev]])
                suffix = '-2' if (prev, curr) == ('LTM-3', 'LTM-2') else '-1' if (prev, curr) == ('LTM-2', 'LTM-1') else ''
                X[f'{v}_growth{suffix}'] = r
                grs.append(r)
            if grs:
                X[f'{v}_avg_growth'] = pd.concat(grs, axis=1).mean(axis=1)
                X[f'{v}_volatility'] = X[[periods[p] for p in seq]].std(axis=1)
            if 'LTM-3' in periods and 'LTM' in periods:
                X[f'{v}_CAGR'] = np.where(
                    X[periods['LTM-3']] > 0,
                    (X[periods['LTM']] / X[periods['LTM-3']])**(1/3) - 1,
                    np.nan
                )

        def calc(a, b, name, per):
            if per in tf.get(a, {}) and per in tf.get(b, {}):
                X[f'{name} ({per})'] = safe_div(X[tf[a][per]], X[tf[b][per]])

        ratios = [
            ('Equity', 'Total Assets', 'BAR'),
            ('Total Liabilities', 'Equity', 'DBR'),
            ('Revenue', 'Total Assets', 'SAR'),
            ('EBIT', 'Revenue', 'OMR'),
            ('EBITDA', 'Revenue', 'EMR'),
            ('Net Income', 'Total Assets', 'EAR'),
            ('Net Income', 'Equity', 'EBR')
        ]
        for p in ['LTM', 'LTM-1', 'LTM-2', 'LTM-3']:
            for a, b, n in ratios:
                calc(a, b, n, p)
            if p in ['LTM-2', 'LTM-1', 'LTM']:
                ni = tf.get('Net Income After Minority', {}).get(p)
                eq = tf.get('Equity', {}).get(p)
                prev = {'LTM-2': 'LTM-3', 'LTM-1': 'LTM-2', 'LTM': 'LTM-1'}[p]
                ep = tf.get('Equity', {}).get(prev)
                if ni and eq and ep:
                    avg_eq = (X[eq] + X[ep]) / 2
                    X[f'ROE ({p})'] = safe_div(X[ni], avg_eq)

        ratio_bases = ['BAR', 'DBR', 'SAR', 'OMR', 'EMR', 'EAR', 'EBR', 'ROE']
        for v in ratio_bases:
            periods = {p: f'{v} ({p})' for p in ['LTM-3', 'LTM-2', 'LTM-1', 'LTM'] if f'{v} ({p})' in X.columns}
            seq = [p for p in ['LTM-3', 'LTM-2', 'LTM-1', 'LTM'] if p in periods]
            grs = []
            for i in range(1, len(seq)):
                prev, curr = seq[i-1], seq[i]
                r = safe_div(X[periods[curr]] - X[periods[prev]], X[periods[prev]])
                suffix = '-2' if (prev, curr) == ('LTM-3', 'LTM-2') else '-1' if (prev, curr) == ('LTM-2', 'LTM-1') else ''
                X[f'{v}_growth{suffix}'] = r
                grs.append(r)
            if grs:
                X[f'{v}_avg_growth'] = pd.concat(grs, axis=1).mean(axis=1)
                X[f'{v}_volatility'] = X[[periods[p] for p in seq]].std(axis=1)
            if 'LTM-3' in periods and 'LTM' in periods:
                X[f'{v}_CAGR'] = np.where(
                    X[periods['LTM-3']] > 0,
                    (X[periods['LTM']] / X[periods['LTM-3']])**(1/3) - 1,
                    np.nan
                )

        dep_cols = [c for c in X.columns if c.startswith('Depreciation')]
        if dep_cols:
            X.drop(columns=dep_cols, inplace=True)

        industry_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5']
        for col in industry_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        return X

# 3) Optuna를 통한 LightGBM 하이퍼파라미터 최적화 (원본 로직 복원)
def optimize_lightgbm(X, y, n_trials=30):
    def objective(trial):
        param = {
            'objective': 'quantile',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        coverages, ratios = [], []

        cat_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5']
        cat_cols = [col for col in cat_cols if col in X.columns]

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            low_model = LGBMRegressor(alpha=0.1, **param)
            low_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='quantile',
                callbacks=[early_stopping(stopping_rounds=10, verbose=False)],
                categorical_feature=cat_cols
            )
            q_low = np.clip(low_model.predict(X_val), y.min() * 0.01, None)

            high_model = LGBMRegressor(alpha=0.9, **param)
            high_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='quantile',
                callbacks=[early_stopping(stopping_rounds=10, verbose=False)],
                categorical_feature=cat_cols
            )
            q_high = np.maximum(high_model.predict(X_val), q_low)

            within = (y_val >= q_low) & (y_val <= q_high)
            coverages.append(within.mean())
            ratios.append((q_high / q_low).mean())

        mean_cov = np.mean(coverages)
        mean_ratio = np.mean(ratios)
        if mean_cov < 0.7:
            mean_ratio += (0.7 - mean_cov) * 100
        return mean_ratio

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    best_params = study.best_params
    best_params.update({
        'objective': 'quantile',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })
    return best_params

# 4) 최적화된 LightGBM 퀀타일 모델 실행 함수
def quick_lightgbm_quantile(df, name):
    df = df.copy()

    # EBITDA (LTM) 컬럼이 없으면 생성
    if 'EBITDA (LTM)' not in df.columns:
        if 'EBIT (LTM)' in df.columns and 'Depreciation (LTM)' in df.columns:
            df['EBITDA (LTM)'] = df['EBIT (LTM)'] + df['Depreciation (LTM)']

    # Listing Date를 days elapsed로 변환
    df['Listing_Age_Days'] = (
        pd.to_datetime('2024-12-31') - pd.to_datetime(df['Listing Date'])
    ).dt.days
    df.drop(columns=['Listing Date'], inplace=True)

    # industry_PER_feature 및 industry_EV_EBITDA_feature 계산
    df['industry_PER_feature'] = calculate_industry_per_feature(df)
    df['industry_EV_EBITDA_feature'] = calculate_industry_EV_EBITDA_feature(df)

    industry_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5']
    for col in industry_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # 설명 변수 구성: EV 드롭
    X = df.select_dtypes(include=[np.number, 'category']).drop(
        columns=[
            'Market Cap (2024-12-31)',
            'market_cap',
            'Enterprise Value (FQ0)'
        ], errors='ignore'
    )
    y = df['market_cap']

    fe = DomainFeatureEngineer().fit(X, y)
    X_fe = fe.transform(X)

    selector = LGBMRegressor(n_estimators=200, random_state=42)
    selector.fit(X_fe, y)
    imp = pd.Series(selector.feature_importances_, index=fe.cols_)
    thresh = imp.mean()
    selected = imp[imp > 1.1 * thresh].sort_values(ascending=False).index.tolist()   # 1.1으로 수정

    print(f"\n[{name}] 선택된 피처 ({len(selected)}개):")
    for feat, score in imp[imp > thresh].sort_values(ascending=False).items():
        print(f"  {feat}: {score:.2f}")

    X_sel = X_fe[selected]
    best_params = optimize_lightgbm(X_sel, y, n_trials=50)   # 30 or 50으로 수정

    cat_cols = [col for col in industry_cols if col in X_sel.columns]
    low_model = LGBMRegressor(alpha=0.1, **best_params)
    high_model = LGBMRegressor(alpha=0.9, **best_params)
    low_model.fit(X_sel, y, categorical_feature=cat_cols)
    high_model.fit(X_sel, y, categorical_feature=cat_cols)

    q_low  = np.clip(low_model.predict(X_sel), y.min() * 0.01, None)
    q_high = np.maximum(high_model.predict(X_sel), q_low)

    res = pd.DataFrame({
        'Ticker': df['ticker'],
        'Actual': y,
        'Q0.1': q_low,
        'Q0.9': q_high,
        'industry_PER_feature': df['industry_PER_feature'].values,
        'industry_EV_EBITDA_feature': df['industry_EV_EBITDA_feature'].values,
    })

    # PER_in_interval, EV_EBITDA_in_interval: NA 처리 포함
    res['PER_in_interval'] = np.where(
        res['industry_PER_feature'].isna(),
        np.nan,
        ((res['industry_PER_feature'] >= res['Q0.1']) & (res['industry_PER_feature'] <= res['Q0.9'])).astype(float)
    )
    res['EV_EBITDA_in_interval'] = np.where(
        res['industry_EV_EBITDA_feature'].isna(),
        np.nan,
        ((res['industry_EV_EBITDA_feature'] >= res['Q0.1']) & (res['industry_EV_EBITDA_feature'] <= res['Q0.9'])).astype(float)
    )

    # 포함된 특성 개수 (NaN은 무시하고 합산)
    res['features_in_interval_count'] = res[['PER_in_interval', 'EV_EBITDA_in_interval']].sum(axis=1)

    res['Within'] = (res['Actual'] >= res['Q0.1']) & (res['Actual'] <= res['Q0.9'])
    res['Ratio']  = res['Q0.9'] / res['Q0.1']

    print(f"\n{name} 결과: Within={res['Within'].mean():.4f}, Mean_ratio={res['Ratio'].mean():.4f}, ratio_std={res['Ratio'].std():.4f}")

    per_mean = res['PER_in_interval'].mean(skipna=True) * 100
    ev_mean = res['EV_EBITDA_in_interval'].mean(skipna=True) * 100
    print(f"[{name}] interval 내에 PER_in_interval 포함 비율: {per_mean:.2f}%")
    print(f"[{name}] interval 내에 EV_EBITDA_in_interval 평균 포함 비율: {ev_mean:.2f}%")

    display(res.head(50), res.tail(50))
    return res

# 5) 메인 실행
if __name__ == "__main__":
    PATH = "/content/4_29_데이터.xlsx"
    df = pd.read_excel(PATH, sheet_name="Sheet1")

    industry_cols = ['Industry1', 'Industry2', 'Industry3', 'Industry4', 'Industry5']
    for col in industry_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    df = df[df["market"].isin(["KOSDAQ", "KOSDAQ GLOBAL", "KOSPI"])]
    df["market_cap"] = df["Market Cap (2024-12-31)"]
    df = df[df["market_cap"] > 0]

    exclude_kosdaq = ['003380', '005290', '005990', '006730', '007330', '012700', '013120', '013310', '015750', '016250',
                      '018310', '019210', '021320', '023410', '023760', '025900', '025980', '027710', '028300', '030530',
                      '031330', '032190', '033160', '033290', '033640', '034810', '035080', '035600', '035760', '035810',
                      '035890', '035900', '036710', '036800', '036830', '036930', '037460', '038110', '038390', '038500',
                      '038540', '041190', '041510', '043370', '046890', '049070', '051500', '053700', '056190', '058470',
                      '061970', '064760', '064820', '067160', '067170', '067310', '067570', '067990', '069080', '071460',
                      '074600', '078020', '078340', '080420', '084110', '084850', '085660', '086520', '091700', '092190',
                      '096530', '100790', '101330', '104480', '112040', '115160', '121440', '122450', '122690', '123040',
                      '124500', '131970', '136480', '137400', '141080', '145020', '151860', '178320', '195940', '205470',
                      '214450', '215000', '215200', '222800', '240810', '247540', '253450', '263750', '267980', '293490',
                      '348370', '357780', '393890', '403870', '900290', '950130']
    exclude_kospi = ['000270', '000660', '000810', '000880', '001040', '003490', '003550', '004020', '005380', '005490',
                     '005830', '005930', '005940', '006400', '006800', '009540', '010950', '011170', '011200', '012330',
                     '015760', '016360', '017670', '023530', '023590', '024110', '028260', '030200', '032830', '034220',
                     '034730', '035420', '035720', '036460', '042660', '051910', '055550', '066570', '071050', '078930',
                     '086790', '088350', '096770', '097950', '105560', '138040', '138930', '139130', '139480', '207940',
                     '267250', '316140', '373220', '402340', '415640']
    exclude_all    = set(exclude_kosdaq + exclude_kospi)
    df = df[~df['ticker'].isin(exclude_all)].reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    missing_ratio = df[num_cols].isnull().mean(axis=1)
    df = df[missing_ratio <= 0.5].reset_index(drop=True)

    kosdaq = df[df["market"].isin(["KOSDAQ", "KOSDAQ GLOBAL"])]
    kospi  = df[df["market"] == "KOSPI"]

    res1 = quick_lightgbm_quantile(kosdaq, "KOSDAQ & KOSDAQ GLOBAL")
    res2 = quick_lightgbm_quantile(kospi,  "KOSPI")
