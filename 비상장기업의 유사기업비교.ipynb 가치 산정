# 코스피, 코스닥 나눠서 fitting
# interval_ratio의 평균이 아닌 중앙값을 minimize 하는 방식으로 학습

# 과적합 방지를 위해서

!pip install optuna

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
import shap
import warnings
import matplotlib.pyplot as plt
from IPython.display import display
import optuna

warnings.filterwarnings("ignore")

# 1. 데이터 불러오기 및 필터링
FILE_PATH = "/content/최종_연습용_2.7_최종_데이터.xlsx"
df = pd.read_excel(FILE_PATH, sheet_name="Sheet1")
df = df[df["market"].isin(["KOSDAQ", "KOSDAQ GLOBAL", "KOSPI"])]

# 2. 종속변수 처리
financial_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if "Market Cap (2024-12-31)" in financial_columns:
    financial_columns.remove("Market Cap (2024-12-31)")

target_col = "Market Cap (2024-12-31)"
if 'exchange_rates' not in globals():
    exchange_rates = {"KOSDAQ": 1, "KOSDAQ GLOBAL": 1, "KOSPI": 1}
df["market_cap"] = df[target_col] / df["market"].map(exchange_rates)
df = df[df["market_cap"] >= 0]

# 3. 데이터 분리: KOSDAQ & KOSDAQ GLOBAL vs KOSPI
kosdaq_df = df[df["market"].isin(["KOSDAQ", "KOSDAQ GLOBAL"])]
kospi_df = df[df["market"] == "KOSPI"]

# 4. 이상치 제거 함수
def remove_outliers(df_group):
    numeric_data = df_group.select_dtypes(include='number')
    imputer = IterativeImputer(max_iter=5, random_state=42,
                               estimator=HistGradientBoostingRegressor(random_state=42))
    imputed_data = imputer.fit_transform(numeric_data)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    iso_forest = IsolationForest(contamination='auto', random_state=42)
    outlier_pred = iso_forest.fit_predict(scaled_data)
    return df_group[outlier_pred == 1]

# 5. 특성 공학 클래스
class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_transformed = self._transform_internal(X)
        self.feature_names_ = X_transformed.columns
        return self

    def transform(self, X):
        X_transformed = self._transform_internal(X)
        X_transformed = X_transformed.reindex(columns=self.feature_names_, fill_value=0)
        return X_transformed

    def _transform_internal(self, X):
        X = X.copy()
        if "market" in X.columns:
            X.drop(columns=["market"], inplace=True)

        time_cols = [col for col in X.columns if "(" in col and "LTM" in col]
        time_features = {}
        for col in time_cols:
            base = col.split(" (")[0]
            if base not in time_features:
                time_features[base] = {}
            period = col.split("(")[1].replace(")", "").strip()
            time_features[base][period] = col

        for base, periods in time_features.items():
            if "LTM" in periods and "LTM-1" in periods:
                current_col = periods["LTM"]
                prev_col = periods["LTM-1"]
                X[f"{base}_growth"] = (X[current_col] - X[prev_col]) / X[prev_col].replace({0: np.nan})
            if "LTM" in periods and "LTM-3" in periods:
                current_col = periods["LTM"]
                past_col = periods["LTM-3"]
                X[f"{base}_CAGR"] = np.where(
                    X[past_col] > 0,
                    (X[current_col] / X[past_col]) ** (1 / 3) - 1,
                    np.nan
                )
            available_periods = [periods[p] for p in ["LTM-3", "LTM-2", "LTM-1", "LTM"] if p in periods]
            if len(available_periods) >= 2:
                X[f"{base}_momentum"] = X[available_periods[-1]] - X[available_periods[0]]
                X[f"{base}_volatility"] = X[available_periods].std(axis=1)
                growth_rates = [
                    (X[available_periods[i]] - X[available_periods[i-1]]) / X[available_periods[i-1]].replace({0: np.nan})
                    for i in range(1, len(available_periods))
                ]
                if growth_rates:
                    X[f"{base}_avg_growth"] = pd.concat(growth_rates, axis=1).mean(axis=1)

        def safe_divide(a, b):
            return a / b.replace({0: np.nan})

        if "Revenue (LTM)" in X.columns and "Net Income (LTM)" in X.columns:
            X["profit_margin"] = safe_divide(X["Net Income (LTM)"], X["Revenue (LTM)"])
        if "EBIT (LTM)" in X.columns and "Revenue (LTM)" in X.columns:
            X["operating_margin"] = safe_divide(X["EBIT (LTM)"], X["Revenue (LTM)"])
        if "Net Income (LTM)" in X.columns and "Total Assets (LTM)" in X.columns:
            X["ROA"] = safe_divide(X["Net Income (LTM)"], X["Total Assets (LTM)"])
        if "Net Income (LTM)" in X.columns and "Equity (LTM)" in X.columns:
            X["ROE"] = safe_divide(X["Net Income (LTM)"], X["Equity (LTM)"])
        if "Net Debt (LTM)" in X.columns and "Equity (LTM)" in X.columns:
            X["debt_to_equity"] = safe_divide(X["Net Debt (LTM)"], X["Equity (LTM)"])
        if "Total Assets (LTM)" in X.columns and "Equity (LTM)" in X.columns:
            X["leverage"] = safe_divide(X["Total Assets (LTM)"], X["Equity (LTM)"])
        if "Revenue (LTM)" in X.columns and "Total Assets (LTM)" in X.columns:
            X["asset_turnover"] = safe_divide(X["Revenue (LTM)"], X["Total Assets (LTM)"])

        industry_cols = [col for col in X.columns if col.startswith('Industry')]
        if industry_cols:
            unique_industries = set(X[industry_cols].stack().dropna().unique())
            for industry in unique_industries:
                X[f'industry_{industry}'] = X[industry_cols].apply(
                    lambda row: 1 if industry in set(row.dropna()) else 0, axis=1
                )
            X.drop(columns=industry_cols, inplace=True)

        return X

# 6. VIF 제거 클래스
class VIFThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=8):
        self.thresh = thresh
        self.selected_features_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_temp = X.copy()
        while True:
            vif = pd.DataFrame({
                "VIF Factor": [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])],
                "features": X_temp.columns
            })
            if vif["VIF Factor"].max() < self.thresh:
                break
            drop_col = vif.sort_values("VIF Factor", ascending=False).iloc[0]["features"]
            X_temp = X_temp.drop(columns=[drop_col])
        self.selected_features_ = X_temp.columns
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.selected_features_]

# 7. 트리 기반 특성 선택 클래스
class TreeBasedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, threshold='median'):
        self.estimator = estimator if estimator else lgb.LGBMRegressor(random_state=42, n_jobs=-1, n_estimators=100)
        self.threshold = threshold
        self.selector = None
        self.selected_features_ = None

    def fit(self, X, y=None):
        self.selector = SelectFromModel(self.estimator, threshold=self.threshold)
        self.selector.fit(X, y)
        self.selected_features_ = X.columns[self.selector.get_support()]
        return self

    def transform(self, X):
        X_transformed = self.selector.transform(X)
        return pd.DataFrame(X_transformed, columns=self.selected_features_, index=X.index)

# 8. Quantile Crossing 방지 함수
def prevent_quantile_crossing(pred_01, pred_09):
    adjusted_01 = np.minimum(pred_01, pred_09)
    adjusted_09 = np.maximum(pred_01, pred_09)
    return adjusted_01, adjusted_09

# 9. 모델 구축 및 평가 함수
def build_and_evaluate(df_group, market_name):
    df_clean = remove_outliers(df_group)
    non_na_threshold = int(len(financial_columns) / 2) + 1
    df_clean = df_clean.dropna(subset=financial_columns, thresh=non_na_threshold)

    y = df_clean["market_cap"]
    exclude_cols = ["ticker", target_col, "market_cap", "CODE1", "CODE2", "CODE3", "CODE4", "CODE5", "market"]
    cols_to_drop = [col for col in df_clean.columns if "Enterprise Value" in col]
    exclude_cols.extend(cols_to_drop)
    X = df_clean.drop(columns=exclude_cols, errors="ignore")

    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[('feature_engineer', DomainFeatureEngineer(), X.columns)],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', IterativeImputer(max_iter=5, random_state=42,
                                     estimator=HistGradientBoostingRegressor(random_state=42))),
        ('scaler', RobustScaler()),
        ('vif', VIFThreshold(thresh=8)),
        ('feature_selector', TreeBasedFeatureSelector(threshold='1.25*median')),
    ])

    # 파이프라인의 앞단 작업을 훈련 데이터에 대해 한 번만 수행
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_val_processed = pipeline.transform(X_val)

    # Optuna 최적화 함수
    def objective(trial):
        # 하이퍼파라미터 제안
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }

        # 조기 종료 설정
        early_stopping = lgb.early_stopping(stopping_rounds=5, verbose=False)

        # 모델 학습
        model_01 = lgb.LGBMRegressor(objective='quantile', alpha=0.1, **params)
        model_09 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, **params)

        model_01.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            callbacks=[early_stopping]
        )
        model_09.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            callbacks=[early_stopping]
        )

        # 예측 및 조정
        pred_01 = model_01.predict(X_val_processed)
        pred_09 = model_09.predict(X_val_processed)
        pred_01, pred_09 = prevent_quantile_crossing(pred_01, pred_09)

        # 성능 평가
        within_interval = (y_val >= pred_01) & (y_val <= pred_09)
        coverage = within_interval.mean()
        interval_ratio = (pred_09 / pred_01).median()   # median 으로 수정

        if coverage >= 0.65:
            return interval_ratio
        else:
            return 1e6  # 페널티

    # Optuna 최적화
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, timeout=7200)   # timeout optuna 2시간 제한

    # 최종 모델 학습
    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        'objective': 'quantile'
    })

    # 전체 학습 데이터로 파이프라인 재적합
    X_processed = pipeline.fit_transform(X, y)

    model_01 = lgb.LGBMRegressor(alpha=0.1, **best_params)
    model_05 = lgb.LGBMRegressor(alpha=0.5, **best_params)
    model_09 = lgb.LGBMRegressor(alpha=0.9, **best_params)

    model_01.fit(X_processed, y)
    model_05.fit(X_processed, y)
    model_09.fit(X_processed, y)

    # 결과 생성
    result_df = pd.DataFrame({'Actual': y.values, 'Ticker': df_clean['ticker'].values})
    result_df['Quantile_0.1'] = model_01.predict(X_processed)
    result_df['Quantile_0.5'] = model_05.predict(X_processed)
    result_df['Quantile_0.9'] = model_09.predict(X_processed)

    result_df['Quantile_0.1'], result_df['Quantile_0.9'] = prevent_quantile_crossing(
        result_df['Quantile_0.1'], result_df['Quantile_0.9']
    )

    # 평가 지표
    for quantile in [0.1, 0.5, 0.9]:
        mape = mean_absolute_percentage_error(result_df['Actual'], result_df[f'Quantile_{quantile}'])
        print(f"{market_name} - Quantile {quantile}: MAPE = {mape:.4f}")

    # 교차 검증
    cv_preds = cross_val_predict(model_05, X_processed, y, cv=KFold(n_splits=3, shuffle=True, random_state=42))
    cv_mape = mean_absolute_percentage_error(y, cv_preds)
    print(f"{market_name} - Cross-validated MAPE (Quantile 0.5): {cv_mape:.4f}")

    # 커버리지 및 구간 비율
    result_df['Within_Interval'] = (result_df['Actual'] >= result_df['Quantile_0.1']) & \
                                   (result_df['Actual'] <= result_df['Quantile_0.9'])
    coverage = result_df['Within_Interval'].mean()
    print(f"{market_name} - Coverage: {coverage:.4f}")

    result_df['Interval_Ratio'] = result_df['Quantile_0.9'] / result_df['Quantile_0.1']
    avg_interval_ratio = result_df['Interval_Ratio'].mean()
    print(f"{market_name} - Average Interval Ratio: {avg_interval_ratio:.4f}")

    # SHAP 분석
    selected_features = pipeline.named_steps['feature_selector'].selected_features_
    X_shap = shap.sample(X_processed, 100, random_state=42)
    explainer = shap.TreeExplainer(model_05)
    shap_values = explainer.shap_values(X_shap)
    shap.summary_plot(shap_values, X_shap, feature_names=selected_features, plot_type="dot", show=False)
    plt.title(f"SHAP Feature Importance for {market_name} - Quantile 0.5")
    plt.savefig(f'shap_{market_name.lower().replace(" ", "_")}.png')

    return result_df, model_05, shap_values, selected_features, pipeline.named_steps['preprocessor'].named_transformers_['feature_engineer'].feature_names_

# 10. 모델 실행
kosdaq_result, kosdaq_model, kosdaq_shap_values, kosdaq_features, kosdaq_feature_names = build_and_evaluate(kosdaq_df, "KOSDAQ & KOSDAQ GLOBAL")
kospi_result, kospi_model, kospi_shap_values, kospi_features, kospi_feature_names = build_and_evaluate(kospi_df, "KOSPI")

# 11. 결과 출력
print("\nKOSDAQ & KOSDAQ GLOBAL Results:")
display(kosdaq_result.head(50))
display(kosdaq_result.tail(50))
print(f"Within 80% Interval Proportion: {kosdaq_result['Within_Interval'].mean():.4f}")
print(f"Average Interval Ratio: {kosdaq_result['Interval_Ratio'].mean():.4f}")

print("\nKOSPI Results:")
display(kospi_result.head(50))
display(kospi_result.tail(50))
print(f"Within 80% Interval Proportion: {kospi_result['Within_Interval'].mean():.4f}")
print(f"Average Interval Ratio: {kospi_result['Interval_Ratio'].mean():.4f}")

# 12. SHAP 중요도 정리
def display_shap_importance(shap_values, selected_indices, original_names, market_name):
    shap_importance = np.abs(shap_values).mean(axis=0)
    mapping_names = [original_names[i] for i in selected_indices]
    shap_summary_df = pd.DataFrame({
        'Feature_Index': selected_indices,
        'Feature_Name': mapping_names,
        'Mean_Abs_SHAP': shap_importance
    })
    shap_summary_df = shap_summary_df.sort_values(by="Mean_Abs_SHAP", ascending=False)
    print(f"\n{market_name} - Top 30 Feature Importance (SHAP):")
    display(shap_summary_df.head(30))

# 매핑된 SHAP 중요도 출력
display_shap_importance(kosdaq_shap_values, kosdaq_features, kosdaq_feature_names, "KOSDAQ & KOSDAQ GLOBAL")
display_shap_importance(kospi_shap_values, kospi_features, kospi_feature_names, "KOSPI")
