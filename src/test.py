import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

MERGED_CACHE_PATH = "/home/ubuntu/runyourai/kt/KT-DINJAE-2026-AI/roster_seongbuk_2024_full_merged.parquet"

categorical_cols = ["route_id", "board_stop_id", "alight_stop_id", "weekday", "weather"]
numeric_cols = ["hour", "is_holiday", "headway_sec"]
feature_cols = categorical_cols + numeric_cols

model_a = lgb.Booster(model_file="experiments/26.08.20.15-32-18/model_a.txt")
model_b = lgb.Booster(model_file="experiments/26.08.20.15-32-18/model_b.txt")

# ── 데이터 로드 및 전처리 (학습 때와 동일하게) ──
df = pd.read_parquet(MERGED_CACHE_PATH)
df = df[df["alight_stop_id"].notna()]
df["y_standing"] = (df["is_standing"] == "Y").astype(int)

# ── 12월 데이터만 추출 ──
dec_mask = df["board_datetime"] >= "2024-12-01"
X_dec = df.loc[dec_mask, feature_cols]
y_dec = df.loc[dec_mask, "y_standing"]

# ── 모델A 예측 (12월 자체 예측 — 학습에도 쓰인 데이터라 순수 검증은 아님, 참고용) ──
proba = model_a.predict(X_dec)
pred = (proba >= 0.5).astype(int)
auc = roc_auc_score(y_dec, proba)
acc = accuracy_score(y_dec, pred)
print(f"[모델A] 12월 데이터 자체 예측 AUC: {auc:.4f}")
print(f"[모델A] 12월 데이터 자체 예측 Accuracy: {acc*100:.2f}%")

# ── 모델B 예측 (12월 중 입석 케이스만) ──
dec_standing_mask = dec_mask & (df["y_standing"] == 1)
Xb_dec = df.loc[dec_standing_mask, feature_cols]
yb_dec = df.loc[dec_standing_mask, "standing_seconds"]

pred_b = model_b.predict(Xb_dec)
mae = mean_absolute_error(yb_dec, pred_b)
rmse = mean_squared_error(yb_dec, pred_b) ** 0.5
print(f"[모델B] 12월 데이터 자체 예측 MAE: {mae:.1f}초")
print(f"[모델B] 12월 데이터 자체 예측 RMSE: {rmse:.1f}초")

# ── 이전 검증 실험(12월 홀드아웃) 결과와 비교 ──
print("\n--- 비교 (이전 검증 실험, num_leaves=127, 1000트리 기준) ---")
print("모델A AUC: 0.9180, Accuracy: 88.29%")
print("모델B MAE: 153.5초, RMSE: 239.5초")