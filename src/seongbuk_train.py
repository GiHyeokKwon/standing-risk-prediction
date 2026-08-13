import time
import os
import json
import subprocess
import pandas as pd
import lightgbm as lgb
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import sys

# ── 0. 경로 설정 ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_DIR, "experiments")

AUTO_PUSH = True

DATASET_LABEL = "seongbuk_2024"
MERGED_CACHE_PATH = "/home/ubuntu/runyourai/kt/roster_seongbuk_2024_full_merged.parquet"

# ── CPU 최대 활용 ────────────────────────────────────
NUM_THREADS = os.cpu_count()
print(f"사용 가능 CPU 코어 수: {NUM_THREADS}")

# ── 하이퍼파라미터: A/B 공통 ────────────────────────
N_ESTIMATORS = 1  # 지금은 실험/파이프라인 확인용
FEATURE_FRACTION = 1.0
BAGGING_FRACTION = 1.0
BAGGING_FREQ = 0

# 모델A(분류) 전용
LEARNING_RATE_A = 0.1
REG_LAMBDA_A = 0.0
NUM_LEAVES_A = 31
MIN_CHILD_SAMPLES_A = 20

# 모델B(회귀) 전용
LEARNING_RATE_B = 0.1
REG_LAMBDA_B = 0.0
NUM_LEAVES_B = 31
MIN_CHILD_SAMPLES_B = 20

# ── 검증 방식 ──────────────────────────────────────
USE_TEMPORAL_SPLIT = True
TEMPORAL_CUTOFF = "2024-12-01 00:00:00"

RUN_TS = datetime.now().strftime("%y.%m.%d.%H-%M-%S")
RUN_DIR = os.path.join(EXPERIMENTS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)

# ── 로그 저장 설정 ──────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_path = os.path.join(RUN_DIR, "train_log.txt")
log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)

# ── STEP 1: 병합된 데이터 불러오기 (RAM 넉넉하니 그대로 통으로 로드) ──
categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
feature_cols = categorical_cols + numeric_cols

t0 = time.time()

if not os.path.exists(MERGED_CACHE_PATH):
    raise FileNotFoundError(f"병합된 데이터가 없습니다: {MERGED_CACHE_PATH}")

df = pd.read_parquet(MERGED_CACHE_PATH)
print(f"로딩 시간: {time.time()-t0:.1f}초")
print(df.shape)
print(df.isna().sum())

# ── STEP 2: 결측치 처리 ──────────────────────────────
df = df[df["alight_stop_id"].notna()]
df["y_standing"] = (df["is_standing"] == "Y").astype(int)
print("결측 제외 후:", df.shape)
print(df["y_standing"].value_counts())

# ── STEP 3: 학습/검증 분리 ────────────────────────────
if USE_TEMPORAL_SPLIT:
    print(f"\n검증 방식: 시간분할 (기준시각={TEMPORAL_CUTOFF})")
    cutoff = pd.Timestamp(TEMPORAL_CUTOFF)
    train_mask = df["board_datetime"] < cutoff

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "y_standing"]
    X_test = df.loc[~train_mask, feature_cols]
    y_test = df.loc[~train_mask, "y_standing"]

    standing_train_mask = train_mask & (df["y_standing"] == 1)
    standing_test_mask = (~train_mask) & (df["y_standing"] == 1)
    Xb_train = df.loc[standing_train_mask, feature_cols]
    yb_train = df.loc[standing_train_mask, "standing_seconds"]
    Xb_test = df.loc[standing_test_mask, feature_cols]
    yb_test = df.loc[standing_test_mask, "standing_seconds"]

    print(f"[모델A] 학습 행: {len(X_train):,} / 검증 행: {len(X_test):,}")
    print(f"[모델B] 학습 행: {len(Xb_train):,} / 검증 행: {len(Xb_test):,}")
else:
    print("\n검증 방식: 랜덤분할 (random_state=42)")
    X = df[feature_cols]
    y = df["y_standing"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    standing_df = df[df["y_standing"] == 1]
    Xb = standing_df[feature_cols]
    yb = standing_df["standing_seconds"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=42
    )

# RAM이 넉넉하니 df를 계속 들고 있어도 무방하지만, 습관적으로 정리
del df

# ── STEP 4: 모델A 학습 (입석여부 분류) ───────────────
wandb.init(
    project="bus-standing-prediction",
    name=f"model_a_{DATASET_LABEL}_{N_ESTIMATORS}trees_{'temporal' if USE_TEMPORAL_SPLIT else 'random'}_{RUN_TS}",
    config={
        "model": "A_classifier",
        "dataset_label": DATASET_LABEL,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE_A,
        "num_leaves": NUM_LEAVES_A,
        "min_child_samples": MIN_CHILD_SAMPLES_A,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "reg_lambda": REG_LAMBDA_A,
        "num_threads": NUM_THREADS,
        "split_type": "temporal" if USE_TEMPORAL_SPLIT else "random",
        "train_rows": len(X_train),
        "features": feature_cols,
    }
)

model_a = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE_A,
    num_leaves=NUM_LEAVES_A,
    min_child_samples=MIN_CHILD_SAMPLES_A,
    feature_fraction=FEATURE_FRACTION,
    bagging_fraction=BAGGING_FRACTION,
    bagging_freq=BAGGING_FREQ,
    reg_lambda=REG_LAMBDA_A,
    n_jobs=NUM_THREADS,
    force_col_wise=True,  # 코어 많은 환경에서 스레드 분배 오버헤드 자동 탐지 스킵 (속도 개선)
)
model_a.fit(
    X_train, y_train,
    categorical_feature=categorical_cols,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
        wandb_callback(),
    ]
)
log_summary(model_a.booster_, save_model_checkpoint=False)

proba = model_a.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
auc_a = roc_auc_score(y_test, proba)
acc_a = accuracy_score(y_test, pred)
print("[모델A] AUC:", auc_a)
print("[모델A] Accuracy:", acc_a)

imp_a_split = pd.Series(model_a.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[모델A 피처 중요도 - split]")
print(imp_a_split)

imp_a_gain = pd.Series(
    model_a.booster_.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
print("\n[모델A 피처 중요도 - gain]")
print(imp_a_gain)

wandb.log({
    "model_a/AUC": auc_a,
    "model_a/Accuracy": acc_a,
    "model_a/feature_importance_split": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_a_split.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model A Feature Importance (split)"
    ),
    "model_a/feature_importance_gain": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_a_gain.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model A Feature Importance (gain)"
    ),
})

model_a_path = os.path.join(RUN_DIR, "model_a.txt")
model_a.booster_.save_model(model_a_path)  # 리눅스 경로엔 한글이 없어서 우회 저장 불필요

artifact_a = wandb.Artifact("model_a", type="model")
artifact_a.add_file(model_a_path)
wandb.log_artifact(artifact_a)
wandb.finish()

# ── STEP 5: 모델B 학습 (입석시간 회귀) ───────────────
wandb.init(
    project="bus-standing-prediction",
    name=f"model_b_{DATASET_LABEL}_{N_ESTIMATORS}trees_{'temporal' if USE_TEMPORAL_SPLIT else 'random'}_{RUN_TS}",
    config={
        "model": "B_regressor",
        "dataset_label": DATASET_LABEL,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE_B,
        "num_leaves": NUM_LEAVES_B,
        "min_child_samples": MIN_CHILD_SAMPLES_B,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "reg_lambda": REG_LAMBDA_B,
        "num_threads": NUM_THREADS,
        "split_type": "temporal" if USE_TEMPORAL_SPLIT else "random",
        "train_rows": len(Xb_train),
        "features": feature_cols,
    }
)

model_b = lgb.LGBMRegressor(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE_B,
    num_leaves=NUM_LEAVES_B,
    min_child_samples=MIN_CHILD_SAMPLES_B,
    feature_fraction=FEATURE_FRACTION,
    bagging_fraction=BAGGING_FRACTION,
    bagging_freq=BAGGING_FREQ,
    reg_lambda=REG_LAMBDA_B,
    n_jobs=NUM_THREADS,
    force_col_wise=True,
)
model_b.fit(
    Xb_train, yb_train,
    categorical_feature=categorical_cols,
    eval_set=[(Xb_train, yb_train), (Xb_test, yb_test)],
    eval_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
        wandb_callback(),
    ]
)
log_summary(model_b.booster_, save_model_checkpoint=False)

pred_b = model_b.predict(Xb_test)
mae = mean_absolute_error(yb_test, pred_b)
rmse = mean_squared_error(yb_test, pred_b) ** 0.5
print(f"[모델B] MAE: {mae:.1f}초  RMSE: {rmse:.1f}초  (평균 입석시간: {yb_train.mean():.1f}초)")

imp_b_split = pd.Series(model_b.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[모델B 피처 중요도 - split]")
print(imp_b_split)

imp_b_gain = pd.Series(
    model_b.booster_.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
print("\n[모델B 피처 중요도 - gain]")
print(imp_b_gain)

wandb.log({
    "model_b/MAE": mae,
    "model_b/RMSE": rmse,
    "model_b/feature_importance_split": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_b_split.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model B Feature Importance (split)"
    ),
    "model_b/feature_importance_gain": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_b_gain.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model B Feature Importance (gain)"
    ),
})

model_b_path = os.path.join(RUN_DIR, "model_b.txt")
model_b.booster_.save_model(model_b_path)

artifact_b = wandb.Artifact("model_b", type="model")
artifact_b.add_file(model_b_path)
wandb.log_artifact(artifact_b)
wandb.finish()

print(f"\n저장 완료: {model_a_path}, {model_b_path}")

# ── STEP 6: metrics.json 저장 ─────────────────────────
metrics = {
    "run_ts": RUN_TS,
    "dataset_label": DATASET_LABEL,
    "num_threads": NUM_THREADS,
    "n_estimators": N_ESTIMATORS,
    "feature_fraction": FEATURE_FRACTION,
    "bagging_fraction": BAGGING_FRACTION,
    "bagging_freq": BAGGING_FREQ,
    "learning_rate_a": LEARNING_RATE_A,
    "reg_lambda_a": REG_LAMBDA_A,
    "num_leaves_a": NUM_LEAVES_A,
    "min_child_samples_a": MIN_CHILD_SAMPLES_A,
    "learning_rate_b": LEARNING_RATE_B,
    "reg_lambda_b": REG_LAMBDA_B,
    "num_leaves_b": NUM_LEAVES_B,
    "min_child_samples_b": MIN_CHILD_SAMPLES_B,
    "split_type": "temporal" if USE_TEMPORAL_SPLIT else "random",
    "temporal_cutoff": TEMPORAL_CUTOFF if USE_TEMPORAL_SPLIT else None,
    "train_rows_a": len(X_train),
    "train_rows_b": len(Xb_train),
    "model_a": {"auc": auc_a, "accuracy": acc_a},
    "model_b": {"mae": mae, "rmse": rmse},
    "feature_importance_a_gain": imp_a_gain.to_dict(),
    "feature_importance_b_gain": imp_b_gain.to_dict(),
}
metrics_path = os.path.join(RUN_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"metrics.json 저장 완료: {metrics_path}")

# ── STEP 7: 로그 파일 닫고 stdout 원복 ──────────────
sys.stdout = sys.stdout.files[0]
log_file.close()

# ── STEP 8: Git add + commit + push ──────────────────
def run_git(args):
    result = subprocess.run(
        ["git"] + args, cwd=REPO_DIR,
        capture_output=True, text=True, encoding="utf-8"
    )
    print(f"$ git {' '.join(args)}")
    print(result.stdout)
    if result.returncode != 0:
        print("⚠ git 에러:", result.stderr)
    return result.returncode == 0

rel_path = os.path.relpath(RUN_DIR, REPO_DIR)

if AUTO_PUSH:
    ok = run_git(["add", rel_path])
    if ok:
        ok = run_git(["commit", "-m", f"Add experiment result {RUN_TS} ({DATASET_LABEL}, threads={NUM_THREADS}, A: lr={LEARNING_RATE_A}/leaves={NUM_LEAVES_A}, B: lr={LEARNING_RATE_B}/leaves={NUM_LEAVES_B}, AUC={auc_a:.4f}, MAE={mae:.1f}s)"])
    if ok:
        ok = run_git(["push"])
    if ok:
        print(f"\n✅ GitHub에 experiments/{RUN_TS}/ 업로드 완료")
    else:
        print(f"\n⚠ 자동 push 실패 — 수동으로 'git add {rel_path} && git commit && git push' 실행 필요")
else:
    print(f"\n(AUTO_PUSH 꺼짐)")