"""
배포용 최종 학습 — 검증 홀드아웃 없이 366일(1~12월) 전체를 학습에 사용.
하이퍼파라미터는 이전 실험(num_leaves=127)에서 검증된 값을 그대로 사용하되,
learning_rate/reg_lambda는 이번 요청대로 A/B 동일하게 통일.
"""

import time
import os
import json
import subprocess
import pandas as pd
import lightgbm as lgb
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary
from datetime import datetime
import sys

# ── 0. 경로 설정 ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_DIR, "experiments")

AUTO_PUSH = True

DATASET_LABEL = "seongbuk_2024"
RUN_LABEL = "deploy_final"  # 배포용 최종 학습임을 명시
MERGED_CACHE_PATH = "/home/ubuntu/runyourai/kt/KT-DINJAE-2026-AI/roster_seongbuk_2024_full_merged.parquet"

NUM_THREADS = os.cpu_count()
print(f"사용 가능 CPU 코어 수: {NUM_THREADS}")

# ── 하이퍼파라미터: 배포용 (검증셋 없이 전체 학습) ────
N_ESTIMATORS = 3000
FEATURE_FRACTION = 1.0
BAGGING_FRACTION = 1.0
BAGGING_FREQ = 0

LEARNING_RATE_A = 0.1
REG_LAMBDA_A = 0.0
NUM_LEAVES_A = 127
MIN_CHILD_SAMPLES_A = 20

LEARNING_RATE_B = 0.1
REG_LAMBDA_B = 0.0
NUM_LEAVES_B = 127
MIN_CHILD_SAMPLES_B = 20

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

# ── STEP 1: 병합된 데이터 불러오기 (검증 분할 없이 전체 사용) ──
categorical_cols = ["route_id", "board_stop_id", "alight_stop_id", "weekday", "weather"]
numeric_cols = ["hour", "is_holiday", "headway_sec"]
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

# ── STEP 3: 전체를 학습에 사용 (배포용 — 검증 분할 없음) ──
X_train = df[feature_cols]
y_train = df["y_standing"]

standing_df = df[df["y_standing"] == 1]
Xb_train = standing_df[feature_cols]
yb_train = standing_df["standing_seconds"]

print(f"\n[배포용 전체 학습] 모델A 학습 행: {len(X_train):,}")
print(f"[배포용 전체 학습] 모델B 학습 행: {len(Xb_train):,}")

del df, standing_df

# ── STEP 4: 모델A 학습 (검증셋 없이, early stopping 없이 5000트리 고정) ──
wandb.init(
    project="bus-standing-prediction",
    name=f"model_a_{DATASET_LABEL}_{RUN_LABEL}_{N_ESTIMATORS}trees_{RUN_TS}",
    config={
        "model": "A_classifier",
        "dataset_label": DATASET_LABEL,
        "run_label": RUN_LABEL,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE_A,
        "num_leaves": NUM_LEAVES_A,
        "min_child_samples": MIN_CHILD_SAMPLES_A,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "reg_lambda": REG_LAMBDA_A,
        "num_threads": NUM_THREADS,
        "split_type": "none_full_train",
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
    force_col_wise=True,
)

t_fit_start = time.time()
model_a.fit(
    X_train, y_train,
    categorical_feature=categorical_cols,
    eval_set=[(X_train, y_train)],   # 진행상황 모니터링용 (train self-eval, 조기종료 없음)
    eval_names=['train'],
    callbacks=[
        lgb.log_evaluation(50),
    ]
)
elapsed_a = time.time() - t_fit_start
print(f"[모델A] 학습 소요시간: {elapsed_a:.1f}초, 트리 수: {N_ESTIMATORS} (전량 사용, 조기종료 없음)")

log_summary(model_a.booster_, save_model_checkpoint=False)

imp_a_gain = pd.Series(
    model_a.booster_.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
print("\n[모델A 피처 중요도 - gain]")
print(imp_a_gain)

wandb.log({
    "model_a/fit_seconds": elapsed_a,
    "model_a/feature_importance_gain": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_a_gain.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model A Feature Importance (gain)"
    ),
})

model_a_path = os.path.join(RUN_DIR, "model_a.txt")
model_a.booster_.save_model(model_a_path)

artifact_a = wandb.Artifact("model_a", type="model")
artifact_a.add_file(model_a_path)
wandb.log_artifact(artifact_a)
wandb.finish()

# ── STEP 5: 모델B 학습 (검증셋 없이, early stopping 없이 5000트리 고정) ──
wandb.init(
    project="bus-standing-prediction",
    name=f"model_b_{DATASET_LABEL}_{RUN_LABEL}_{N_ESTIMATORS}trees_{RUN_TS}",
    config={
        "model": "B_regressor",
        "dataset_label": DATASET_LABEL,
        "run_label": RUN_LABEL,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE_B,
        "num_leaves": NUM_LEAVES_B,
        "min_child_samples": MIN_CHILD_SAMPLES_B,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "reg_lambda": REG_LAMBDA_B,
        "num_threads": NUM_THREADS,
        "split_type": "none_full_train",
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

t_fit_start_b = time.time()
model_b.fit(
    Xb_train, yb_train,
    categorical_feature=categorical_cols,
    eval_set=[(Xb_train, yb_train)],
    eval_names=['train'],
    callbacks=[
        lgb.log_evaluation(50),
    ]
)
elapsed_b = time.time() - t_fit_start_b
print(f"[모델B] 학습 소요시간: {elapsed_b:.1f}초, 트리 수: {N_ESTIMATORS} (전량 사용, 조기종료 없음)")

log_summary(model_b.booster_, save_model_checkpoint=False)

imp_b_gain = pd.Series(
    model_b.booster_.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
print("\n[모델B 피처 중요도 - gain]")
print(imp_b_gain)

wandb.log({
    "model_b/fit_seconds": elapsed_b,
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
    "run_label": RUN_LABEL,
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
    "split_type": "none_full_train",
    "note": "배포용 최종 모델 — 검증 홀드아웃 없이 1~12월 전체 데이터로 학습. "
            "성능 지표(AUC/MAE 등)는 이전 시간분할 검증 실험(12월 홀드아웃) 결과를 참조.",
    "train_rows_a": len(X_train),
    "train_rows_b": len(Xb_train),
    "model_a": {"fit_seconds": elapsed_a, "trees": N_ESTIMATORS},
    "model_b": {"fit_seconds": elapsed_b, "trees": N_ESTIMATORS},
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
        ok = run_git(["commit", "-m", f"Add DEPLOY FINAL model {RUN_TS} ({DATASET_LABEL}, full 12-month train, n_estimators={N_ESTIMATORS}, leaves=127, lr=0.1, lambda=0.0)"])
    if ok:
        ok = run_git(["push"])
    if ok:
        print(f"\n✅ GitHub에 experiments/{RUN_TS}/ 업로드 완료")
    else:
        print(f"\n⚠ 자동 push 실패 — 수동으로 'git add {rel_path} && git commit && git push' 실행 필요")
else:
    print(f"\n(AUTO_PUSH 꺼짐)")