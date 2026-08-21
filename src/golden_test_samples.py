"""
골든 테스트용 입력-출력 샘플 세트 생성 (배포용 최종 모델 기준)
"""

import json
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

# ── 0. 경로 설정 ──────────────────────────────────────
EXPERIMENT_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\KT-DINJAE-2026-AI\experiments\26.08.20.15-32-18"
MERGED_PATH = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster_seongbuk_2024_full_merged.parquet"

MODEL_A_PATH = r"C:\model_a.txt"
MODEL_B_PATH = r"C:\model_b.txt"
MAPPING_A_PATH = EXPERIMENT_DIR + r"\category_code_mapping_model_a.json"
MAPPING_B_PATH = EXPERIMENT_DIR + r"\category_code_mapping_model_b.json"
OUTPUT_PATH = EXPERIMENT_DIR + r"\golden_test_samples.json"

CATEGORICAL_COLS = ["route_id", "board_stop_id", "alight_stop_id", "weekday", "weather"]
NUMERIC_COLS = ["hour", "is_holiday", "headway_sec"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

N_COMMON_SAMPLES = 10
N_RARE_SAMPLES = 5
N_BOUNDARY_SAMPLES = 5


def load_models_and_mappings():
    model_a = lgb.Booster(model_file=MODEL_A_PATH)
    model_b = lgb.Booster(model_file=MODEL_B_PATH)
    with open(MAPPING_A_PATH, "r", encoding="utf-8") as f:
        mapping_a = json.load(f)
    with open(MAPPING_B_PATH, "r", encoding="utf-8") as f:
        mapping_b = json.load(f)
    return model_a, model_b, mapping_a, mapping_b


def encode_row(row_dict, mapping):
    encoded = {}
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            col_map = mapping.get(col, {})
            code = col_map.get(str(row_dict[col]))
            if code is None:
                raise ValueError(f"매핑에 없는 값: {col}={row_dict[col]} (학습 데이터에 없던 값)")
            encoded[col] = code
        else:
            encoded[col] = row_dict[col]
    return encoded


def predict_sample(model_a, model_b, encoded_a, encoded_b):
    import numpy as np
    X_a = np.array([[encoded_a[col] for col in FEATURE_COLS]], dtype=float)
    X_b = np.array([[encoded_b[col] for col in FEATURE_COLS]], dtype=float)

    proba = float(model_a.predict(X_a)[0])
    standing_seconds = float(model_b.predict(X_b)[0])

    return {
        "model_a_proba_standing": round(proba, 6),
        "model_a_is_standing": "Y" if proba >= 0.5 else "N",
        "model_b_predicted_standing_seconds": round(standing_seconds, 2),
    }


def main():
    print("모델·매핑 로드 중...")
    model_a, model_b, mapping_a, mapping_b = load_models_and_mappings()

    print("병합 데이터 로드 중 (일부만 샘플링)...")
    needed_cols = FEATURE_COLS + ["sample_count_route_segment_hour"]
    parquet_file = pq.ParquetFile(MERGED_PATH)

    collected_chunks = []
    total_valid_rows = 0
    TARGET_ROWS = 200_000

    for batch in parquet_file.iter_batches(columns=needed_cols, batch_size=500_000):
        chunk = batch.to_pandas()
        chunk = chunk.dropna(subset=FEATURE_COLS)
        if len(chunk) > 0:
            collected_chunks.append(chunk)
            total_valid_rows += len(chunk)
        if total_valid_rows >= TARGET_ROWS:
            break

    df = pd.concat(collected_chunks, ignore_index=True)
    print(f"샘플링용으로 {len(df):,}행 확보")

    samples = []

    # ── 1. 일반적인 케이스 ──
    common_pool = df[df["sample_count_route_segment_hour"] >= 100]
    common_rows = common_pool.sample(n=min(N_COMMON_SAMPLES, len(common_pool)), random_state=42)

    # ── 2. 희귀 케이스 ──
    rare_pool = df[(df["sample_count_route_segment_hour"] >= 1) & (df["sample_count_route_segment_hour"] < 10)]
    rare_rows = rare_pool.sample(n=min(N_RARE_SAMPLES, len(rare_pool)), random_state=42)

    # ── 3. 경계값 케이스 ──
    boundary_pool = df[
        (df["hour"].isin([0, 1, 23])) |
        (df["headway_sec"] <= df["headway_sec"].quantile(0.01)) |
        (df["headway_sec"] >= df["headway_sec"].quantile(0.99))
    ]
    boundary_rows = boundary_pool.sample(n=min(N_BOUNDARY_SAMPLES, len(boundary_pool)), random_state=42)

    all_rows = pd.concat([
        common_rows.assign(sample_type="common"),
        rare_rows.assign(sample_type="rare"),
        boundary_rows.assign(sample_type="boundary"),
    ], ignore_index=True)

    print(f"총 {len(all_rows)}개 샘플 선정, 예측 실행 중...")

    for i, row in all_rows.iterrows():
        raw_input = {col: row[col] for col in FEATURE_COLS}
        raw_input["is_holiday"] = int(row["is_holiday"])
        raw_input["hour"] = int(row["hour"])
        raw_input["headway_sec"] = float(row["headway_sec"])

        try:
            encoded_a = encode_row(raw_input, mapping_a)
            encoded_b = encode_row(raw_input, mapping_b)
        except ValueError as e:
            print(f"  [건너뜀] 샘플 {i}: {e}")
            continue

        prediction = predict_sample(model_a, model_b, encoded_a, encoded_b)

        samples.append({
            "sample_id": f"golden_{len(samples)+1:03d}",
            "sample_type": row["sample_type"],
            "raw_input": raw_input,
            "encoded_input_model_a": encoded_a,
            "encoded_input_model_b": encoded_b,
            "expected_output": prediction,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(samples)}개 샘플 (common: {sum(1 for s in samples if s['sample_type']=='common')}, "
          f"rare: {sum(1 for s in samples if s['sample_type']=='rare')}, "
          f"boundary: {sum(1 for s in samples if s['sample_type']=='boundary')})")


if __name__ == "__main__":
    main()
