import json
import re
import os

EXPERIMENT_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\KT-DINJAE-2026-AI\experiments\26.08.20.15-32-18"

categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather"]

def extract_mapping(model_path):
    with open(model_path, "r", encoding="utf-8-sig") as f:  # -sig로 BOM 처리
        lines = f.readlines()

    last_line = lines[-1].strip()
    match = re.search(r"pandas_categorical:(\[.*\])", last_line)
    categories_list = json.loads(match.group(1))

    mapping = {}
    for col, cats in zip(categorical_cols, categories_list):
        mapping[col] = {str(v): i for i, v in enumerate(cats)}
    return mapping

for model_name in ["model_a", "model_b"]:
    model_path = os.path.join(EXPERIMENT_DIR, f"{model_name}.txt")
    output_path = os.path.join(EXPERIMENT_DIR, f"category_code_mapping_{model_name}.json")

    mapping = extract_mapping(model_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[{model_name}] 저장 완료: {output_path}")
    for col in categorical_cols:
        print(f"  {col}: {len(mapping[col])}개 카테고리")
    print()
