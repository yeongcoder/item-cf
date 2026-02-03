import random
import os
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from math import sqrt, log2
from typing import Dict, List, Tuple
import time

# =========================================
# 0) 설정
# =========================================
CSV_PATH = "data/_select_o_o_mid_op_op_pcode_p_p_name_UNIX_TIMESTAMP_op_op_rdate__202602021014.csv"
NEIGHBOR_CSV_PATH = "neighbors/item_cf_neighbors.csv"

SIM_METRIC = "cosine"
MIN_COCOUNT = 2
TOPK_NEIGHBORS = 50
TOPN_RECS = 20
DEDUP_PER_USER = True
RUN_EVAL = True
EVAL_K = 20

# =========================================
# 1) 데이터 로드 & 전처리
# =========================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"user_id", "item_id", "ts"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}, but got {set(df.columns)}")

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["item_name"] = df["item_name"].astype(str)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0).astype(np.int64)

    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)
    return df

# =========================================
# 2) 유저별 아이템 시퀀스
# =========================================
def build_user_items(df: pd.DataFrame, dedup_per_user=True):
    user_items = {}
    for uid, g in df.groupby("user_id", sort=False):
        items = g["item_id"].tolist()
        if dedup_per_user:
            items = list(dict.fromkeys(items))
        user_items[uid] = items
    return user_items

# =========================================
# 2-1) 아이템ID-이름 매핑
# =========================================
def build_item_name_map(df: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(df["item_id"], df["item_name"]))

# =========================================
# 3) 아이템/페어 count
# =========================================
def count_items_and_pairs(user_items, min_cocount=1):
    item_count = defaultdict(int)
    pair_count = defaultdict(int)

    for items in user_items.values():
        item_set = set(items)
        for it in item_set:
            item_count[it] += 1
        for i, j in combinations(sorted(item_set), 2):
            pair_count[(i, j)] += 1

    if min_cocount > 1:
        pair_count = {k: v for k, v in pair_count.items() if v >= min_cocount}

    return dict(item_count), dict(pair_count), len(user_items)

# =========================================
# 4) 유사도 계산
# =========================================
def compute_similarity(item_count, pair_count, N_users, metric="cosine"):
    sims = {}
    for (i, j), co in pair_count.items():
        ci = item_count.get(i, 0)
        cj = item_count.get(j, 0)
        if ci == 0 or cj == 0:
            continue

        if metric == "cosine":
            sim = co / sqrt(ci * cj)
        elif metric == "jaccard":
            sim = co / (ci + cj - co)
        elif metric == "lift":
            sim = (co * N_users) / (ci * cj)
        elif metric == "pmi":
            val = (co * N_users) / (ci * cj)
            sim = log2(val) if val > 0 else -999.0
        else:
            raise ValueError("unknown metric")

        sims[(i, j)] = float(sim)
    return sims

# =========================================
# 5) TopK 이웃 구축
# =========================================
def build_item_neighbors(sims, topk=50):
    neigh = defaultdict(list)
    for (i, j), s in sims.items():
        neigh[i].append((j, s))
        neigh[j].append((i, s))

    neighbors = {}
    for item, lst in neigh.items():
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)[:topk]
        neighbors[item] = lst_sorted

    return neighbors

# =========================================
# 6) 추천 생성
# =========================================
def recommend_for_user(user_id, user_items, neighbors, topn=10):
    seen = set(user_items.get(user_id, []))
    scores = defaultdict(float)

    for it in seen:
        for nb, s in neighbors.get(it, []):
            if nb in seen:
                continue
            scores[nb] += s

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

# =========================================
# 7) 이웃 테이블 CSV 저장
# =========================================
def save_neighbors_csv(neighbors, path=NEIGHBOR_CSV_PATH):
    rows = []
    for i, lst in neighbors.items():
        for j, s in lst:
            rows.append((i, j, s))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=["item_id", "neighbor_item_id", "similarity"])
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved neighbors → {path}")

# =========================================
# 8) 이웃 테이블 CSV 로드
# =========================================
def load_neighbors_csv(path=NEIGHBOR_CSV_PATH):
    df = pd.read_csv(path)
    neighbors = defaultdict(list)
    for _, row in df.iterrows():
        neighbors[row["item_id"]].append((row["neighbor_item_id"], row["similarity"]))
    return neighbors

# =========================================
# 9) 평가
# =========================================
def evaluate_leave_one_out(user_items, neighbors, k=20):
    hits = 0
    ndcg_sum = 0.0
    n_eval = 0

    for uid, items in user_items.items():
        if len(items) < 2:
            continue
        gt = items[-1]
        hist = items[:-1]

        temp_user_items = {uid: hist}
        recs = recommend_for_user(uid, temp_user_items, neighbors, topn=k)
        rec_list = [it for it, _ in recs]

        n_eval += 1
        if gt in rec_list:
            hits += 1
            rank = rec_list.index(gt) + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)

    hr = hits / n_eval if n_eval else 0
    ndcg = ndcg_sum / n_eval if n_eval else 0
    return hr, ndcg, n_eval

# =========================================
# 10) 명령에 따라 실행
# =========================================
def main():
    start_main = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="이웃테이블 생성")
    parser.add_argument("--recommend", action="store_true", help="랜덤 유저 추천")
    args = parser.parse_args()

    # 1) 데이터 로드
    # print("\n⏱️  [데이터 로드 시작...]")
    
    start_load = time.time()
    df = load_data(CSV_PATH)
    time_load = time.time() - start_load
    # print(f"   ✓ load_data: {time_load*1000:.3f}ms")
    
    start_build = time.time()
    user_items = build_user_items(df, dedup_per_user=DEDUP_PER_USER)
    time_build = time.time() - start_build
    # print(f"   ✓ build_user_items: {time_build*1000:.3f}ms")
    
    start_name = time.time()
    item_name_map = build_item_name_map(df)
    time_name = time.time() - start_name
    # print(f"   ✓ build_item_name_map: {time_name*1000:.3f}ms")

    # -----------------------------
    # (A) 이웃테이블 생성
    # -----------------------------
    if args.build:
        item_count, pair_count, N_users = count_items_and_pairs(user_items, min_cocount=MIN_COCOUNT)

        # 각 유사도 측정법 비교
        metrics = ["cosine", "jaccard", "lift", "pmi"]
        results = {}
        
        print("\n" + "="*60)
        print("📊 유사도 측정법 성능 비교")
        print("="*60)
        
        for metric in metrics:
            print(f"\n🔍 Testing {metric.upper()} similarity...")
            sims = compute_similarity(item_count, pair_count, N_users, metric=metric)
            neighbors = build_item_neighbors(sims, topk=TOPK_NEIGHBORS)
            
            if RUN_EVAL:
                hr, ndcg, n_eval = evaluate_leave_one_out(user_items, neighbors, k=EVAL_K)
                results[metric] = (hr, ndcg)
                print(f"   ✓ HR@{EVAL_K}={hr:.4f} | NDCG@{EVAL_K}={ndcg:.4f}")
        
        # 결과 요약
        print("\n" + "="*60)
        print("📈 최종 비교 결과")
        print("="*60)
        for metric in metrics:
            hr, ndcg = results[metric]
            print(f"{metric.upper():10s} → HR@{EVAL_K}={hr:.4f} | NDCG@{EVAL_K}={ndcg:.4f}")
        
        # 최고 성능 찾기
        best_metric = max(results.items(), key=lambda x: x[1][0])[0]
        best_hr, best_ndcg = results[best_metric]
        print(f"\n🏆 최고 성능: {best_metric.upper()} (HR={best_hr:.4f})")
        print("="*60)
        
        # 최고 성능 metric으로 최종 저장
        print(f"\n💾 {best_metric.upper()}로 최종 neighbors 저장 중...")
        sims = compute_similarity(item_count, pair_count, N_users, metric=best_metric)
        neighbors = build_item_neighbors(sims, topk=TOPK_NEIGHBORS)
        save_neighbors_csv(neighbors, NEIGHBOR_CSV_PATH)
        
        # 사용된 metric 저장
        metric_file = NEIGHBOR_CSV_PATH.replace(".csv", "_metric.txt")
        with open(metric_file, "w") as f:
            f.write(best_metric)
        print(f"✓ Saved neighbors with {best_metric.upper()} metric\n")

    # -----------------------------
    # (B) 추천만 실행
    # -----------------------------
    if args.recommend:
        # 파일 없으면 경고
        if not os.path.exists(NEIGHBOR_CSV_PATH):
            print("⚠️ neighbors CSV가 없습니다. 먼저 아래를 실행하세요:")
            print("   python index.py --build")
            return

        neighbors = load_neighbors_csv(NEIGHBOR_CSV_PATH)
        
        # 사용된 metric 확인
        metric_file = NEIGHBOR_CSV_PATH.replace(".csv", "_metric.txt")
        if os.path.exists(metric_file):
            with open(metric_file, "r") as f:
                used_metric = f.read().strip()
            print(f"📊 사용된 유사도: {used_metric.upper()}\n")

        sample_user = random.choice(list(user_items.keys()))
        print("\n🎯 Sample user:", sample_user)
        print("history:")
        for it in user_items[sample_user][:20]:
            item_name = item_name_map.get(it, "Unknown")
            print(f"  {it} | {item_name}")

        recs = recommend_for_user(sample_user, user_items, neighbors, topn=TOPN_RECS)

        print("\n🔽 Top recommendations")
        for it, sc in recs:
            item_name = item_name_map.get(it, "Unknown")
            print(f"{it} | {item_name} | {sc}")
    
    # main 함수 전체 실행시간 출력
    time_main = time.time() - start_main
    # print(f"\n⏱️  [main] 전체 실행시간: {time_main*1000:.3f}ms")


if __name__ == "__main__":
    main()
