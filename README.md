# Item-Based Collaborative Filtering (Item-CF) Recommendation System Prototype

아이템 기반 협업 필터링을 활용한 상품 추천 시스템입니다. 사용자의 구매 이력을 분석하여 유사한 상품을 추천합니다.

## 🎯 프로젝트 개요

이 프로젝트는 다음을 구현합니다:
- **아이템 유사도 계산**: Cosine, Jaccard, Lift, PMI 등 4가지 유사도 측정법 지원
- **이웃 테이블 생성**: 각 아이템의 유사 상품 Top-K 저장
- **개인화 추천**: 사용자의 구매 이력을 기반으로 다음 상품 추천
- **성능 평가**: Leave-One-Out 방식의 HR@K, NDCG@K 평가

## 📊 주요 기능

### 1. 데이터 로드 & 전처리
- CSV 파일에서 사용자-상품 상호작용 데이터 로드
- 사용자별 아이템 시퀀스 구성 (시간 순서 정렬)
- 중복 제거 옵션 제공

### 2. 유사도 계산
4가지 유사도 측정법 비교:
- **Cosine Similarity**: sim(i,j) = co-occur(i,j) / sqrt(count(i) × count(j))
- **Jaccard Similarity**: sim(i,j) = co-occur(i,j) / (count(i) + count(j) - co-occur(i,j))
- **Lift**: sim(i,j) = (co-occur(i,j) × N) / (count(i) × count(j))
- **PMI**: sim(i,j) = log2((co-occur(i,j) × N) / (count(i) × count(j)))

### 3. 이웃 테이블 생성
- 각 아이템별 유사 상품 Top-K (기본값: 50개) 저장
- CSV 형식으로 저장하여 추천 시 빠르게 로드

### 4. 개인화 추천
- 사용자의 구매 이력 분석
- 보유 아이템과 유사한 미구매 상품 추천
- 상품명과 유사도 점수 함께 출력

### 5. 성능 평가
- Leave-One-Out 방식으로 마지막 구매를 GT로 설정
- **HR@K (Hit Rate)**: Top-K 내 GT 포함 여부 (정확성)
- **NDCG@K**: 랭킹 고려 평가

## � 평가 지표 상세 설명

### Leave-One-Out 검증 (evaluate_leave_one_out 함수)

각 유사도 측정법을 평가하는 방식:

**평가 원리:**
- 각 사용자의 **마지막 구매 아이템** → Ground Truth (정답)
- **그 이전 구매 아이템들** → 모델 입력 데이터
- 모델이 정답을 추천했는지 검증

**평가 지표:**

1. **HR@K (Hit Rate)**
   ```
   HR@K = (정답을 추천 목록 Top-K에 포함시킨 사용자 수) / (평가 대상 총 사용자 수)
   ```
   - 추천이 정답을 맞췄는가? (YES/NO)
   - 0~1 사이의 값 (높을수록 좋음)
   - 예: HR@20=0.3245 → 32.45%의 사용자에게 정답을 Top-20에 포함

2. **NDCG@K (Normalized Discounted Cumulative Gain)**
   ```
   NDCG@K = (1 / log2(rank + 1)) for each user / (최적값)
   ```
   - 정답이 몇 번째 순위인가?
   - 순위가 높을수록 더 높은 점수 (Rank 1: 1.0, Rank 2: 0.63, ...)
   - 정확도 + 랭킹 순서 모두 고려
   - 0~1 사이의 값 (높을수록 좋음)

**코드 예시:**
```python
def evaluate_leave_one_out(user_items, neighbors, k=20):
    hits = 0           # 정답을 맞춘 사용자 수
    ndcg_sum = 0.0     # 누적 NDCG 점수
    n_eval = 0         # 평가 대상 사용자 수
    
    for uid, items in user_items.items():
        if len(items) < 2:  # 최소 2개 이상의 구매 필요
            continue
        
        gt = items[-1]              # 마지막 아이템 = Ground Truth
        hist = items[:-1]           # 이전 아이템들 = 입력
        
        # 이전 아이템만 사용해서 추천 생성
        recs = recommend_for_user(uid, {uid: hist}, neighbors, topn=k)
        rec_list = [it for it, _ in recs]
        
        n_eval += 1
        if gt in rec_list:          # 정답이 Top-K에 있는가?
            hits += 1               # HR 카운트
            rank = rec_list.index(gt) + 1  # 몇 번째 순위?
            ndcg_sum += 1.0 / np.log2(rank + 1)  # NDCG 계산
    
    hr = hits / n_eval      # 최종 HR@K
    ndcg = ndcg_sum / n_eval  # 최종 NDCG@K
    return hr, ndcg, n_eval
```

**실제 예시:**

사용자 A의 구매 이력: `[상품1, 상품2, 상품3, 상품4]`
- **모델 입력:** `[상품1, 상품2, 상품3]`
- **정답 (Ground Truth):** `상품4`
- **추천 결과:** `[상품3.5점, 상품4 0.8점, 상품2.5점, ...]`

결과:
- ✅ HR: 정답(상품4)이 Top-20에 있음 → HR +1
- ✅ NDCG: 정답이 2번째 순위 → 1/log₂(3) ≈ 0.631 점

### 최고 성능 모델 선택

`--build` 실행 시 다음 과정을 거칩니다:
1. 4가지 유사도 측정법(Cosine, Jaccard, Lift, PMI) 각각 적용
2. Leave-One-Out 방식으로 각 방법의 HR@K, NDCG@K 계산
3. **HR@K가 가장 높은 방법을 최종 모델로 선택**
4. 선택된 모델의 이웃 테이블을 CSV로 저장

## �📋 설정 파라미터

```python
CSV_PATH = "data/_select_o_o_mid_op_op_pcode_p_p_name_UNIX_TIMESTAMP_op_op_rdate__202602021014.csv"
NEIGHBOR_CSV_PATH = "neighbors/item_cf_neighbors.csv"

SIM_METRIC = "cosine"          # 유사도 측정법: cosine, jaccard, lift, pmi
MIN_COCOUNT = 2                # 최소 공동 구매 횟수 (threshold)
TOPK_NEIGHBORS = 50            # 각 아이템의 이웃 개수
TOPN_RECS = 20                 # 추천 결과 수
DEDUP_PER_USER = True          # 사용자별 중복 제거
RUN_EVAL = True                # 평가 실행 여부
EVAL_K = 20                    # 평가 시 K값
```

## � 환경 설정 및 설치

### 필수 요구사항
- Python 3.7 이상
- macOS / Linux / Windows

### 1. 저장소 클론

```bash
git clone https://github.com/yeongcoder/-.git
cd Item-CF
```

### 2. Python 가상환경 생성 (권장)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 필수 패키지 설치

```bash
pip install --upgrade pip
pip install pandas numpy scipy scikit-learn
```

또는 requirements.txt에서:
```bash
pip install -r requirements.txt
```

**requirements.txt 내용:**
```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=0.24.0
```

### 4. 데이터 준비

CSV 파일을 `data/` 디렉토리에 배치하세요:
```
data/
└── _select_o_o_mid_op_op_pcode_p_p_name_UNIX_TIMESTAMP_op_op_rdate__202602021014.csv
```

CSV 파일에는 다음 컬럼이 필수:
- `user_id`: 사용자 ID
- `item_id`: 상품 ID
- `item_name`: 상품명
- `ts`: 타임스탐프

## �🚀 사용 방법

### 1. 이웃 테이블 생성 (학습)

4가지 유사도 측정법을 비교하여 최고 성능의 모델 저장:

```bash
python3 index.py --build
```

**출력 예시:**
```
============================================================
📊 유사도 측정법 성능 비교
============================================================

🔍 Testing COSINE similarity...
   ✓ HR@20=0.3245 | NDCG@20=0.2156

🔍 Testing JACCARD similarity...
   ✓ HR@20=0.3180 | NDCG@20=0.2089

...

📈 최종 비교 결과
============================================================
COSINE     → HR@20=0.3245 | NDCG@20=0.2156
JACCARD    → HR@20=0.3180 | NDCG@20=0.2089
LIFT       → HR@20=0.3012 | NDCG@20=0.1934
PMI        → HR@20=0.2890 | NDCG@20=0.1756

🏆 최고 성능: COSINE (HR=0.3245)
============================================================
```

### 2. 추천 실행 (추론)

랜덤 사용자에 대해 상품 추천:

```bash
python3 index.py --recommend
```

**출력 예시:**
```
📊 사용된 유사도: COSINE

🎯 Sample user: del_ko_123456789
history:
  K6634-S1249-C5502 | 이순신수산 통영 해산 해녀가 채취한 제철 자연산 참 해삼 500g
  Y9700-M4881-G1849 | 국내 자연산 생물 보라성게 성게알 우니 100g부터
  ...

🔽 Top recommendations
U5071-J1590-R5296 | 산지직송 통영굴 삼배체굴 석화 하프셀 생굴 손질 바위굴 벚굴 | 0.8234
M2010-Z3456-A7890 | 제주도 자연산 전복 중 500g | 0.7856
...
```

## 📁 디렉토리 구조

```
Item-CF/
├── index.py                              # 메인 프로그램
├── README.md                             # 이 파일
├── .gitignore                            # Git 무시 파일
├── data/
│   └── _select_o_o_mid_op_op_pcode_p_p_name_UNIX_TIMESTAMP_op_op_rdate__202602021014.csv
│       └── 사용자-상품 상호작용 데이터 (user_id, item_id, item_name, ts)
└── neighbors/
    ├── item_cf_neighbors.csv             # 생성된 이웃 테이블
    └── item_cf_neighbors_metric.txt      # 사용된 유사도 측정법
```

## 📊 데이터 포맷

### 입력 데이터 (CSV)

필수 컬럼:
- `user_id`: 사용자 ID (문자열)
- `item_id`: 상품 ID (문자열)
- `item_name`: 상품명 (문자열)
- `ts`: 타임스탬프 (Unix Timestamp)

### 생성되는 이웃 테이블 (CSV)

```csv
item_id,neighbor_item_id,similarity
K6634-S1249-C5502,Y9700-M4881-G1849,0.8234
K6634-S1249-C5502,U5071-J1590-R5296,0.7856
...
```

## 🔧 핵심 함수

| 함수명 | 설명 |
|--------|------|
| `load_data()` | CSV 파일 로드 및 전처리 |
| `build_user_items()` | 사용자별 아이템 시퀀스 생성 |
| `build_item_name_map()` | 아이템ID-이름 매핑 생성 |
| `count_items_and_pairs()` | 아이템 및 페어 카운팅 |
| `compute_similarity()` | 유사도 계산 (4가지 방법) |
| `build_item_neighbors()` | Top-K 이웃 테이블 생성 |
| `recommend_for_user()` | 특정 사용자에 대한 추천 생성 |
| `evaluate_leave_one_out()` | 모델 성능 평가 (HR@K, NDCG@K) |

## 💡 알고리즘 플로우

```
1. 데이터 로드
   └─ 사용자별 아이템 시퀀스 구성

2. 이웃 테이블 생성 (--build)
   ├─ 아이템/페어 카운팅
   ├─ 4가지 유사도 측정법 적용
   ├─ 각 방법별 성능 평가
   └─ 최고 성능 모델을 이웃 테이블로 저장

3. 추천 (--recommend)
   ├─ 이웃 테이블 로드
   ├─ 사용자의 구매 아이템 분석
   ├─ 유사 아이템 가중합 계산
   └─ Top-N 추천 결과 출력
```

## 🎓 주요 개념

### 아이템 기반 협업 필터링 (Item-CF)
- 상품 간 유사도를 기반으로 추천
- 사용자의 과거 구매 상품과 유사한 다른 상품 추천
- User-CF와 달리 새로운 사용자도 추천 가능 (Cold-start 완화)

### 유사도 측정법 비교
- **Cosine**: 벡터 각도 기반, 일반적인 추천 시스템에서 효과적
- **Jaccard**: 집합 겹침 비율, 희소 데이터에 강건
- **Lift**: 기대도 대비 실제 동시 구매율, 강한 연관성 탐지
- **PMI**: 정보이론 기반, 통계적 의존성 측정

## 📈 성능 지표

- **HR@K (Hit Rate)**: 추천 목록에서 정답을 맞춘 비율
- **NDCG@K**: 랭킹 위치를 고려한 평가 메트릭 (0~1, 높을수록 좋음)

## ⚡ 빠른 시작 (Quick Start)

```bash
# 1. 프로젝트 셋업
git clone https://github.com/yeongcoder/-.git
cd Item-CF
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install pandas numpy scipy

# 2. 이웃 테이블 생성 (모든 유사도 측정법 비교)
python3 index.py --build

# 3. 추천 실행
python3 index.py --recommend

# 4. 가상환경 빠져나가기
deactivate
```

## 🐛 트러블슈팅

### ImportError: No module named 'pandas'
```bash
pip install pandas numpy
```

### FileNotFoundError: CSV 파일을 찾을 수 없음
- CSV 파일이 `data/` 디렉토리에 있는지 확인
- 파일 경로가 `index.py`의 `CSV_PATH`와 일치하는지 확인

### PermissionError: 이웃 테이블 저장 실패
```bash
mkdir -p neighbors
chmod 755 neighbors
```

### 느린 성능
- `MIN_COCOUNT` 값을 증가시켜 페어 개수 감소
- `EVAL_K` 값을 감소시켜 평가 속도 향상
- `RUN_EVAL` 을 False로 설정하면 평가 과정 생략

## 📚 참고 자료

### Item-Based Collaborative Filtering
- [Amazon.com Recommendations: Item-to-Item Collaborative Filtering](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)
- [Item-Based Collaborative Filtering Recommendation Algorithms](https://www.semanticscholar.org/paper/Item-Based-Collaborative-Filtering-Recommendation-Sarwar/a2d7c6c7d91f11e2c6b8d6b1d1f1e1e1e)

### 유사도 측정
- Cosine Similarity
- Jaccard Index
- Lift & PMI (Association Rules)

## 📝 라이선스

내부 프로젝트

## 👨‍💻 작성자

서영제
