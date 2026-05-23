# BDA 학습자 수료 예측 AI 경진대회 풀이

> [데이콘 x BDA 제2회 학습자 수료 예측 AI 경진대회](https://dacon.io/competitions/official/236664/overview/description) | Private **26위** / 733팀 (상위 **3.5%**)

## 대회 결과

- **참가팀**: 733팀
- **평가 지표**: F1 Score
- **Public F1**: 0.44897 — 43위 (상위 5.7%)
- **Private F1**: 0.41071 — 26위 (상위 3.5%) ← Public 대비 +17등 상승

---

##  문제 정의

**핵심 문제**: Train(9기, 748명)과 Test(10기, 814명)의 분포가 다를 때, 어떻게 예측할 것인가?

### 데이터 이질성

| 변수 | L1 Distance | 판정 |
|------|-------------|------|
| major1_1 (세부전공) | 1.947 |  사용 불가 |
| onedayclass_topic | 1.912 |  사용 불가 |
| whyBDA | 0.413 |  일부만 사용 |
| job | 0.075 |  안정 |
| **re_registration** | **0.010** |  매우 안정 |

- **CV-LB Gap**: -0.1163 (정상: ±0.01) → 일반적 ML 과적합 위험

### 해결 전략
1. **안정적 신호 발견** — L1 < 0.2 변수 우선, Lift 통계 검증
2. **도메인 기반 Rule** — 페르소나 점수, Hard Kill
3. **순위 기반 예측** — Flip-rate (Top-k 선발, threshold 대신)

자세한 내용: [PROBLEM.md](PROBLEM.md)

---

##  실험 여정 (총 ~100회 제출)

| Phase | 기간 | 최고 점수 | 핵심 발견 |
|-------|------|-----------|----------|
| 0: 첫 시도 | Jan 13 | 0.310 | LGBM 베이스라인 |
| 1: CatBoost + Threshold | Jan 25~28 | 0.372 | CV-LB Gap -0.12 발견 |
| 2: Rule Discovery | Jan 29~Feb 12 | 0.441 | 페르소나+Kill Rule 효과 확인 |
| 3: NLP_V2 정밀화 | Feb 13~20 | **0.44897** | **최고점 달성** |

### 핵심 실험 비교

| 접근 | 점수 | 결과 |
|------|------|------|
| LGBM 단독 | 0.310 |  |
| CatBoost + threshold 튜닝 | 0.372 |  |
| **순수 ML 피처** | **0.384** | **역효과** |
| Pseudo Labeling | 0.334 |  역효과 |
| Rule v1 (기본 페르소나) | 0.418 |  |
| NLP_V2 + 5-seed 앙상블 | **0.44897** | **최고** |

>  **중요**: 순수 ML 피처 접근(`ML_FEATURE_VER_384`)이 **0.384**로 가장 낮았음.  
> 이 데이터셋에서는 도메인 Rule이 일반적 ML보다 강력.

전체 실험 기록: [EXPERIMENTS.md](EXPERIMENTS.md)

---

##  최종 접근법 (NLP_V2)

### 핵심 구조

```
[ Step 1: Feature Engineering ]
  refined_score  = 페르소나 점수 + NLP 키워드 점수
  is_non_major   = 비데이터 전공 여부
  len_total      = 자소서 전체 글자수

[ Step 2: 5-Seed × 5-Fold Ensemble ]
  CatBoost (범주형 포함)  →  OOF 예측
  LightGBM (수치형)       →  OOF 예측

[ Step 3: Meta-learning ]
  Logistic Regression
    Input: CatBoost OOF + LightGBM OOF
    → 최종 확률

[ Step 4: Hard Kill ]
  법학 관련  → 확률 = 0 (Lift 0.28)
  경영학 × 직장인/취준생 → 확률 = 0 (Lift 0.50)

[ Step 5: Flip-rate 선발 ]
  Top-375명 선발 (814명의 46.1%)
  → 0/1 예측
```

### Rule Score 핵심 구성

```python
# 페르소나 (안정 변수 기반)
자연과학 × 취준생:  +3점
인문학 × 대학생:    +2점

# 텍스트 맥락
"혼자" or "어려워" (긍정 맥락 없음): -2점
"경험" in text:  -2점

# 행동 신호
time_input == 4.0:     +2점
re_registration == 예:  +2점

# Hard Kill
법학 / 경영학×직장인:  확률 → 0
```

---

##  모델 구조 설계 근거

### CatBoost + LightGBM: 귀납적 편향의 다양성

| 모델 | 강점 | 역할 |
|------|------|------|
| CatBoost | 범주형 변수 34개 네이티브 처리 | 범주형 관계 포착 |
| LightGBM | 수치형·연속형 분포 처리 | 수치형 패턴 포착 |

두 모델의 **귀납적 편향이 다르기 때문에 오류가 겹치지 않는다.** CatBoost가 못 잡는 수치 패턴을 LightGBM이 보완하고, 반대도 마찬가지다.

### OOF Meta LR: 리키지 없는 스태킹

```
CatBoost (5-Fold OOF) → meta_cat
LightGBM (5-Fold OOF) → meta_lgb
              ↓
Meta Logistic Regression → 최종 확률
```

- OOF 예측으로 학습 → **Train → Meta LR 정보 누출 없음**
- LR 파라미터 수 극소 → **748명 소규모 데이터에서 과적합 위험 낮음**
- 두 모델의 잔류 오류 비상관화 → LR이 가중치 최적 조합 학습

### 5-Seed: 소규모 데이터 분산 흡수

```
748명 → 5-Fold → Fold당 약 150명
단일 Seed: Fold 배정 운이 결과에 과도하게 영향
5 Seeds × 5 Folds = 25회 → OOF 평균 → 분산 흡수
```

### Rule + ML: 도메인 지식이 ML에 앞선다

```
순수 ML 실험 (ML_FEATURE_VER): 0.384
도메인 Rule + ML (NLP_V2):     0.449  (+0.065)
```

Train(9기)과 Test(10기)의 분포가 다를 때 복잡한 ML 피처는 Train에 과적합된다. 도메인 Rule은 **"기수가 달라도 변하지 않는 인간의 동기 구조"**에 기반하므로 일반화된다. ML은 Rule Score를 피처로 받아 확률 보정 역할을 담당한다.

---

##  프로젝트 구조

```
BDA-Completion-Prediction/
├── README.md               # 이 파일
├── PROBLEM.md              # 문제 정의 + 데이터 분석
├── EXPERIMENTS.md          # 전체 실험 기록 (~100회)
│
├── configs/                # 하이퍼파라미터
│   ├── catboost.yaml
│   ├── lightgbm.yaml
│   ├── decision_tree.yaml
│   └── meta.yaml
│
├── src/
│   ├── config.py           # 경로/상수
│   └── features.py         # Feature Engineering
│
├── notebooks/              # 분석 노트북
│   ├── 00_problem_definition.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_train_test_drift.ipynb
│   ├── 03_lift_analysis.ipynb
│   └── 04_final_model.ipynb  # 최종 모델 구현 (정제·주석 버전)
│
├── run_rule_discovery.py   # Decision Tree Rule 추출
├── run_ensemble.py         # CatBoost + LightGBM 앙상블
├── run_stacking.py         # Meta-model + 최종 예측
│
├── plz.ipynb               # 원본 실험 코드 (아카이브)
└── test.ipynb              # 원본 실험 코드 (아카이브)
```

---

##  실행 방법

```bash
pip install -r requirements.txt

# 1. Rule 추출
python run_rule_discovery.py

# 2. 앙상블 학습
python run_ensemble.py

# 3. 최종 예측
python run_stacking.py
# → outputs/submission.csv
```

---

##  핵심 교훈

### 1. Train/Test 이질성 대응
> "데이터가 다를 때는 공통 신호만 써라"

L1 Distance로 안정 변수 선별, 불안정 변수(세부전공, 학교 코드 등)는 배제.

### 2. 순수 ML의 역설
> "복잡한 모델일수록 이질성 앞에서 더 많이 무너진다"

`ML_FEATURE_VER`: 0.384 vs `NLP_V2`: 0.449 — 65% 차이. 748명 소규모 데이터에서 ML 피처 과적합 발생.

### 3. Flip-rate
> "Threshold가 아닌 Ranking으로 예측하라"

확률 분포 밀집 → threshold 0.005 변화로 선발 인원 15명 변동.  
Top-k 고정 선발 → 안정. Top-k 선발이 Threshold보다 안정적.

### 4. 실패에서 배우기
실패율 ~95% (100회 중 ~5회만 개선). 하지만 실패의 패턴이 쌓여 최종 전략 완성.

---

##  한계점

### school1 (L1=0.718 — 불안정)
```python
if school in ['45', '38', '52']: score += 2  # Train EDA 기반, 실험적 적용
if school in ['81', '54', '19']: score -= 5
```
학교 신호 강화 실험(FINAL_SCHOOL_IMPROVED) 결과 0.449 → 0.418 악화. 현재 규칙은 미미하게 기여하지만 다음 기수에서도 동일하게 작동한다는 보장이 없다.

### 페르소나 Rule — 표본 크기 부족

| 규칙 | Lift | Support | 상태 |
|------|------|---------|------|
| 자연과학 × 취준생 | 1.86 | **9명** |  표본 부족 |
| 인문학 × 대학생 | 1.54 | **37명** |  표본 부족 |
| 재등록 = 예 | 1.38 | 146명 |  안정 |

자연과학×취준생은 도메인 논리(자연과학 ≈ 통계학과 → 학회에 진심인 취준생)가 타당하지만 Train 내 9명만을 근거로 한다. 통계적 신뢰도는 낮다.

### 텍스트 키워드 — incumbents_lecture_scale_reason L1=1.788
"기회", "함께" 키워드는 주로 `incumbents_lecture_scale_reason`에 등장하는데 이 컬럼의 L1=1.788로 Train/Test 분포가 크게 다르다. 해당 키워드의 일반화 가능성은 별도 검증이 필요하다.

### 소규모 데이터
Train 748명, Fold당 약 150명. 전체 결과는 대회 특정 데이터에 최적화된 측면이 있다.

---

##  분석 노트북

| 노트북 | 내용 |
|--------|------|
| [00_problem_definition](notebooks/00_problem_definition.ipynb) | 대회 개요, 데이터 설명, Target 분포 |
| [01_eda](notebooks/01_eda.ipynb) | 변수별 분포, 페르소나 분석, 텍스트 키워드 |
| [02_train_test_drift](notebooks/02_train_test_drift.ipynb) | L1 Distance 계산, 안정/불안정 변수 판별 |
| [03_lift_analysis](notebooks/03_lift_analysis.ipynb) | Lift 통계 검증, Rule 도출 |
| [04_final_model](notebooks/04_final_model.ipynb) | 최종 모델 구현 (정제·주석 버전) |

---

**Key Insight**: "Train과 Test가 다를 때, 안정적인 공통 신호를 찾고 순위로 예측하라" 
