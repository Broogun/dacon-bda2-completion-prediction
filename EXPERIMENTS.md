# 실험 기록

데이콘 x BDA 제2회 학습자 수료 예측 AI 경진대회 전체 실험 기록.

**총 제출 횟수**: ~100회  
**기간**: 2026-01-13 ~ 2026-02-21  
**최고 점수**: Public 0.44897 → 43위 / 733팀 (상위 5.7%) → Private 26위 (상위 **3.5%**)

---

##  전체 흐름 요약

```
Phase 0 │ Jan 13      │ 0.310  LGBM 첫 제출
        │             │
Phase 1 │ Jan 25~28   │ 0.34~0.37  CatBoost + Threshold 튜닝
        │             │
Phase 2 │ Jan 29~Feb 8│ 0.38~0.44  Rule Discovery (페르소나 + Kill)
        │             │
Phase 3 │ Feb 9~Feb 20│ 0.44~0.449 NLP_V2 정밀 튜닝 → 최고점
        │             │
EXP     │ Feb 21      │ 0.439  Semester-only 실험 (참고용)
```

---

## Phase 0: 첫 시도 (Jan 13)

| 파일 | 점수 | 비고 |
|------|------|------|
| submission_lgbm_f1_th0.14.csv | 0.310 | 첫 제출, F1 threshold=0.14 |
| submission_lgbm_f1_th0.05.csv | 0.310 | threshold 조정, 변화 없음 |

**결론**: LGBM 단독으로는 0.31 수준이 한계. CV F1 0.49 vs LB 0.31 — 심각한 Gap.

---

## Phase 1: CatBoost + Threshold 튜닝 (Jan 25~28)

### 전략
> "CatBoost가 범주형에 강하다 → pos_rate sweep으로 최적 threshold 찾기"

| 파일 | 점수 | 메모 |
|------|------|------|
| submission_catboost.csv | 0.288 | CatBoost 첫 시도 |
| sub_pr020.csv | 0.290 | pos_rate 20% |
| sub_pr030.csv | 0.318 | pos_rate 30% |
| sub_pr035.csv | 0.357 | pos_rate 35% |
| submission_best_pr035.csv | 0.343 | 그리드 최적화 |
| submission_fit_probe.csv | 0.160 | 실패 (threshold 너무 낮음) |
| submission_pr040_best.csv | 0.342 | |
| submission_best_global_th.csv | 0.345 | |
| submission_global_th_fixed.csv | 0.372 | **Phase 1 최고** |
| submission_TE_LGBM_th025.csv | 0.379 | Target Encoding 시도 |
| submission_TE_best_0.5126_pr0.50.csv | 0.372 | |
| submission_TE_LGBM_bestguess_th025.csv | 0.379 | |
| submission_fulltrain_SAFE_TE.csv | 0.348 | 전체 훈련 후 예측 |

**핵심 발견**:
- CatBoost 단독 최고: 0.372
- CV OOF F1: ~0.49 vs LB: 0.37 → **Gap -0.12** (이질성 확인)
- Threshold 0.005 변화 → 선발 인원 15~25명 변동 (불안정)
- Target Encoding은 개선 효과 미미

**의사결정**: Threshold 기반 접근 한계. 방향 전환 필요.

---

## Phase 2: Rule Discovery (Jan 29 ~ Feb 12)

### 전략 전환
```
기존: 복잡한 ML → Train에 과적합
새로운: 안정적 신호 발견 + Rule 기반

핵심 아이디어:
1. L1 Distance로 안정 변수 선별
2. Lift 분석으로 통계 검증
3. 순위 기반 선발 (Flip-rate)
```

### 2-A: Ensemble + 기본 Rule (Jan 29 ~ Feb 2)

| 파일 | 점수 | 변경 사항 |
|------|------|----------|
| submission_Final_Base395_Rule.csv | 0.404 | CatBoost+LGBM 앙상블 + 기본 Rule |
| submission_Final_Base385_Rule.csv | 0.401 | |
| submission_Final_Base400_Rule.csv | 0.401 | |
| submission_Final_Base405_Rule.csv | 0.404 | |
| submission_ULTIMATE_Ensemble_Top375.csv | 0.383 | 순위 기반 시도 |
| submission_Final_OneShot.csv | **0.418** | **페르소나 Rule 첫 성공** |
| submission_SVD_Features_Base395.csv | 0.388 | 텍스트 SVD 피처 → 효과 없음 |
| submission_Final_GPT_Agent.csv | 0.390 | GPT 점수 활용 시도 |
| submission_Final_Safe_AddOnly.csv | 0.416 | |
| submission_Final_Hidden_Curse.csv | 0.408 | |
| submission_Final_Smart_Filter.csv | 0.414 | |
| submission_Final_Domain_Adaptation.csv | 0.399 | Domain Adaptation 시도 |
| submission_Final_Pseudo_Labeling.csv | 0.334 | Pseudo Labeling → 역효과 |
| submission_Tomorrow_Hybrid.csv | 0.393 | |
| submission_Final_Fixed_383.csv | 0.373 | |
| submission_Final_Hidden_Gem_Integrated.csv | 0.400 | |
| submission_Final_Refined_383.csv | **0.441** | **NLP 점수 첫 도입** |

**핵심 발견**:
- 페르소나 Rule 도입 → 0.404로 큰 도약 (+0.03)
- Pseudo Labeling: 0.334 → **실패** (이질성이 큰 데이터에서 역효과)
- SVD 텍스트 피처: 0.388 → **실패** (불안정 변수 사용)
- NLP 점수 도입 시 0.441 → 새로운 최고점

### 2-B: 페르소나 + Kill Rule 정밀화 (Feb 3 ~ Feb 8)

| 파일 | 점수 | 변경 사항 |
|------|------|----------|
| submission_Final_Polished_LGBM.csv | 0.420 | LGBM 정밀화 |
| submission_Final_Masterpiece_Fixed.csv | 0.408 | |
| **ML_FEATURE_VER_384.csv** | **0.384** | **순수 ML 피처만 사용 → 실패** |
| ULTIMATE_REFINED_383.csv | 0.404 | |
| FINAL_PURE_KILL_383.csv | **0.441** | Hard Kill 적용 |
| FINAL_CLEAN_384.csv | **0.441** | Clean 버전 |
| FINAL_0441_UPGRADED_384.csv | 0.430 | |
| submission_Final_Refined_375.csv | 0.435 | 선발 인원 375명 시도 |
| submission_Final_Refined_380.csv | **0.439** | 선발 인원 380명 |
| submission_Final_Refined_386.csv | 0.441 | |
| submission_BackToBasic_386.csv | 0.412 | |
| submission_Final_Master_Complete.csv | 0.423 | |

> **중요한 실패 기록: ML_FEATURE_VER_384 (0.384)**
>
> ML 피처(클래스 진행도, 결측률, 텍스트 길이 등)만 사용하고 Rule Score를 제거한 실험.
> 결과: **0.384** — Phase 2 중 최저점. 순수 ML 접근이 Rule 기반보다 현저히 낮음.
>
> **교훈**: 이 데이터셋에서 일반적 ML 피처보다 도메인 기반 Rule이 더 강한 신호.

### 2-C: NLP 점수 + Kill Rule 조합 (Feb 9 ~ Feb 12)

| 파일 | 점수 | 변경 사항 |
|------|------|----------|
| submission_eda_383.csv | 0.407 | EDA 기반 |
| submission_eda_363.csv | 0.400 | 363명 선발 |
| FINAL_PATCHED_383.csv | 0.420 | |
| submission_lgbm_cat_383.csv | 0.428 | LGBM+CatBoost 앙상블 |
| submission_safe_383.csv | 0.419 | |
| submission_catboost_baseline.csv | 0.329 | CatBoost 단독 (재확인) |
| FINAL_CLASS1_SIGNAL_383.csv | **0.426** | class1 신호 활용 |
| FINAL_META_CATBOOST_383.csv | 0.397 | Meta CatBoost → 효과 없음 |
| FINAL_NLP_PYTHON_SQL_383.csv | 0.403 | Python/SQL 키워드 추가 → 악화 |
| FINAL_LAW_KILL_FIX_380.csv | 0.436 | 법학 Kill 버그 수정 |
| FINAL_LAW_KILL_FIX_383.csv | **0.441** | |
| FINAL_LAW_KILL_FIX_386.csv | 0.440 | |
| FINAL_KEYWORD_REFINED_383.csv | 0.426 | 키워드 정밀화 |

**핵심 발견**:
- 법학 Kill 버그 수정 (`na=False` 추가) → 안정적 0.441
- FINAL_META_CATBOOST: 0.397 → **실패** (복잡한 메타 구조 역효과)
- Python/SQL 키워드 추가: 0.403 → **실패** (과적합)
- 선발 인원별 최적값 탐색 시작: 383 ≈ 380 >> 386

---

## Phase 3: NLP_V2 정밀 튜닝 (Feb 13 ~ Feb 20)

### NLP_V2 핵심 로직

```python
def get_refined_score(row):
    text = whyBDA + incumbents_lecture_scale_reason

    # 페르소나 점수 (안정 변수 기반)
    if major_field == "자연과학" and job == "취준생": score += 3
    if major_field == "인문학" and job == "대학생":   score += 2

    # 희망 직무 신호
    if "데이터 분석가" in desired_job and "온라인" in lecture_type: score += 2

    # 키워드 (실제 데이터에 존재하는 것만)
    if "함께" in text: score += 1
    if "기회" in text: score += 3
    if "경험" in text: score -= 2

    # 맥락 기반 부정 (중복 감점 방지)
    positive_context = ["기회", "함께", "도움", "필요", "배우", "성장", "발전"]
    has_positive = any(w in text for w in positive_context)
    if ("어려워" in text or "혼자" in text) and not has_positive:
        score -= 2  # -4점이었다가 -2점으로 수정

    # 기타
    if inflow == "인스타그램" and text_len < 30: score -= 3
    if time_input == 4.0:       score += 2
    if re_registration == "예": score += 2
    if school in ['45','38','52']: score += 2
    if school in ['81','54','19']: score -= 5

    return score
```

| 파일 | 점수 | 선발 | 변경 사항 |
|------|------|------|----------|
| FINAL_NLP_V2_383.csv | 0.443 | 383명 | NLP_V2 기본 |
| FINAL_NLP_V2_380.csv | 0.447 | 380명 | ↑ 선발 줄임 |
| FINAL_NLP_V2_378.csv | 0.447 | 378명 | |
| **FINAL_NLP_V2_375.csv** | **0.44897** | **375명** | **최고점** |
| **FINAL_NLP_V2_376.csv** | **0.44897** | **376명** | **동점** |
| FINAL_NLP_V2_373.csv | 0.445 | 373명 | |
| FINAL_NLP_V2_370.csv | 0.447 | 370명 | |
| FINAL_NLP_V2_365.csv | 0.448 | 365명 | |
| FINAL_SCHOOL_IMPROVED_375.csv | 0.418 | 375명 | 학교 신호 강화 → 악화 |
| FINAL_NLP_JOBSEEKER_375.csv | 0.435 | 375명 | 취준생 신호 강화 |

**선발 인원별 최적값 분석**:
```
365: 0.4483
370: 0.4467
373: 0.4452
375: 0.44897 ← 최고
376: 0.44897 ← 동점
378: 0.4475
380: 0.4475
383: 0.4430
```
→ **375~376명이 최적**. 380명 이상부터 감소.

---

## 참고 실험: EXP 시리즈 (Feb 21)

> Phase 3 이후 대회 종료 전 추가 실험. 최고점 돌파 실패.

| 파일 | 점수 | 내용 |
|------|------|------|
| EXP1_NO_PERSONA_375.csv | 0.401 | 페르소나 제거 → 큰 하락 확인 |
| EXP3_NEW_PERSONA_375.csv | 0.406 | 새 페르소나 시도 |
| EXP4_SELECTIVE_REMOVE_375.csv | 0.418 | 선택적 Rule 제거 |
| EXP5_ALL_SUPER_SIGNALS_375.csv | 0.417 | 모든 신호 추가 → 과적합 |
| **EXP7_SEMESTER2_ONLY_375.csv** | **0.439** | 2학기 이후 데이터만 사용 |

**EXP7 분석**: completed_semester ≥ 2인 학생만 사용했을 때 0.439. 이수학기가 수료 예측에 강한 신호임을 시사하지만 최고점(0.449)에 미달.

---

##  실험 결과 종합 분석

### 접근별 성능 비교

| 접근 방식 | 최고 점수 | 대표 파일 |
|-----------|-----------|-----------|
| LGBM 단독 | 0.310 | submission_lgbm_f1 |
| CatBoost 단독 | 0.288 | submission_catboost |
| CatBoost + Threshold 튜닝 | 0.372 | submission_global_th_fixed |
| Target Encoding + LGBM | 0.379 | submission_TE_LGBM |
| LGBM + CatBoost 앙상블 | 0.428 | submission_lgbm_cat_383 |
| **순수 ML 피처** | **0.384** | **ML_FEATURE_VER_384** |
| Pseudo Labeling | 0.334 | submission_Final_Pseudo_Labeling |
| SVD 텍스트 피처 | 0.388 | submission_SVD_Features |
| Rule v1 (기본 페르소나) | 0.418 | submission_Final_OneShot |
| Rule + 앙상블 (NLP 없음) | 0.441 | FINAL_CLEAN_384 |
| **NLP_V2 + 5-Seed 앙상블** | **0.44897** | **FINAL_NLP_V2_375** |

### 주요 도약 지점

```
0.310 → 0.372  (+0.062)  CatBoost + threshold 튜닝
0.372 → 0.418  (+0.046)  페르소나 Rule 첫 도입
0.418 → 0.441  (+0.023)  NLP 점수 + Hard Kill
0.441 → 0.449  (+0.008)  5-Seed 앙상블 + 선발 인원 최적화
```

### 실패 패턴 분석

| 시도 | 점수 변화 | 이유 |
|------|----------|------|
| 순수 ML 피처 | 0.441 → 0.384 | 불안정 변수 사용, 과적합 |
| Pseudo Labeling | 0.418 → 0.334 | Train/Test 이질성 증폭 |
| SVD 텍스트 피처 | baseline → 0.388 | 고이질성 변수(L1>1) 활용 |
| GPT 점수 활용 | baseline → 0.390 | 일반화 어려움 |
| Python/SQL 키워드 | 0.441 → 0.403 | 과도한 feature engineering |
| 학교 신호 강화 | 0.449 → 0.418 | school1은 L1 높음 (불안정) |
| 페르소나 제거(EXP1) | 0.449 → 0.401 | 페르소나가 핵심 신호임 재확인 |

---

##  핵심 인사이트

### 1. 순수 ML은 역효과였다
`ML_FEATURE_VER_384` (0.384)는 Phase 전체에서 가장 낮은 "진지한" 시도 결과.  
클래스 진행률, 결측률 등 일반적 ML 피처는 이 데이터셋에서 작동하지 않음.  
→ **Train/Test 이질성이 클 때 복잡한 ML은 과적합.**

### 2. 안정 변수 기반 Rule이 핵심
`re_registration` (L1=0.010), `time_input` (분포 유사), `major_field` 일부 카테고리.  
이 변수들만으로 구성된 Rule이 ML보다 강력했음.

### 3. Hard Kill의 효과
법학/경영학×직장인 Hard Kill이 일관되게 +0.001~0.002 기여.  
소수 집단의 극단적 패턴을 통계적으로 확인 후 적용하는 것이 유효.

### 4. 선발 인원이 점수에 직접 영향
F1 Score 특성상 Precision/Recall 균형이 중요.  
375-376명이 최적 (814명의 46.1%). 1등 팀의 46.9%와 근접.

### 5. 키워드는 whyBDA만이 아닌 combined text 기준
`get_refined_score()`는 `whyBDA + incumbents_lecture_scale_reason` 합산 텍스트를 사용한다.  
"기회", "함께"는 `incumbents_lecture_scale_reason`에 등장하며, 03_lift_analysis 노트북도 combined text 기준으로 분석한다.  
다만 해당 컬럼의 L1=1.788로 불안정하므로, 이 키워드의 효과가 얼마나 일반화되는지는 별도 검증이 필요하다.

### 6. 모델 구조 설계 근거
- **CatBoost + LightGBM**: 귀납적 편향(inductive bias)이 달라 오류가 겹치지 않음. CatBoost는 범주형 34개 네이티브 처리, LightGBM은 수치형 패턴 담당.
- **OOF Meta LR 스태킹**: Out-of-Fold 예측으로 학습 → Train → Meta 방향 리키지 없음. LR 파라미터 수 극소 → 748명에서 과적합 위험 없음.
- **5-Seed 평균**: 748명 소규모에서 단일 시드의 Fold 배정이 결과에 과도하게 영향. 5개 시드로 분산 흡수.
- **도메인 Rule 우선**: 순수 ML(0.384) < Rule + ML(0.449). Train/Test 이질성이 클 때 Rule이 ML보다 일반화가 강함.  
  모든 Rule의 근거: "기수가 달라도 변하지 않는 인간의 동기 구조".

---

## 최종 결과

| 리더보드 | 점수 | 순위 | 백분위 |
|---------|------|------|--------|
| Public | **0.44897** | **43위** / 733팀 | 상위 5.7% |
| Private | **0.41071** | **26위** / 733팀 | **상위 3.5%** |

Public 43위 → Private 26위: **+17등 상승**

Public에서 Private로 갈 때 절대 점수는 하락했지만, 이는 대부분의 참가자에게 공통으로 발생한 현상이다. 상대 순위가 17등 오른 것은 이 접근법이 Public에 덜 과적합됐다는 의미다. 안정 변수(re_registration 등) 기반 Rule과 Flip-rate가 일반화 측면에서 유효했음을 Private 결과가 사후 검증해준 것으로 볼 수 있다.

> 대회 측 수료 기준 조정: 0.42 → 0.38 (Train/Test 이질성 문제 인정)
> Private 점수 0.411 → 조정 기준 상회

---

##  현재 리팩토링 방향 (참고)

원본 코드(test1.ipynb, test2.ipynb)의 NLP_V2 접근을 ML 파이프라인으로 재구성.

| 원본 | 리팩토링 |
|------|---------|
| `get_refined_score()` 수동 점수 | Decision Tree 자동 Rule 추출 |
| Hard Kill 하드코딩 | config.yaml 기반 |
| 단일 스크립트 | run_rule_discovery → run_ensemble → run_stacking |
| 5-seed × 5-fold (스크립트 내) | 분리된 모듈 |

>  **주의**: 순수 ML로 바꿀수록 0.384 수준으로 회귀할 위험이 있음.  
> Rule Score (재등록, 페르소나, Kill)를 ML 파이프라인에 반드시 유지해야 함.

---

**원본 실험 코드**: `test1.ipynb`, `test2.ipynb` (아카이브, 수정 없음)  
**정제 버전**: `notebooks/04_final_model.ipynb` — 동일 NLP_V2 로직을 주석과 함께 재현한 클린 노트북
