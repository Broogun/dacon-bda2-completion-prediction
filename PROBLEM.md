# 문제 정의

##  대회 개요

### 대회 정보
- **대회명**: 데이콘 x BDA 제2회 학습자 수료 예측 AI 경진대회
- **주최**: 데이콘, BDA(Big Data Analysis)
- **참가팀**: 733팀
- **평가 지표**: F1 Score
- **최종 결과**: Public 0.44897 (43위, 상위 5.7%) / Private 0.41071 (26위, 상위 3.5%)

### BDA 소개
- 전국 대학생 연합 빅데이터 학회
- 누적 학회원 6,000명
- 전국 70개 이상 대학 네트워크
- Python, SQL 등 단계별 체계적 커리큘럼

### 대회 목표
> 9기 학습자 설문 정보를 바탕으로 10기 학습자의 수료 여부를 예측

---

##  데이터 설명

### 파일 구성

| 파일 | 크기 | 설명 |
|------|------|------|
| train.csv | 748명 (9기) | 학습 데이터 (TARGET 포함) |
| test.csv | **814명** (10기) | 예측 데이터 (TARGET 제외) |
| sample_submission.csv | 814명 | 제출 양식 |

### 주요 변수 (총 46개, TARGET 포함)

#### 1. 기본 정보
- `ID`: 샘플 고유 ID
- `generation`: BDA 기수
- `school1`: 대학교 (익명화, 숫자 코드)
- `nationality`: 내/외국인
- `completed_semester`: 대학교 이수학기

#### 2. 전공 정보
- `major_field`: **전공 분야 (대분류)** 
- `major type`: 복수전공 여부 (컬럼명에 공백 포함)
- `major1_1`: 제1전공 (세부)
- `major1_2`: 제2전공 (세부, 복수전공자만 존재)
- `major_data`: 제1전공 전공자 여부 (bool)

#### 3. 직업 및 희망 진로
- `job`: **현재 직무** 
- `desired_job`: 희망 직무
- `desired_career_path`: 희망 진로
- `desired_job_except_data`: 데이터 외 희망 직무

#### 4. 학습 관련
- **`re_registration`**: **재등록 여부** 
- **`time_input`**: **하루 BDA 투입 시간** 
- `class1~4`: 수강 분반 (class3, class4는 거의 결측)
- `previous_class_3~8`: 과거 기수 수강 분반 (80% 결측)
- `project_type`: 팀/개인 프로젝트 희망

#### 5. 자유 텍스트 / 설문
- `whyBDA`: BDA 선택 이유 
- `what_to_gain`: BDA에서 얻고싶은 것
- `incumbents_lecture_scale_reason`: 강의 규모 선택 이유
- `interested_company`: 관심 기업명 (고유값 과다)
- 현직자 강연 관련: `incumbents_level`, `incumbents_lecture`, `incumbents_company_level`, `incumbents_lecture_type`, `incumbents_lecture_scale`
- 대회/자격증 관련: `certificate_acquisition`, `desired_certificate`, `contest_participation`, `idea_contest`
- `onedayclass_topic`: 원데이 클래스 주제
- `expected_domain`: 희망 도메인

#### 6. 결측치 주의 컬럼
| 컬럼 | 결측률 | 비고 |
|------|--------|------|
| `contest_award` | **100%** | 전체 결측 — 사용 불가 |
| `idea_contest` | **100%** | 전체 결측 — 사용 불가 |
| `class4` | 99.9% | 거의 결측 |
| `contest_participation` | 99.2% | 거의 결측 |
| `class3` | 98.1% | 거의 결측 |
| `previous_class_3~8` | ~80% | 재등록자 위주로 존재 |

#### 7. Target
- **`completed`**: 수료 여부 (0: 미수료, 1: 수료)

---

##  Target 분포 (Train)

| Target | Count | Ratio |
|--------|-------|-------|
| 0 (미수료) | **525** | **70.2%** |
| 1 (수료) | **223** | **29.8%** |

> **불균형**: 미수료가 수료의 2.4배 — 클래스 불균형 심각

---

##  핵심 문제: Train/Test 이질성

### 문제 발견

**1. CV vs LB Gap**
```
CV OOF F1:    0.4885
Public LB F1: 0.3722
Gap:          -0.1163

정상 대회: ±0.01
본 대회: -0.1163 (11.6배!)

→ Train 모델이 Test에서 작동 안 함
```

**2. Threshold 불안정**
```
Test 확률 분포: 0.35~0.40 구간에 밀집

Threshold 0.005 변화
→ 선발 인원 15~25명 변동

→ Threshold 기반 예측 불안정!
```

**3. L1 Distance (실측값 기준)**

> L1 Distance = 두 분포 간 절대 차이의 합. 0에 가까울수록 Train/Test가 유사.

| 변수 | L1 Distance | 판정 |
|------|-------------|------|
| major1_1 (세부전공) | **1.947** |  사용 불가 |
| onedayclass_topic | **1.912** |  사용 불가 |
| incumbents_lecture_scale_reason | **1.788** |  사용 불가 |
| major1_2 (제2전공) | 1.550 |  사용 불가 |
| desired_job | 1.285 |  사용 불가 |
| interested_company | 1.177 |  사용 불가 |
| major_field (대분류) | 0.421 |  주의 |
| whyBDA | 0.413 |  주의 |
| inflow_route | 0.429 |  주의 |
| what_to_gain | **0.097** |  안정 |
| incumbents_lecture_type | 0.083 |  안정 |
| **job** | **0.075** |  안정 |
| **re_registration** | **0.010** |  매우 안정 |

> `school1`은 숫자형으로 범주형 L1 분석에서 제외됨 (실측 L1 ≈ 0.718, 불안정).  
> `major_field`(대분류)는 L1=0.421로 불안정하지만 Lift가 유효한 일부 카테고리(자연과학, 인문학, 법학)는 사용.  
> `major1_1`(세부전공)은 L1=1.947로 직접 사용 불가 — `major_field`를 proxy로 대신 사용.

### whyBDA 텍스트 특이사항

```
Train top 키워드: 혼자, 공부하기, 어려워서, 잘, 규모인...
Test top 키워드:  혼자, 공부하기, 어려워서, 큰, 규모인...

→ "혼자 공부하기 어려워서" 패턴이 train/test 공통으로 존재 (안정)
→ "기회", "함께" 등 긍정 키워드는 train 데이터에 실제로 존재하지 않음
```

### 결론

```
Train(9기) ≠ Test(10기)

1. 세부전공/주제 텍스트 완전히 다름 (L1 > 1.5)
2. whyBDA: 0.413 (주의 수준, 일부 패턴은 공통)
3. Small Data (748명)
4. CV 신뢰도 낮음 (Gap: -0.1163)

→ 일반적 ML 과적합 위험 매우 높음
→ 안정적 신호 기반 접근 필요
```

---

##  해결 전략

### 핵심 질문
> **"Train과 Test가 다를 때, 어떻게 예측할 것인가?"**

### 3가지 접근

**1. 안정적 신호만 사용**
```
L1 Distance < 0.2 우선 사용:
- re_registration (L1: 0.010) ← 가장 신뢰
- job (L1: 0.075)
- time_input (연속형, 분포 유사)

L1 < 0.5 중 Lift 유의미한 것:
- major_field (자연과학, 인문학, 법학 카테고리)
```

**2. 통계적 검증 (Lift 분석)**
```
4가지 조건 모두 충족한 패턴만 사용:
1. Lift ≥ 1.3 (강한 긍정 신호)
2. Support ≥ 50 (표본 충분)
3. 95% CI 안정적
4. L1 < 0.2 (일반화 가능)

→ 자연과학×취준생: support=9 (기준 미달, 불안정)
     도메인 판단: major1_1(세부전공) L1=1.947로 직접 사용 불가.
     그러나 Train 데이터에서 major_field='자연과학'의 세부전공을 확인하면
     거의 전원이 통계학과임을 확인. 자연과학×취준생 = "통계학과 출신 취준생"
     → 데이터 학회에 진심인 사람의 proxy. 표본(9명)이 적어 통계적 신뢰도는 낮지만
     도메인 논리가 타당하여 실험적으로 적용.
→ 재등록=예: Lift 1.38, Support 146 (4조건 통과, 가장 신뢰할 수 있는 신호)
→ 인문학×대학생: Lift 1.54, Support 37 (지원 불충분)
```

**3. 순위 기반 예측 (Flip-rate)**
```
Threshold 방식 (X):
  확률 분포 밀집 → 불안정

Flip-rate 방식 (O):
  순위 기반 Top-k 선발 → 안정
  k=375명 (46.1%)


```

---

##  평가 지표

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**본 대회 특징**:
- 확률 분포 밀집 (0.35~0.40) → Threshold에 극도로 민감
- Flip-rate(Top-k)로 해결: 선발 인원 고정 → 안정

---

**다음**: [실험 기록 (EXPERIMENTS.md)](EXPERIMENTS.md)
